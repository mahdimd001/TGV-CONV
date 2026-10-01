"""Local cache (StreamingLLM-style sliding window), as in github.com/liuzuyan/ElasticCache (kv_cache.py, LocalCache).

Keeps the first `start_size` entries (attention sinks) and the most recent entries; whenever the cache holds more than
num_of_token * (1 - ratio) entries, the oldest entries after the sinks are dropped. The same rule runs at the prefill
and at every decode step, and it needs no attention weights.

MultiTurnLocalCache carries the window across the turns of a conversation: each prefill (question k on top of the
carried cache) drops the oldest entries down to the budget (fixed at turn 1 by default, see carried_cache.py); while decoding it drops only with
decode_eviction=True (MULTI_TURN_DECODE_EVICTION=1).
"""

import torch

from .carried_cache import CarriedCache, gather_entries


class LocalCache:
    def __init__(self, layer_num=None, start_size=1, k_seq_dim=2, v_seq_dim=2, ratio=0.0, **kwargs):
        self.layer_num = layer_num
        self.start_size = start_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.ratio = ratio

    def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        if past_key_values is None:
            return None
        cache, _ = self._step(past_key_values, num_of_token, attentions, prefill=False, evict=True)
        return cache

    def _step(self, past_key_values, num_of_token, attentions, prefill: bool, evict: bool):
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - self._target(num_of_token))
        if not evict or forget_num <= 0:
            return past_key_values, None
        device = past_key_values[0][0].device
        keep = torch.cat([torch.arange(min(self.start_size, seq_len), device=device), torch.arange(forget_num + self.start_size, seq_len, device=device)])
        kept = [keep] * len(past_key_values)
        return gather_entries(past_key_values, kept), kept

    def _target(self, num_of_token: int) -> float:
        """Entries to keep (single-turn: the original rule, (1 - ratio) x all tokens so far)."""
        return num_of_token * (1 - self.ratio)


class MultiTurnLocalCache(CarriedCache, LocalCache):
    def __init__(self, *args, layer_num=None, image_token_id=None, decode_eviction=False, fixed_budget=True, **kwargs):
        LocalCache.__init__(self, *args, layer_num=layer_num, **kwargs)
        self._init_carried(layer_num, image_token_id, decode_eviction, fixed_budget)

    __call__ = CarriedCache.__call__
