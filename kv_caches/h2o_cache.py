"""H2O (heavy-hitter oracle), as implemented in github.com/liuzuyan/ElasticCache (kv_cache.py, H2OCache).

Every cached entry carries an accumulated attention score: the attention it receives, averaged over heads and over
layers, summed over every query processed so far (prompt tokens and generated tokens). At the prefill the entries with
the highest scores are kept up to num_of_token * (1 - ratio), plus `start_size` sinks and the newest entry; while
decoding, whenever the cache is over budget the entry with the lowest accumulated score is evicted. Scores are averaged
over layers, so every layer keeps the same entries.

selection="paper" (default) keeps the highest-scoring entries and keeps each score aligned with its entry.
selection="official" reproduces the original code exactly, including two bugs: it ranks with a single argsort
(`argsort(score) > forget_num`), which does not keep the highest-scoring entries, and it does not compact its score
buffer after the prefill, so during decoding the scores belong to different entries than the ones in the cache. The
original prefill also ignores `start_size` (it is used only while decoding).

MultiTurnH2OCache (selection="paper" only) carries the cache and the accumulated scores across the turns of a
conversation: each prefill adds the new question's attention to the carried scores and keeps the top entries down to
the budget (fixed at turn 1 by default, see carried_cache.py); decode steps keep accumulating scores and evict only with decode_eviction=True.
"""

from typing import Optional

import torch

from .carried_cache import CarriedCache, gather_entries, without_index


class H2OCache:
    supports_online_prefill = True  # the prompt's scores are collected during the forward pass (eager attention)

    def __init__(self, layer_num, start_size=1, recent_size=2047, k_seq_dim=2, v_seq_dim=2, ratio=0.0, selection="paper", **kwargs):
        if selection not in ("paper", "official"):
            raise ValueError(f"H2O selection must be 'paper' or 'official', got {selection!r}")
        self.layer_num = layer_num
        self.start_size = start_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.ratio = ratio
        self.selection = selection
        self.protect_size = 1
        self.scores: Optional[torch.Tensor] = None  # paper: one accumulated score per cached entry
        self.score_sum = torch.zeros(start_size + recent_size + 1)  # official: the original fixed-size buffer
        self.flag = True  # official: the original's "prefill not done yet"
        self._layer_scores = None

    # ---------------------------------------------------------------- online prefill statistics

    def begin_online_prefill(self, input_ids) -> bool:
        self._layer_scores = [None] * self.layer_num
        return True

    def collect_online_prefill_attention(self, layer_idx, attention) -> None:
        if self._layer_scores is not None and attention is not None:
            self._layer_scores[layer_idx] = attention[0].float().mean(0).sum(0)  # received attention: [keys]

    def finish_online_prefill(self):
        scores, self._layer_scores = self._layer_scores, None
        missing = [i for i, s in enumerate(scores or []) if s is None]
        if scores is None or missing:
            raise RuntimeError(f"Missing H2O prefill statistics for layers: {missing}")
        return {"h2o_online_prefill": True, "scores": torch.stack([s.to(scores[0].device) for s in scores]).mean(0)}

    # ---------------------------------------------------------------- compression

    def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        if past_key_values is None:
            return None
        prefill = isinstance(attentions, dict) or (attentions is not None and attentions[0] is not None and attentions[0].shape[-2] > 1)
        if self.selection == "official":
            return self._official(past_key_values, num_of_token, attentions, prefill)
        cache, _ = self._step(past_key_values, num_of_token, attentions, prefill=prefill, evict=True)
        return cache

    def _step(self, past_key_values, num_of_token, attentions, prefill: bool, evict: bool):
        received = _received_attention(attentions)  # layer-mean attention each key received in this forward pass
        old = self.scores if self.scores is not None else received[:0]
        self.scores = torch.cat([old.to(received.device), received.new_zeros(received.numel() - old.numel())]) + received
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - self._target(num_of_token))
        if not evict or forget_num <= 0:
            return past_key_values, None
        start, end = self.start_size, seq_len - self.protect_size
        if prefill:  # keep the highest accumulated scores
            n_keep = max(end - start - forget_num, 0)
            order = torch.argsort(self.scores[start:end], descending=True, stable=True)
            keep = torch.cat([torch.arange(start, device=order.device), order[:n_keep].sort().values + start, torch.tensor([end], device=order.device)])
        else:  # evict the lowest accumulated score
            if start >= end:
                return past_key_values, None
            keep = without_index(seq_len, int(self.scores[start:end].argmin()) + start, self.scores.device)
        self.scores = self.scores[keep]
        kept = [keep] * len(past_key_values)
        return gather_entries(past_key_values, kept), kept

    def _target(self, num_of_token: int) -> float:
        """Entries to keep (single-turn: the original rule, (1 - ratio) x all tokens so far)."""
        return num_of_token * (1 - self.ratio)

    def _official(self, past_key_values, num_of_token, attentions, prefill):
        """The original H2OCache.__call__, unchanged except for device handling, online statistics and DynamicCache."""
        received = _received_attention(attentions).to(self.score_sum.device)
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if prefill:
            assert self.flag is True  # only use for the first time
            self.score_sum[: received.numel()] += received
        else:
            self.score_sum[:seq_len] += received
        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        device = past_key_values[0][0].device
        if forget_num > 1:
            assert self.flag is True
            self.flag = False
            selected = torch.where(torch.argsort(self.score_sum[: (seq_len - self.protect_size)]) > forget_num)[0]
            keep = torch.cat([selected, torch.arange(seq_len - self.protect_size, seq_len)])
        else:
            index = int(self.score_sum[self.start_size : (seq_len - self.protect_size)].argmin()) + self.start_size
            self.score_sum[index:-1] = self.score_sum[index + 1 :].clone()
            keep = without_index(seq_len, index)
        return gather_entries(past_key_values, [keep.to(device)] * len(past_key_values))


class MultiTurnH2OCache(CarriedCache, H2OCache):
    def __init__(self, *args, layer_num=None, image_token_id=None, decode_eviction=False, fixed_budget=True, **kwargs):
        H2OCache.__init__(self, *args, layer_num=layer_num, **kwargs)
        if self.selection != "paper":
            raise ValueError("H2O_SELECTION=official reproduces the original single-turn code; multi-turn runs need H2O_SELECTION=paper.")
        self._init_carried(layer_num, image_token_id, decode_eviction, fixed_budget)

    __call__ = CarriedCache.__call__


def _received_attention(attentions) -> torch.Tensor:
    """Attention each key received in this forward pass, averaged over heads and layers and summed over queries."""
    if isinstance(attentions, dict):
        if not attentions.get("h2o_online_prefill"):
            raise RuntimeError("H2O got another method's prefill statistics")
        return attentions["scores"]
    if attentions is None or len(attentions) == 0 or attentions[0] is None:
        raise RuntimeError("H2O needs the attention weights: load the model with attn_implementation=eager.")
    return torch.stack([a[0].float().mean(0).sum(0).to(attentions[0].device) for a in attentions]).mean(0)
