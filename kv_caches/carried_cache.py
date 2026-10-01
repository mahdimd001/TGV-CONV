"""What lmms_eval/models/model_utils/kv_carryover.py needs from a KV cache method carried across the turns of a conversation.

Budget: with fixed_budget=True (MULTI_TURN_BUDGET=fixed, the default) every prefill compresses to the budget set at
turn 1's prefill, (1 - ratio) x the turn-1 prompt length, so a long answer does not buy the next turn more cache; with
False it is (1 - ratio) x the whole conversation so far. Methods ask `_target(num_of_token)` for the entries to keep.

A method subclasses CarriedCache and implements
    _step(past_key_values, num_of_token, attentions, prefill, evict) -> (cache, kept)
where `kept` gives, per layer, the indices of the entries kept (None: every entry kept). `prefill` is True when this
call follows a forward pass over several new tokens (turn 1: the prompt; later turns: the new question on top of the
carried cache); `evict` is False for decode steps when MULTI_TURN_DECODE_EVICTION=0 (statistics may still be updated).

The mixin tracks the conversation length the cache stands for (`logical_len`), one image/text flag per cached entry
and layer (`is_image`, for the debug log), the entries right after the last prefill (`after_prefill`) and the error a
call raised (`error`, because generate() swallows them).
"""

from typing import List, Optional, Tuple

import torch


class CarriedCache:
    def _init_carried(self, layer_num: int, image_token_id: Optional[int], decode_eviction: bool, fixed_budget: bool = True) -> None:
        self.layer_num = layer_num
        self.fixed_budget = fixed_budget
        self.budget: Optional[float] = None  # entries per layer kept at turn 1's prefill, (1 - ratio) x turn-1 prompt
        self.last_target: Optional[float] = None  # entries per layer the last prefill compressed to (for the log)
        self.image_token_id = image_token_id
        self.decode_eviction = decode_eviction
        self.is_image: Optional[List[torch.Tensor]] = None
        self.logical_len = 0
        self.after_prefill: Optional[Tuple[float, float]] = None
        self.error: Optional[Exception] = None

    def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        try:
            if past_key_values is None:
                return None
            new_flags = input_ids[0, self.logical_len : num_of_token] == self.image_token_id  # tokens this forward processed
            prefill = new_flags.numel() > 1
            if prefill and self.budget is None:
                self.budget = num_of_token * (1 - self.ratio)
            if prefill:
                self.last_target = self._target(num_of_token)
            old = self.is_image or [new_flags[:0]] * self.layer_num
            flags = [torch.cat([f, new_flags.to(f.device)]) for f in old]
            if any(p[0].size(-2) != f.numel() for p, f in zip(past_key_values, flags)):
                raise RuntimeError(f"{type(self).__name__}: cache and image flags are out of sync")
            out, kept = self._step(past_key_values, num_of_token, attentions, prefill=prefill, evict=prefill or self.decode_eviction)
            self.is_image = flags if kept is None else [f[idx.to(f.device)] for f, idx in zip(flags, kept)]
            if prefill:
                image = sum(int(f.sum()) for f in self.is_image) / self.layer_num
                self.after_prefill = (image, sum(f.numel() for f in self.is_image) / self.layer_num - image)
            self.logical_len = num_of_token
            return out
        except Exception as e:  # generate() swallows criteria errors; keep it so the caller can raise
            self.error = e
            raise


    def _target(self, num_of_token: int) -> float:
        """Entries per layer to keep: the turn-1 budget (fixed) or (1 - ratio) x the conversation so far."""
        if self.fixed_budget and self.budget is not None:
            return self.budget
        return num_of_token * (1 - self.ratio)


def gather_entries(past_key_values, kept: List[torch.Tensor]):
    """A DynamicCache with, in every layer, only the entries at `kept[layer]` (in that order)."""
    from transformers.cache_utils import DynamicCache

    out = []
    for (k, v), idx in zip(past_key_values, kept):
        index = idx.to(k.device).view(1, 1, -1, 1)
        out.append([k.gather(-2, index.expand(k.shape[0], k.shape[1], -1, k.shape[-1])), v.gather(-2, index.expand(v.shape[0], v.shape[1], -1, v.shape[-1]))])
    return DynamicCache(out)


def without_index(seq_len: int, index: int, device=None) -> torch.Tensor:
    """All entry indices except `index`."""
    keep = torch.arange(seq_len, device=device)
    return keep[keep != index]
