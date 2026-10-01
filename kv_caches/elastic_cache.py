"""Elastic Cache (Liu et al., ECCV 2024, https://github.com/liuzuyan/ElasticCache) for TGV-KV's generate patches.

Prefill ("cache merging"): in every layer, a cached entry's importance is the attention it receives from the
tokens processed at the prefill (averaged over heads, summed over queries). Each layer keeps the most important
entries up to the budget, plus `start_size` sink entries and the newest entry; every other entry is merged
(averaged) into its nearest kept entry. All layers get the same budget.
Decode ("fixed-point elimination"): whenever the cache is over budget, the entry at one fixed index is deleted:
the compressed prompt length + `distance` (default -25).

Budget: keep num_of_token * (1 - ratio) entries, the rule of the original code (`ratio` = PRUNE_RATIO).

selection="paper" (default) keeps the highest-importance entries, as the paper describes. selection="official"
reproduces the original code exactly, which ranks with a single argsort (`argsort(score) > forget_num`) and so
does not keep the highest-importance entries unless importance falls monotonically with position.

MultiTurnElasticCache carries the cache across the turns of one conversation (see
lmms_eval/models/model_utils/kv_carryover.py): every prefill (question k on top of the carried cache) merges the
whole cache again, scored by the attention from the newly processed tokens, to (1 - ratio) x the conversation
length. Fixed-point elimination runs while answers are generated only with decode_eviction=True.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from transformers.cache_utils import DynamicCache

from .carried_cache import CarriedCache, without_index


class ElasticCache:
    supports_online_prefill = True  # per-layer importance is computed during the forward pass (eager attention)

    def __init__(self, layer_num, start_size=1, k_seq_dim=2, v_seq_dim=2, ratio=0.0, distance=-25, selection="paper", **kwargs):
        if selection not in ("paper", "official"):
            raise ValueError(f"Elastic Cache selection must be 'paper' or 'official', got {selection!r}")
        self.layer_num = layer_num
        self.start_size = start_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.ratio = ratio
        self.distance = distance
        self.selection = selection
        self.protect_size = 1  # the newest entry is always kept
        self.fixed_point: Optional[int] = None  # index deleted at each over-budget decode step
        self._scores: Optional[List[Optional[torch.Tensor]]] = None

    # ---------------------------------------------------------------- online prefill statistics

    def begin_online_prefill(self, input_ids) -> bool:
        self._scores = [None] * self.layer_num
        return True

    def collect_online_prefill_attention(self, layer_idx: int, attention: torch.Tensor) -> None:
        if self._scores is not None and attention is not None:
            self._scores[layer_idx] = _importance(attention)

    def finish_online_prefill(self):
        scores, self._scores = self._scores, None
        missing = [i for i, s in enumerate(scores or []) if s is None]
        if scores is None or missing:
            raise RuntimeError(f"Missing Elastic Cache prefill statistics for layers: {missing}")
        return {"elastic_online_prefill": True, "scores": tuple(scores)}

    # ---------------------------------------------------------------- compression

    def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        if past_key_values is None:
            return None
        _require_attention(attentions)
        scores = _prefill_scores(attentions)
        if scores is not None:
            cache, _ = self._prefill(past_key_values, num_of_token, scores)
            return cache
        cache, _ = self._decode(past_key_values, num_of_token)
        return cache

    def _prefill(self, past_key_values, num_of_token, scores) -> Tuple[object, Optional[List[torch.Tensor]]]:
        """Merges every layer down to the budget; returns the cache and, per layer, the kept indices (None: all kept)."""
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - self._target(num_of_token))
        if forget_num <= 0:
            self.fixed_point = self._fixed_point(seq_len, 0)
            return past_key_values, None

        cache, kept = [], []
        for idx in range(self.layer_num):
            k, v = past_key_values[idx]
            keep, throw, anchors = self._split(scores[idx].to(k.device), k.size(self.k_seq_dim), forget_num)
            k, v = _merge(k, throw, anchors), _merge(v, throw, anchors)
            index = keep.view(1, 1, -1, 1)
            cache.append([k.gather(-2, index.expand(k.shape[0], k.shape[1], -1, k.shape[-1])), v.gather(-2, index.expand(v.shape[0], v.shape[1], -1, v.shape[-1]))])
            kept.append(keep)
        self.fixed_point = self._fixed_point(seq_len, forget_num)
        return DynamicCache(cache), kept

    def _split(self, score: torch.Tensor, seq_len: int, forget_num: int):
        """(kept indices incl. sinks and the newest entry, indices merged away, the kept index each one merges into)."""
        device = score.device
        start, end = self.start_size, seq_len - self.protect_size
        candidates = score[start:end]
        if self.selection == "official":  # the original code, unchanged
            order = torch.argsort(candidates)
            selected = torch.where(order > forget_num)[0] + start
            throw = torch.where(order <= forget_num)[0]  # (no start offset in the original)
        else:  # paper: the highest-importance candidates, as many as the budget allows
            n_keep = max(end - start - forget_num, 0)
            order = torch.argsort(candidates, descending=True, stable=True)
            selected = order[:n_keep].sort().values + start
            throw = order[n_keep:].sort().values + start
        if selected.numel() and throw.numel():
            anchors = selected[(selected[None, :] - throw[:, None]).abs().argmin(dim=1)]  # nearest kept entry
        else:  # nothing kept to merge into: dropped entries are discarded
            throw = anchors = throw[:0]
        keep = torch.cat([torch.arange(start, device=device), selected, torch.tensor([end], device=device)])
        return keep, throw, anchors

    def _target(self, num_of_token: int) -> float:
        """Entries per layer to keep (single-turn: the original rule, (1 - ratio) x all tokens so far)."""
        return num_of_token * (1 - self.ratio)

    def _fixed_point(self, seq_len: int, forget_num: int) -> int:
        if self.distance > 0:
            return self.distance
        point = seq_len - forget_num + self.distance
        if self.selection == "official":  # may be negative for short prompts: the original then counts from the end
            return point
        return max(point, self.start_size)

    def _decode(self, past_key_values, num_of_token) -> Tuple[object, Optional[int]]:
        """Fixed-point elimination: deletes the entry at `fixed_point` when over budget; returns the cache and that index."""
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - self._target(num_of_token))
        point = self.fixed_point
        if point is not None and point < 0:
            point += seq_len  # Python slicing in the original code
        if forget_num <= 0 or point is None or not 0 <= point < seq_len - self.protect_size:
            return past_key_values, None
        cache = [
            [torch.cat([k[:, :, :point], k[:, :, point + 1 :]], dim=self.k_seq_dim), torch.cat([v[:, :, :point], v[:, :, point + 1 :]], dim=self.v_seq_dim)]
            for k, v in past_key_values
        ]
        return DynamicCache(cache), point


class MultiTurnElasticCache(CarriedCache, ElasticCache):
    """Elastic Cache whose compressed cache is carried across the turns of one conversation.

    Every prefill merges the carried cache plus the new tokens down to the budget (fixed at turn 1 by default, see
    carried_cache.py), scored by the attention from the newly processed tokens. With decode_eviction=False every generated token is
    kept until the next prefill; True applies fixed-point elimination while decoding, as in single-turn use.
    """

    def __init__(self, *args, layer_num=None, image_token_id=None, decode_eviction=False, fixed_budget=True, **kwargs):
        ElasticCache.__init__(self, *args, layer_num=layer_num, **kwargs)
        self._init_carried(layer_num, image_token_id, decode_eviction, fixed_budget)

    __call__ = CarriedCache.__call__

    def _step(self, past_key_values, num_of_token, attentions, prefill: bool, evict: bool):
        if prefill:
            scores = _prefill_scores(attentions)
            if scores is None:
                raise RuntimeError("Elastic Cache needs the attention weights: load the model with attn_implementation=eager.")
            return self._prefill(past_key_values, num_of_token, scores)
        if not evict:
            return past_key_values, None
        out, point = self._decode(past_key_values, num_of_token)
        if point is None:
            return out, None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        return out, [without_index(seq_len, point)] * self.layer_num


def _require_attention(attentions) -> None:
    if attentions is None or (not isinstance(attentions, dict) and (len(attentions) == 0 or attentions[0] is None)):
        raise RuntimeError("Elastic Cache needs the attention weights: load the model with attn_implementation=eager.")


def _importance(attention: torch.Tensor) -> torch.Tensor:
    """Attention each key receives from the processed queries: mean over heads, sum over queries ([1, H, Q, K] -> [K])."""
    return attention[0].float().mean(0).sum(0)


def _prefill_scores(attentions) -> Optional[Sequence[torch.Tensor]]:
    """Per-layer importance at a prefill (online statistics or full attention maps), or None for a decode step."""
    if isinstance(attentions, dict) and attentions.get("elastic_online_prefill"):
        return attentions["scores"]
    if attentions is not None and not isinstance(attentions, dict) and attentions[0] is not None and attentions[0].shape[-2] > 1:
        return [_importance(a) for a in attentions]
    return None


def _merge(x: torch.Tensor, throw: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Averages each thrown entry into its anchor (the anchor's own value included), as the original scatter_reduce."""
    if throw.numel() == 0:
        return x
    shape = (x.shape[0], x.shape[1], -1, x.shape[-1])
    src = x.gather(-2, throw.view(1, 1, -1, 1).expand(*shape))
    return x.scatter_reduce(-2, anchors.view(1, 1, -1, 1).expand(*shape), src, "mean")
