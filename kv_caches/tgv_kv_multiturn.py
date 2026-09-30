"""TGV-KV for multi-turn conversations: the compressed KV cache is carried from one turn to the next."""

import numpy as np
import torch
from colorama import Fore
from transformers.cache_utils import DynamicCache

from .tgv_kv import TGVKVCache


def _text_after_first_image(kept_flags):
    """Kept text entries after the first kept image entry: the text TGV-KV protects at the end of the cache."""
    if kept_flags.any():
        first = int(kept_flags.nonzero()[0])
        return int((~kept_flags[first:]).sum())
    return int((~kept_flags).sum())


class MultiTurnTGVKVCache(TGVKVCache):
    """TGV-KV whose compressed cache is carried across the turns of one conversation.

    Turn 1 is compressed exactly like TGVKVCache (full image + first question at prefill).
    From turn 2 on, prefill only processes the new tokens on top of the carried cache, then the whole
    cache is compressed again:
      * which cached entries are image tokens is tracked per layer (`is_image`), because each layer keeps
        a different subset after eviction;
      * image entries are ranked by text-weighted attention from the new text tokens (TWR), text entries are
        kept first (TPR), and layer budgets follow the new text tokens' attention to the image (TVB);
      * the budget is (1 - ratio) x all tokens of the conversation so far, the rule TGV-KV's decode step uses.
    decode_eviction=False keeps every generated token until the next prefill (compression only at prefills);
    True applies TGV-KV's usual one-entry-per-step decode eviction as well.
    """

    def __init__(self, *args, decode_eviction=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_eviction = decode_eviction
        self.is_image = None  # per layer: bool tensor with one flag per cached entry
        self.logical_len = 0  # conversation tokens the cache represents (positions are logical)
        self.error = None
        self.after_prefill = None  # (image, text) entries per layer right after the last prefill's compression
        self._history_stats = None

    # ---------------------------------------------------------------- online prefill statistics

    def begin_online_prefill(self, input_ids):
        if self.is_image is None:
            return super().begin_online_prefill(input_ids)
        new_ids = input_ids[0, self.logical_len :]
        self._history_stats = {
            "new_is_image": new_ids == self.image_token_id,
            "score_sums": [None] * self.layer_num,
            "text_image_attn_sums": [None] * self.layer_num,
        }
        return True

    def collect_online_prefill_attention(self, layer_idx, attention):
        stats = self._history_stats
        if stats is None:
            return super().collect_online_prefill_attention(layer_idx, attention)

        attn = attention.squeeze(0).mean(0).float()  # [new tokens (queries), cached + new keys]
        q_len, k_len = attn.shape
        flags = torch.cat([self.is_image[layer_idx], stats["new_is_image"].to(self.is_image[layer_idx].device)]).to(attn.device)
        if flags.numel() != k_len:
            raise RuntimeError(f"TGV-KV layer {layer_idx}: {k_len} keys but {flags.numel()} tracked entries")

        new_text = ~flags[k_len - q_len :]
        text_rows = attn[new_text]  # text queries
        n_text = text_rows.shape[0]
        text_text = text_rows[:, k_len - q_len :][:, new_text]  # new text x new text (causal)
        weight = text_text.sum(0) / torch.arange(n_text, 0, -1, device=attn.device)  # dominant text tokens
        text_image = text_rows[:, flags]
        score = torch.empty(k_len, device=attn.device)
        score[flags] = (text_image * weight[:, None]).sum(0)  # TWR
        score[~flags] = attn[:, ~flags].sum(0) + 100  # TPR: text first
        stats["score_sums"][layer_idx] = score.unsqueeze(0)
        stats["text_image_attn_sums"][layer_idx] = text_image.sum()

    def finish_online_prefill(self):
        stats = self._history_stats
        if stats is None:
            return super().finish_online_prefill()
        missing = [i for i, s in enumerate(stats["score_sums"]) if s is None]
        if missing:
            raise RuntimeError(f"Missing TGV-KV online prefill stats for layers: {missing}")
        self._history_stats = None
        return {"tgv_kv_online_prefill": True, "history": True, **stats}

    # ---------------------------------------------------------------- compression

    def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        try:
            if past_key_values is None:
                return None
            if self._is_online_prefill_stats(attentions):
                self.initial_text_len_list = []
                if attentions.get("history"):
                    out = self._prefill_history(past_key_values, num_of_token, attentions)
                else:
                    out = self._prefill_fresh(past_key_values, num_of_token, attentions, input_ids)
                image = sum(int(f.sum()) for f in self.is_image) / self.layer_num
                self.after_prefill = (image, sum(f.numel() for f in self.is_image) / self.layer_num - image)
            elif attentions[0].shape[-2] > 1:
                raise RuntimeError("Multi-turn TGV-KV needs attn_implementation=eager and an image in the first turn.")
            else:
                out = self._decode_tracked(past_key_values, num_of_token, attentions)
            self.logical_len = num_of_token
            return out
        except Exception as e:  # generate() swallows criteria errors; keep it so the caller can raise
            self.error = e
            raise

    def _select(self, past_key_values, score_sums, forget_nums, flags, protected_text):
        """TGV-KV's per-layer selection (as in TGVKVCache._prefill_from_online_stats), also carrying the flags."""
        cache, kept = [], []
        for idx in range(self.layer_num):
            k, v = past_key_values[idx]
            seq_len = k.size(self.k_seq_dim)
            selected_idx = torch.argsort(score_sums[idx][:, self.start_size : (seq_len - self.protect_size)])[:, forget_nums[idx] :] + self.start_size
            selected_idx = selected_idx.sort().values

            device = selected_idx.device
            pre = torch.arange(self.start_size, device=device).unsqueeze(0).expand(self.batch_size, -1)
            post = torch.tensor([seq_len - self.protect_size], device=device).unsqueeze(0).expand(self.batch_size, -1)
            selected_idx = torch.cat([pre, selected_idx, post], dim=-1)
            kept_flags = flags[idx].to(device)[selected_idx[0]]
            self.initial_text_len_list.append(max(protected_text(selected_idx[0], kept_flags), self.protect_size))

            selected_idx = selected_idx.to(k.device)
            k_select = k.gather(dim=-2, index=selected_idx.view(self.batch_size, 1, -1, 1).expand(-1, k.shape[1], -1, k.shape[-1]))
            v_select = v.gather(dim=-2, index=selected_idx.view(self.batch_size, 1, -1, 1).expand(-1, v.shape[1], -1, v.shape[-1]))
            cache.append([k_select, v_select])
            kept.append(kept_flags.to(flags[idx].device))
        self.is_image = kept
        return DynamicCache(cache)

    def _prefill_fresh(self, past_key_values, num_of_token, stats, input_ids):
        """Turn 1: identical to TGVKVCache._prefill_from_online_stats, plus the image flags."""
        self.is_image = [(input_ids[0] == self.image_token_id) for _ in range(self.layer_num)]
        self.ratios = np.zeros(self.layer_num)
        self.initial_text_len_list = [self.protect_size] * self.layer_num

        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if int(seq_len - num_of_token * (1 - self.ratio)) * self.layer_num <= 0:
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        text_start = stats["text_start"]
        text_image_attn_sum = torch.stack(list(stats["text_image_attn_sums"]))
        normalized_layer_ratio = text_image_attn_sum / text_image_attn_sum.sum()
        layer_ratio = (seq_len - (len(normalized_layer_ratio) * seq_len * (1 - self.ratio) * normalized_layer_ratio)) / seq_len
        self.ratios = layer_ratio.float().cpu().numpy()
        forget_nums = (self.ratios * seq_lens).round().astype(np.int32)
        forget_nums[forget_nums < 0] = 0
        if np.all(forget_nums <= 0):
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        self.initial_text_len_list = []
        return self._select(past_key_values, list(stats["score_sums"]), forget_nums, self.is_image, lambda idx, _: int((idx >= text_start).sum()))

    def _prefill_history(self, past_key_values, num_of_token, stats):
        """Turn 2+: compress carried cache + new tokens to (1 - ratio) x conversation length."""
        new_flags = stats["new_is_image"]
        flags = [torch.cat([f, new_flags.to(f.device)]) for f in self.is_image]
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        if any(seq_lens[i] != flags[i].numel() for i in range(self.layer_num)):
            raise RuntimeError("TGV-KV cache and image flags are out of sync")
        self.is_image = flags

        tia = torch.stack([t.float() for t in stats["text_image_attn_sums"]])
        total = tia.sum()
        share = tia / total if total > 0 else torch.full_like(tia, 1.0 / self.layer_num)  # TVB
        keep = (self.layer_num * num_of_token * (1 - self.ratio) * share).cpu().numpy()
        self.ratios = 1 - keep / num_of_token  # decode keeps num_of_token * (1 - ratios) per layer
        forget_nums = np.round(seq_lens - keep).astype(np.int32)
        forget_nums[forget_nums < 0] = 0

        self.initial_text_len_list = [max(_text_after_first_image(f), self.protect_size) for f in flags]
        if np.all(forget_nums <= 0):
            return past_key_values
        self.initial_text_len_list = []
        return self._select(past_key_values, list(stats["score_sums"]), forget_nums, flags, lambda _, kept_flags: _text_after_first_image(kept_flags))

    def _decode_tracked(self, past_key_values, num_of_token, attentions):
        """TGVKVCache._decode that also drops the evicted entry's flag; with decode_eviction=False nothing is evicted."""
        self.is_image = [torch.cat([f, f.new_zeros(1)]) for f in self.is_image]  # the token just processed
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        if any(seq_lens[i] != self.is_image[i].numel() for i in range(self.layer_num)):
            raise RuntimeError("TGV-KV cache and image flags are out of sync")
        if not self.decode_eviction:
            return past_key_values

        forget_nums = (seq_lens - num_of_token * (1 - self.ratios)).astype(np.int32)
        forget_nums[forget_nums < 0] = 0
        if np.all(forget_nums <= 0):
            return past_key_values

        out = []
        for i, (k, v) in enumerate(past_key_values):
            seq_len = seq_lens[i]
            evict_start, evict_end = self.start_size, seq_len - self.initial_text_len_list[i]
            if forget_nums[i] == 0 or evict_start >= evict_end:
                out.append([k, v])
                continue
            decode_score = attentions[i].mean(1).squeeze(0).sum(0)
            pruned_idx = decode_score[evict_start:evict_end].argmin().item() + evict_start
            out.append(
                [
                    torch.cat([k[:, :, 0:pruned_idx], k[:, :, (pruned_idx + 1) : seq_len]], dim=self.k_seq_dim),
                    torch.cat([v[:, :, 0:pruned_idx], v[:, :, (pruned_idx + 1) : seq_len]], dim=self.v_seq_dim),
                ]
            )
            self.is_image[i] = torch.cat([self.is_image[i][:pruned_idx], self.is_image[i][pruned_idx + 1 :]])
        return DynamicCache(out)
