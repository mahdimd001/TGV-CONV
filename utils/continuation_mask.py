import torch


def continuation_causal_mask(attn_weights: torch.Tensor) -> torch.Tensor:
    """Causal mask for a prefill of q new tokens on top of an existing cache, aligned to the end of the keys.

    With multi-turn KV carry-over each layer's cache has its own length after eviction, so the model-level mask
    (built for one layer, in logical positions) does not fit: new token i may see every cached entry and new tokens 0..i.
    """
    q_len, k_len = attn_weights.shape[-2], attn_weights.shape[-1]
    rows = torch.arange(q_len, device=attn_weights.device)[:, None]
    cols = torch.arange(k_len, device=attn_weights.device)[None, :]
    mask = torch.zeros(q_len, k_len, device=attn_weights.device, dtype=attn_weights.dtype)
    return mask.masked_fill(cols > rows + (k_len - q_len), torch.finfo(attn_weights.dtype).min)
