import os

from colorama import Fore, Style

from .elastic_cache import ElasticCache, MultiTurnElasticCache
from .h2o_cache import H2OCache, MultiTurnH2OCache
from .local_cache import LocalCache, MultiTurnLocalCache
from .tgv_kv import TGVKVCache
from .tgv_kv_multiturn import MultiTurnTGVKVCache

TGV_KV_NAMES = ("tgv_kv", "tgv-kv", "tgvkv")
ELASTIC_NAMES = ("elastic", "elastic_cache", "elastic-cache", "elasticcache")
H2O_NAMES = ("h2o", "h2o_cache")
LOCAL_NAMES = ("local", "local_cache", "streamingllm", "streaming_llm")


def get_kv_cache(
    method="tgv_kv",
    start_size=4,
    recent_size=2047,
    k_seq_dim=2,
    v_seq_dim=2,
    prune_ratio=0.2,
    layer_num=36,
    model_name="llava-v1.5-7b",
    multi_turn=False,
    decode_eviction=False,
    fixed_budget=True,
):
    first_call = not hasattr(get_kv_cache, "_printed")
    if first_call:
        setattr(get_kv_cache, "_printed", True)

    if method is None or method.lower() == "none":
        if first_call:
            print(f"{Fore.RED}!!! No caching method is used. !!!{Style.RESET_ALL}")
        return None
    method = method.lower()

    if method not in TGV_KV_NAMES + ELASTIC_NAMES + H2O_NAMES + LOCAL_NAMES:
        raise ValueError(f"Unsupported KV cache method: {method}. Available: tgv_kv, elastic, h2o, local.")

    model_type = os.environ.get("MODEL_TYPE", None)
    if model_type == "llava-7B":
        image_token_id = 32000
        layer_num = 32
    elif model_type == "qwen-8B" or model_type == "qwen-4B":
        image_token_id = 151655
        layer_num = 36
    elif model_type == "qwen-8B-video" or model_type == "qwen-4B-video":
        image_token_id = 151656
        layer_num = 36
    elif model_type == "llava-ov-0.5B":
        image_token_id = 151646
        layer_num = 24
    elif model_type == "qwen2_5_vl_7B":
        image_token_id = 151655
        layer_num = 28
    elif model_type == "qwen2_5_vl_3B":
        image_token_id = 151655
        layer_num = 36
    else:
        raise NotImplementedError("Not supported model! Please manually set image_token_id and layer_num.")

    if method in ELASTIC_NAMES:
        # Elastic Cache (ECCV 2024). Defaults follow the authors' scripts: 1 sink token, fixed point 25 before the
        # end of the compressed prompt. ELASTIC_SELECTION=official reproduces the original code's token ranking.
        if first_call:
            print(f"{Fore.GREEN}+++ Using Elastic Cache +++{Style.RESET_ALL}")
        cache_cls, extra = (MultiTurnElasticCache, {"decode_eviction": decode_eviction, "image_token_id": image_token_id, "fixed_budget": fixed_budget}) if multi_turn else (ElasticCache, {})
        return cache_cls(
            **extra,
            layer_num=layer_num,
            start_size=int(os.environ.get("ELASTIC_START_SIZE", 1)),
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=prune_ratio,
            distance=int(os.environ.get("ELASTIC_DISTANCE", -25)),
            selection=os.environ.get("ELASTIC_SELECTION", "paper").strip().lower(),
        )

    # H2O and Local cache as implemented in the Elastic Cache repository (1 sink token, as in its scripts).
    # H2O_SELECTION=official reproduces the original H2O code exactly (single-turn only).
    if method in H2O_NAMES + LOCAL_NAMES:
        is_h2o = method in H2O_NAMES
        if first_call:
            print(f"{Fore.GREEN}+++ Using {'H2O' if is_h2o else 'Local'} Cache +++{Style.RESET_ALL}")
        single, multi = (H2OCache, MultiTurnH2OCache) if is_h2o else (LocalCache, MultiTurnLocalCache)
        cache_cls, extra = (multi, {"decode_eviction": decode_eviction, "image_token_id": image_token_id, "fixed_budget": fixed_budget}) if multi_turn else (single, {})
        options = {"recent_size": recent_size, "selection": os.environ.get("H2O_SELECTION", "paper").strip().lower()} if is_h2o else {}
        return cache_cls(
            **extra,
            **options,
            layer_num=layer_num,
            start_size=int(os.environ.get("H2O_START_SIZE" if is_h2o else "LOCAL_START_SIZE", 1)),
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=prune_ratio,
        )

    if method in TGV_KV_NAMES:
        if first_call:
            print(f"{Fore.GREEN}+++ Using TGV-KV Cache +++{Style.RESET_ALL}")
        # multi_turn: one cache object for a whole conversation, carried across turns (see tgv_kv_multiturn.py)
        cache_cls, extra = (MultiTurnTGVKVCache, {"decode_eviction": decode_eviction, "fixed_budget": fixed_budget}) if multi_turn else (TGVKVCache, {})
        return cache_cls(
            **extra,
            image_token_id=image_token_id,
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=prune_ratio,
            layer_num=layer_num,
        )
