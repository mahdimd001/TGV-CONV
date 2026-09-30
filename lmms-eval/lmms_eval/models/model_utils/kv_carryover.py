"""Multi-turn generation that carries the KV cache from one turn to the next.

Turn 1 prefills the image and the first question; with TGV-KV the cache is compressed at that prefill.
Turn r > 1 prefills only the new tokens (the end of the previous answer, the chat-template glue and the
next question) on top of the carried cache, and TGV-KV compresses the whole cache again. So turn 2 sees the
compressed image + question 1 and answer 1 as the model generated it, instead of re-encoding everything.

Without KV compression (KV_CACHE_TYPE unset) the cache is carried in full, which is equivalent to
re-encoding the whole conversation, only cheaper.

Environment variables:
  MULTI_TURN_KV_CARRYOVER    1 (default) carry the cache | 0 re-encode the full history every turn (v3 behaviour)
  MULTI_TURN_DECODE_EVICTION 0 (default) compress only at each turn's prefill, answers are kept in full until the
                             next prefill | 1 also apply TGV-KV's one-entry-per-step eviction while decoding
  CONVBENCH_DEBUG_PROMPTS    1 to log, per turn, the cached image/text entries and the tokens processed
  PPL_HISTORY                model (default) score reference k where answer k is generated, on a copy of the
                             conversation's cache | reference: after the reference history (see reference_scoring)
"""

import copy
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from lmms_eval.models.model_utils.reference_scoring import (
    ForcedTokenScorer,
    ReferenceScores,
    reference_requests,
)
from loguru import logger as eval_logger
from transformers import GenerationConfig, LogitsProcessorList

GENERATED = "\x00<model answer>\x00"  # placeholder used to ask the task which assistant turns it fixes


def carryover_enabled() -> bool:
    return os.environ.get("MULTI_TURN_KV_CARRYOVER", "1").strip() != "0"


def make_criteria():
    """A TGV-KV state shared by all turns of one conversation, or None without KV compression."""
    method = os.environ.get("KV_CACHE_TYPE")
    if not method or method.lower() == "none":
        return None
    from kv_caches import get_kv_cache  # TGV-KV root package, importable when TGV-KV's patches are loaded

    decode_eviction = os.environ.get("MULTI_TURN_DECODE_EVICTION", "0").strip() == "1"
    return get_kv_cache(method, prune_ratio=float(os.environ.get("PRUNE_RATIO", 0.9)), multi_turn=True, decode_eviction=decode_eviction)


@dataclass
class ChatAdapter:
    """Model-specific pieces the conversation loop needs."""

    chat_text: Callable[[List[Dict[str, str]]], str]  # chat history -> prompt text (ends with the generation prompt)
    first_inputs: Callable[[List[Dict[str, str]]], Dict]  # chat history -> processor outputs incl. the image
    tokenize: Callable[[str], List[int]]  # prompt text -> ids, same tokenization as the processor (placeholders not expanded)
    tokenize_answer: Callable[[str], List[int]]  # assistant text -> ids
    decode: Callable[[List[int]], str]  # generated ids -> answer text
    decode_prompt: Callable[[List[int]], str]  # prompt ids -> readable text (for CONVBENCH_DEBUG_PROMPTS)
    generate_kwargs: Dict  # settings for model.generate on the model's own turns
    eos_ids: Set[int]
    pad_id: Optional[int]
    image_token_ids: Set[int]


def collapse_placeholders(text: str, token: str) -> str:
    """Shows a run of repeated placeholder tokens (e.g. 576 x '<image>') as '[<image> x576]'."""
    return re.sub(rf"(?:{re.escape(token)}\s*)+", lambda m: f"[{token} x{m.group(0).count(token)}] ", text)


def continuation_ids(prev_prompt: str, answer: str, next_prompt: str, tokenize: Callable[[str], List[int]]) -> List[int]:
    """Token ids that follow the previous answer in the next turn's prompt (template glue + next question)."""
    if not next_prompt.startswith(prev_prompt):
        raise RuntimeError("The chat template renders earlier turns differently in later prompts, so the KV cache cannot be carried over. Set MULTI_TURN_KV_CARRYOVER=0.")
    start = next_prompt.find(answer, len(prev_prompt)) if answer else len(prev_prompt)
    if start < 0:
        raise RuntimeError("The previous answer was not found in the next prompt; set MULTI_TURN_KV_CARRYOVER=0.")
    full, prefix = tokenize(next_prompt), tokenize(next_prompt[: start + len(answer)])
    k = 0
    while k < min(len(full), len(prefix)) and full[k] == prefix[k]:
        k += 1
    return full[k:]


class CarryOverConversation:
    """One conversation: model.generate is called once per turn with the cache left by the previous turn."""

    def __init__(self, model, adapter: ChatAdapter):
        self.model, self.adapter = model, adapter
        self.criteria = make_criteria()
        if self.criteria is not None:
            config = getattr(model.config, "text_config", model.config)
            if getattr(config, "_attn_implementation", None) != "eager":
                raise RuntimeError("TGV-KV multi-turn carry-over needs attn_implementation=eager.")
        self.cache = None
        self.stream: List[int] = []  # conversation tokens the cache represents
        self.pending: List[int] = []  # tokens produced but not yet processed by the model
        self.first_inputs = None
        self.turn = 0
        self.label = ""  # shown in the debug log

    def fork(self) -> "CarryOverConversation":
        """An independent copy of the conversation (cache, TGV-KV state, tokens); the original is left untouched."""
        twin = copy.copy(self)
        twin.cache = copy.deepcopy(self.cache)
        twin.criteria = copy.deepcopy(self.criteria)
        twin.stream, twin.pending = list(self.stream), list(self.pending)
        return twin

    def score_on_fork(self, target_ids: List[int]) -> Tuple[float, int]:
        """NLL of `target_ids` as the next assistant turn, scored on a copy so the conversation itself does not change."""
        twin = self.fork()
        twin.label = " scoring the reference"
        scorer = ForcedTokenScorer(target_ids)
        twin.force(target_ids, len(target_ids), scorer)
        return (-torch.stack(scorer.logprobs).sum().item(), len(scorer.logprobs)) if scorer.logprobs else (0.0, 0)

    def start(self, first_inputs: Dict) -> None:
        self.first_inputs = first_inputs

    def extend(self, ids: List[int]) -> None:
        self.pending = self.pending + ids

    def generate(self) -> List[int]:
        return self._run(**self.adapter.generate_kwargs)

    def force(self, answer: List[int], max_forced: int, scorer: Optional[ForcedTokenScorer] = None) -> List[int]:
        """Teacher-forces a fixed assistant turn (a reference answer); `scorer` records the log-probs.

        At most `max_forced` tokens are forced (and scored); the rest of the answer is processed with the next
        turn's prefill, so the conversation the cache holds always matches the chat history.
        """
        target = answer[:max_forced]
        if not target:
            raise ValueError("Cannot force an empty assistant turn")
        scorer = scorer or ForcedTokenScorer(target)
        config = GenerationConfig(max_new_tokens=len(target), do_sample=False, num_beams=1, pad_token_id=self.adapter.pad_id)
        produced = self._run(generation_config=config, use_model_defaults=False, logits_processor=LogitsProcessorList([scorer]))
        self.pending = self.pending + answer[len(produced) :]
        return produced

    def _run(self, **gen) -> List[int]:
        self.turn += 1
        if self.cache is None:
            inputs = dict(self.first_inputs)
            full = inputs["input_ids"][0].tolist()
            new = full
        else:
            new = self.pending
            full = self.stream + new
            device = self.model.device
            inputs = {
                "input_ids": torch.tensor([full], device=device),
                "attention_mask": torch.ones(1, len(full), dtype=torch.long, device=device),
                "past_key_values": self.cache,
                "cache_position": torch.arange(len(self.stream), len(full), device=device),
            }
        before = self._cache_summary()
        self.model._tgv_kv_session = self
        try:
            out = self.model.generate(**inputs, return_dict_in_generate=True, **gen)
        finally:
            del self.model._tgv_kv_session
        if self.criteria is not None and self.criteria.error is not None:
            raise RuntimeError(f"TGV-KV failed during turn {self.turn}") from self.criteria.error

        produced = out.sequences[0, len(full) :].tolist()
        self.cache = out.past_key_values
        self.stream = full + produced[:-1]  # the last token has not been through the model yet
        self.pending = [] if produced[-1:] and produced[-1] in self.adapter.eos_ids else produced[-1:]
        represented = self.criteria.logical_len if self.criteria is not None else self.cache.get_seq_length()
        if represented != len(self.stream):
            raise RuntimeError(f"KV cache covers {represented} tokens but the conversation has {len(self.stream)}")
        if os.environ.get("CONVBENCH_DEBUG_PROMPTS"):
            self._log(new, produced, before)
        return produced

    def _cache_summary(self):
        if self.cache is None:
            return None
        if self.criteria is None:
            n = self.cache.get_seq_length()
            image = sum(t in self.adapter.image_token_ids for t in self.stream)
            return image, n - image
        flags = self.criteria.is_image
        image = sum(int(f.sum()) for f in flags) / len(flags)
        return image, sum(f.numel() for f in flags) / len(flags) - image

    def _log(self, new, produced, before):
        new_image = sum(t in self.adapter.image_token_ids for t in new)
        after = self._cache_summary()
        cached = "empty" if before is None else f"{before[0]:.0f} image + {before[1]:.0f} text"
        compressed = self.criteria.after_prefill if self.criteria is not None else None
        compressed = "no compression" if compressed is None else f"{compressed[0]:.0f} image + {compressed[1]:.0f} text"
        eval_logger.info(
            f"[turn {self.turn}{self.label}] cache before: {cached} | prefill processes {new_image} image + {len(new) - new_image} text"
            f" | after prefill compression: {compressed} | produced {len(produced)} tokens"
            f" | cache after the turn: {after[0]:.0f} image + {after[1]:.0f} text (entries per layer, mean)"
            f"\nprefill text: {self.adapter.decode_prompt(new)}"
        )


def _fixed_answer(doc_to_text, doc, outputs, round_info) -> Optional[str]:
    """The assistant text the task puts in the history for the next turn instead of the model's (e.g. a reference)."""
    turn = len(outputs) + 1
    _, history, terminal, _, _ = doc_to_text(doc, previous_output=list(outputs) + [GENERATED], round_idx=turn, previous_round_info=round_info)
    if terminal or not isinstance(history, list) or len(history) < 2 * turn:
        return None
    message = history[2 * turn - 1]
    if message.get("role") != "assistant" or message.get("content") == GENERATED:
        return None
    return message["content"]


def run_conversation(model, adapter: ChatAdapter, doc, context: str, doc_to_text, max_new_tokens: int, references: Optional[List[str]] = None) -> Tuple[List[str], Optional[Dict[str, list]]]:
    """All turns of one conversation with the KV cache carried over; returns the per-turn answers and reference NLLs.

    With `references`, the reference of turn k is scored right before answer k is produced: on a copy of the cache
    after image, Q1, A1, ..., Qk (the model's own answers), so it sees exactly the context answer k is generated from.
    """
    conv = CarryOverConversation(model, adapter)
    scores = ReferenceScores(references or [], adapter.tokenize_answer, max_new_tokens)
    messages = [{"role": "user", "content": context}]
    prompt = adapter.chat_text(messages)
    conv.start(adapter.first_inputs(messages))
    outputs, round_info = [], None
    while True:
        scores.add(len(outputs) + 1, conv.score_on_fork)
        fixed = _fixed_answer(doc_to_text, doc, outputs, round_info)
        if fixed is None:
            answer = adapter.decode(conv.generate())
        else:
            conv.force(adapter.tokenize_answer(fixed), max_new_tokens)
            answer = fixed
        outputs.append(answer)
        _, history, terminal, _, round_info = doc_to_text(doc, previous_output=list(outputs), round_idx=len(outputs), previous_round_info=round_info)
        if terminal:
            return outputs, scores.result()
        if not isinstance(history, list):
            raise RuntimeError("KV carry-over needs doc_to_text to return the chat history as a list of messages; set MULTI_TURN_KV_CARRYOVER=0.")
        next_prompt = adapter.chat_text(history)
        conv.extend(continuation_ids(prompt, answer, next_prompt, adapter.tokenize))
        prompt = next_prompt


def score_references_carryover(model, adapter: ChatAdapter, doc, doc_to_text, max_tokens: int) -> Optional[Dict[str, list]]:
    """PPL_HISTORY=reference: reference NLLs with the cache carried across turns, each forced after the compressed reference history."""
    requests = list(reference_requests(doc_to_text, doc))
    if not requests:
        return None
    conv = CarryOverConversation(model, adapter)
    nll, tokens, truncated = [], [], []
    prev_prompt = prev_reference = None
    for history, reference in requests:
        prompt = adapter.chat_text(history)
        if prev_prompt is None:
            conv.start(adapter.first_inputs(history))
        else:
            conv.extend(continuation_ids(prev_prompt, prev_reference, prompt, adapter.tokenize))
        all_ids = adapter.tokenize_answer(reference)
        scorer = ForcedTokenScorer(all_ids[:max_tokens])
        conv.force(all_ids, max_tokens, scorer)
        nll.append(-torch.stack(scorer.logprobs).sum().item() if scorer.logprobs else 0.0)
        tokens.append(len(scorer.logprobs))
        truncated.append(len(scorer.logprobs) < len(all_ids))
        prev_prompt, prev_reference = prompt, reference
    return {"reference_nll": nll, "reference_tokens": tokens, "reference_truncated": truncated}
