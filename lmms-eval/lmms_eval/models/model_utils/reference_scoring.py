"""Teacher-forced scoring of reference answers through `model.generate`.

Scoring runs the same decode loop as generation, one forced token per step, so any KV-cache
compression hooked into `generate` (e.g. TGV-KV) applies exactly as it does when the model answers.
A plain forward pass over prompt + reference would bypass that compression.

Protocol: a multi-round task opts in by giving its `doc_to_text` a `reference_round` keyword.
`doc_to_text(doc, reference_round=k)` returns `(chat_messages, reference_text)` for k = 1, 2, ...
and None when there is nothing more to score.

Which history the reference of turn k is conditioned on (PPL_HISTORY):
  model (default)  the conversation the model is actually having: image, Q1, A1, ..., Qk, with its own answers A;
                   R_k is scored at the point where A_k is generated
  reference        the reference conversation: image, Q1, R1, ..., Qk (the history `doc_to_text` returns)
"""

import inspect
import os
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList


class ForcedTokenScorer(LogitsProcessor):
    """Records the log-probability of the next target token under the raw logits, then forces that token."""

    def __init__(self, target_ids: List[int]) -> None:
        self.target_ids = target_ids
        self.logprobs: List[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        target = self.target_ids[min(len(self.logprobs), len(self.target_ids) - 1)]
        self.logprobs.append(torch.log_softmax(scores[0].float(), dim=-1)[target])
        forced = torch.full_like(scores, float("-inf"))
        forced[:, target] = 0.0
        return forced


def forced_decode_nll(model, inputs: Dict, target_ids: List[int], eos_token_id: Optional[int], pad_token_id: Optional[int]) -> Tuple[float, int]:
    """Sum of negative log-likelihoods of `target_ids` given `inputs`, and the number of tokens scored.

    Fewer tokens than requested are scored if the decode loop stops early (e.g. TGV-KV's MAX_GENERATED_TOKENS).
    The model's own generation defaults (repetition penalty etc.) are disabled so the log-probs are raw.
    """
    scorer = ForcedTokenScorer(target_ids)
    config = GenerationConfig(max_new_tokens=len(target_ids), do_sample=False, num_beams=1, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    model.generate(**inputs, generation_config=config, use_model_defaults=False, logits_processor=LogitsProcessorList([scorer]))
    if not scorer.logprobs:
        return 0.0, 0
    return -torch.stack(scorer.logprobs).sum().item(), len(scorer.logprobs)


def reference_requests(doc_to_text: Callable, doc: Dict) -> Iterator[Tuple[List[Dict[str, str]], str]]:
    """Yields the task's (chat_messages, reference_text) pairs, or nothing if the task does not opt in."""
    if "reference_round" not in inspect.signature(doc_to_text).parameters:
        return
    k = 1
    while (request := doc_to_text(doc, reference_round=k)) is not None:
        yield request
        k += 1


def ppl_on_model_history() -> bool:
    """True (default) to score each reference after the model's own earlier answers, False for the reference history."""
    value = os.environ.get("PPL_HISTORY", "model").strip().lower()
    if value not in ("model", "reference"):
        raise ValueError(f"PPL_HISTORY must be 'model' or 'reference', got {value!r}")
    return value == "model"


def reference_answers(doc_to_text: Callable, doc: Dict) -> List[str]:
    """The reference answer of every turn the task scores (empty if it does not opt in)."""
    return [reference for _, reference in reference_requests(doc_to_text, doc)]


class ReferenceScores:
    """Collects per-turn reference NLLs; `add` scores the reference of one turn with `score(target_ids) -> (nll, n)`.

    Errors propagate: a failure here is usually systematic (the same one hits every doc), so it should stop the run.
    """

    def __init__(self, references: List[str], tokenize: Callable[[str], List[int]], max_tokens: int) -> None:
        self.references, self.tokenize, self.max_tokens = references, tokenize, max_tokens
        self.nll: List[float] = []
        self.tokens: List[int] = []
        self.truncated: List[bool] = []

    def add(self, turn: int, score: Callable[[List[int]], Tuple[float, int]]) -> None:
        if turn > len(self.references):
            return
        all_ids = self.tokenize(self.references[turn - 1])
        target_ids = all_ids[: self.max_tokens]
        total, n = score(target_ids) if target_ids else (0.0, 0)
        self.nll.append(total)
        self.tokens.append(n)
        self.truncated.append(n < len(all_ids))

    def result(self) -> Optional[Dict[str, list]]:
        if not self.tokens:
            return None
        return {"reference_nll": self.nll, "reference_tokens": self.tokens, "reference_truncated": self.truncated}


def score_references(doc: Dict, doc_to_text: Callable, tokenize: Callable[[str], List[int]], score: Callable[[List[Dict[str, str]], List[int]], Tuple[float, int]], max_tokens: int) -> Optional[Dict[str, list]]:
    """Scores every reference the task offers for `doc` after the reference history; per-turn NLLs and token counts, or None."""
    requests = list(reference_requests(doc_to_text, doc))
    scores = ReferenceScores([reference for _, reference in requests], tokenize, max_tokens)
    for turn, (messages, _) in enumerate(requests, start=1):
        scores.add(turn, lambda ids, messages=messages: score(messages, ids))
    return scores.result()
