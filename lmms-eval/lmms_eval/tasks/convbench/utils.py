"""ConvBench (NeurIPS 2024 D&B) for lmms-eval.

Liu et al., "ConvBench: A Multi-Turn Conversation Evaluation Benchmark with Hierarchical
Ablation Capability for Large Vision-Language Models", arXiv:2403.20194.

Each doc is one 3-turn conversation over a single image: perception -> reasoning -> creation.
Generation runs through `generate_until_multi_round`: the model answers turn 1 from the image,
then this module hands it the chat history for turns 2 and 3.

Scoring re-implements the paper's ConvEval with a text-only LLM judge that sees the
instruction-conditioned caption instead of the image:
  * pairwise (default): the judge compares the model's conversation with the human-verified
    reference in random A/B order; the metric is the model's win rate (%) against the reference.
  * direct: the judge rates each turn 1-10 with the reference counted as 10; the metric is the mean.
Turn scores S1/S2/S3 are judged first, and their judgements feed the overall score S0.
R2 = mean(S1, S2, S3) and R1 = (R2 + S0) / 2, as in Tables 1-2 of the paper.

Settings (lmms_eval_specific_kwargs.n_ref_turns):
  0 -> S   the model sees its own earlier answers (main result)
  1 -> S^  turn 1 in the history is replaced by the reference (perfect perception)
  2 -> S~  turns 1 and 2 are replaced by references (perfect perception and reasoning)

Reference-based metrics (no judge needed):
  * BLEU-1..4, METEOR, ROUGE-L (x100) between each generated turn and its reference, computed with
    pycocoevalcap (PTB tokenizer, METEOR 1.5) like lmms-eval's coco_cap; needs Java. Pooled over all
    scored turns; per-turn values are written to <output_path>/submissions/<task>_by_turn_*.json.
  * PPL: perplexity of the reference answers. By default (PPL_HISTORY=model) reference k is scored
    in the model's own conversation: after the image, Q1, A1, ..., Qk, where the A are the model's
    answers, i.e. exactly the context answer k is generated from. PPL_HISTORY=reference conditions
    it on the reference history instead (Q1, R1, ..., Qk). Tokens are forced one at a time through
    model.generate, so KV-cache compression (e.g. TGV-KV) applies as it does during generation.
    Token-weighted over all turns. Main task only.

Environment variables:
  CONVBENCH_GRADING      pairwise (default) | direct | none (skip the judge)
  CONVBENCH_JUDGE_MODEL  judge model name (default gpt-4o-mini; the paper used gpt-3.5-turbo)
  CONVBENCH_TEXT_METRICS 1 (default) | 0 to skip BLEU/METEOR/ROUGE
  CONVBENCH_PPL          1 (default) | 0 to skip perplexity
  CONVBENCH_LOG_TURNS    1 to log every turn (question, model output, reference, PPL) as results are processed
  PPL_HISTORY            model (default) | reference: the history each reference is conditioned on
  API_TYPE               lmms-eval judge provider (default openai)
  OPENAI_API_KEY / OPENAI_API_URL  e.g. https://api.openai.com/v1 or a local vLLM server's /v1
"""

import hashlib
import json
import math
import os
import random
import re
import shutil
from datetime import datetime

from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.llm_judge.protocol import Request
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

GRADING = os.getenv("CONVBENCH_GRADING", "pairwise").strip().lower()
JUDGE_MODEL = os.getenv("CONVBENCH_JUDGE_MODEL", "gpt-4o-mini")
API_TYPE = os.getenv("API_TYPE", "openai")
TEXT_METRICS = os.getenv("CONVBENCH_TEXT_METRICS", "1").strip() != "0"
COMPUTE_PPL = os.getenv("CONVBENCH_PPL", "1").strip() != "0"
LOG_TURNS = os.getenv("CONVBENCH_LOG_TURNS", "0").strip() == "1"
MAX_RESPONSE_CHARS = 6000  # keeps degenerate (e.g. repetitive) outputs from blowing up the judge prompt
RUN_TAG = f"{os.getenv('KV_CACHE_TYPE') or 'fullkv'}_ratio{os.getenv('PRUNE_RATIO', 'na')}_{datetime.now():%Y%m%d_%H%M%S}"

NUM_TURNS = 3
ORDINAL = {1: "first", 2: "second", 3: "third"}
SKILL = {1: "visual perception", 2: "visual reasoning", 3: "visual creation"}
TASK_NAMES = {0: "convbench", 1: "convbench_ref1", 2: "convbench_ref2"}

if GRADING not in ("pairwise", "direct", "none"):
    raise ValueError(f"CONVBENCH_GRADING must be 'pairwise', 'direct' or 'none', got {GRADING!r}")

_server = None
_server_config = ServerConfig(model_name=JUDGE_MODEL, temperature=0.0, max_tokens=1024)


def _judge(system_prompt, user_prompt):
    """Call the judge once; returns the text or '' on failure."""
    global _server
    if _server is None:
        _server = get_server(server_name=API_TYPE, config=_server_config)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    try:
        return _server.evaluate(Request(messages=messages, config=_server_config)).content or ""
    except Exception as e:
        eval_logger.error(f"ConvBench judge call failed: {e}")
        return ""


# ---------------------------------------------------------------- inputs


def convbench_process_docs(dataset):
    """Runs when the task loads, before generation: fails fast on a missing judge key or Java."""
    if GRADING != "none" and API_TYPE == "openai":
        missing = [v for v in ("OPENAI_API_KEY", "OPENAI_API_URL") if not os.getenv(v)]
        if missing:
            raise RuntimeError(f"ConvBench judge needs {', '.join(missing)} (e.g. OPENAI_API_URL=https://api.openai.com/v1, " "or a local vLLM server's http://localhost:8000/v1). Set CONVBENCH_GRADING=none to run without the judge.")
    if TEXT_METRICS and shutil.which("java") is None:
        raise RuntimeError("ConvBench BLEU/METEOR/ROUGE use the COCO-caption toolkit, which needs Java on PATH " "(e.g. `conda install -c conda-forge openjdk` or `apt install default-jre`). " "Set CONVBENCH_TEXT_METRICS=0 to skip them.")
    return dataset


def convbench_doc_to_visual(doc):
    return [Image.open(doc["image_path"]).convert("RGB")]


def convbench_doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None, reference_round=None):
    """Multi-round prompt builder.

    round_idx=None -> the turn-1 instruction (a string, sent with the image).
    round_idx=r    -> after r answered turns: (visuals, chat_history, terminal, previous_output, info),
                      where chat_history is a list of {"role", "content"} text messages ending with the
                      next user instruction. The model attaches the image to the first user message.
    reference_round=k -> (chat_history, reference_k) for perplexity: the reference history of turns < k
                      plus instruction k, and the reference answer to score; None when nothing is left.
    """
    kwargs = lmms_eval_specific_kwargs or {}
    if reference_round is not None:
        if not COMPUTE_PPL or not kwargs.get("score_references", True) or reference_round > NUM_TURNS:
            return None
        history = []
        for k in range(1, reference_round):
            history += [{"role": "user", "content": doc[f"instruction_{k}"]}, {"role": "assistant", "content": doc[f"reference_{k}"]}]
        history.append({"role": "user", "content": doc[f"instruction_{reference_round}"]})
        return history, doc[f"reference_{reference_round}"]

    n_ref = int(kwargs.get("n_ref_turns", 0))
    if round_idx is None:
        return doc["instruction_1"]
    if round_idx >= NUM_TURNS:
        return None, None, True, previous_output, previous_round_info

    history = []
    for k in range(1, round_idx + 1):
        history.append({"role": "user", "content": doc[f"instruction_{k}"]})
        answer = doc[f"reference_{k}"] if k <= n_ref else (previous_output[k - 1] or "")
        history.append({"role": "assistant", "content": answer})
    history.append({"role": "user", "content": doc[f"instruction_{round_idx + 1}"]})
    return None, history, False, previous_output, previous_round_info


# ---------------------------------------------------------------- judge prompts (from the paper's appendix)


def _clip(text):
    text = (text or "").strip()
    return text if len(text) <= MAX_RESPONSE_CHARS else text[:MAX_RESPONSE_CHARS] + " ...[truncated]"


def _conversation_block(doc, responses, speaker):
    lines = [f"<|The Start of {speaker}'s Conversation with User|>"]
    for k in range(1, NUM_TURNS + 1):
        lines.append(f"### The {ORDINAL[k]} turn question from user:\n{doc[f'instruction_{k}']}\n")
        lines.append(f"### The {ORDINAL[k]} turn response from {speaker}:\n{_clip(responses[k - 1])}\n")
    lines.append(f"<|The End of {speaker}'s Conversation with User|>")
    return "\n".join(lines)


def _reference_block(doc):
    lines = ["<|The Start of Reference Answer|>"]
    for k in range(1, NUM_TURNS + 1):
        lines.append(f"### The {ORDINAL[k]} turn question from user:\n{doc[f'instruction_{k}']}\n")
        lines.append(f"### The {ORDINAL[k]} turn high quality reference:\n{doc[f'reference_{k}']}\n")
    lines.append("<|The End of Reference Answer|>")
    return "\n".join(lines)


def _focus_block(doc, turn):
    if turn == NUM_TURNS and doc.get("focus_points"):
        return f"\nThere are some concerns which you should focus on when you make your judgements for the {ORDINAL[turn]} turn response:\n{doc['focus_points']}\n"
    return ""


def _judgement_block(turn_judgements):
    return "\n\n".join(f"The {ORDINAL[k]} turn evaluation: {j}" for k, j in sorted(turn_judgements.items()) if j)


_PAIRWISE_SYSTEM = """You are ImageTaskEvaluationGPT, an expert language model at judging {what}. More specifically, you will be given the following:
1. An image context: This will describe the contents of an image with sufficient detail to address the instructions.
2. Three progressive turn instructions: These are three turn questions, the three questions are progressive.
3. Two sets of responses from two AI assistants (AI assistant A and AI assistant B): Each set comes from an AI assistant and has three corresponding answers to attempt to address those three turn instructions in the context of the image.{extra}
Your job is to judge whether {target} from Assistant A or {target} from Assistant B is better. A and B are randomly ordered.
Some things to remember:
- Even though you are just a language model, the image description will be sufficiently detailed so that your judgements can be accurate.
- You should choose the assistant that {follows}.
- You are capable of judging response quality, accounting for important factors like correctness, relevance, fluency, specificity, etc.
- Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
- Do not allow the length of the responses to influence your evaluation.
- Do not favor certain names of the assistants. Be as objective as possible.
- You think step-by-step, but ultimately respond with "Response A" or "Response B"."""

_DIRECT_SYSTEM = """You are ImageTaskEvaluationGPT, an expert language model at judging {what}. More specifically, you will be given the following:
1. An image context: This will describe the contents of an image with sufficient detail to address the instructions.
2. Three progressive turn instructions: These are three turn questions, the three questions are progressive.
3. Three turn reference outputs: These are high-quality example outputs that humans have judged to be accurate responses for the three progressive instructions.
4. Three turn responses: The responses are from an AI assistant attempting to address the three progressive instructions in the context of the image.{extra}
Your job is to rate {target} from the AI assistant on a scale of 1 to 10, regarding the rating of the corresponding reference as 10.
Some things to remember:
- Even though you are just a language model, the image description will be sufficiently detailed so that your judgement can be accurate.
- Regard the ratings of the high-quality references as 10. Make your rating judgement for the AI assistant compared with the high-quality references.
- Correctness, relevance, fluency and the level of detail are the most important factors.{overall_note}
- Do not allow the length of the responses to influence your evaluation.
- You think step-by-step and are as objective as possible. After providing your explanation, you must rate on a scale of 1 to 10 by strictly following this format: "Rating: [[rating]]", for example: "Rating: [[5]]"."""


def _pairwise_prompts(doc, model_responses, model_is_a, turn=None, turn_judgements=None):
    ref = [doc[f"reference_{k}"] for k in range(1, NUM_TURNS + 1)]
    a, b = (model_responses, ref) if model_is_a else (ref, model_responses)
    if turn is None:
        system = _PAIRWISE_SYSTEM.format(
            what="the multi-turn conversation instruction-following ability of an AI assistant",
            extra="\n4. Evaluations of each turn: judgements that compared the two assistants turn by turn.",
            target="the overall conversation",
            follows="follows the user's instructions and answers the user's questions better across the whole multi-turn conversation",
        )
        task = "compare the overall conversations from the two assistants"
        extra = f"\n{_judgement_block(turn_judgements)}\n" if turn_judgements else ""
    else:
        system = _PAIRWISE_SYSTEM.format(
            what="whether or not a response adequately addresses an instruction in the context of an image",
            extra="\n4. Focus points: some points you should consider when you make the judgement." if turn == NUM_TURNS else "",
            target=f"the {ORDINAL[turn]} turn response",
            follows=f"follows the user's {ORDINAL[turn]} instruction and answers the user's {ORDINAL[turn]} question better",
        )
        task = f"compare the {ORDINAL[turn]} turn responses from the two assistants"
        extra = _focus_block(doc, turn)
    user = (
        "Here are the image context, the instructions, and the two conversations.\n\n"
        f"Image context: {doc['caption']}\n\n"
        f"{_conversation_block(doc, a, 'Assistant A')}\n\n{_conversation_block(doc, b, 'Assistant B')}\n{extra}\n"
        f'Think step-by-step, {task}, and finish your response with "Overall, Response X is better." where X is either A or B.'
    )
    return system, user


def _direct_prompts(doc, model_responses, turn=None, turn_judgements=None):
    if turn is None:
        system = _DIRECT_SYSTEM.format(
            what="the multi-turn conversation instruction-following ability of an AI assistant",
            extra="\n5. Evaluations of each turn: judgements of the AI assistant's individual turns.",
            target="the overall multi-turn conversation",
            overall_note="\n- Account for the multi-turn conversation and instruction-following ability, e.g. whether later turns correctly build on earlier ones.",
        )
        target = "the overall conversation"
        extra = f"\n{_judgement_block(turn_judgements)}\n" if turn_judgements else ""
    else:
        system = _DIRECT_SYSTEM.format(
            what="whether or not a response adequately addresses an instruction in the context of an image",
            extra="\n5. Focus points: some points you should consider when you make the judgement." if turn == NUM_TURNS else "",
            target=f"the {ORDINAL[turn]} turn response for the {SKILL[turn]} performance",
            overall_note="",
        )
        target = f"the {ORDINAL[turn]} turn response"
        extra = _focus_block(doc, turn)
    user = (
        "Here are the image context, the instructions, the high-quality references, and the responses.\n\n"
        f"Image context: {doc['caption']}\n\n"
        f"{_reference_block(doc)}\n\n{_conversation_block(doc, model_responses, 'Assistant A')}\n{extra}\n"
        f'Think step-by-step, rate {target} from Assistant A, and finish your response with "Rating: [[X]]" where X is on a scale of 1 to 10.'
    )
    return system, user


def _parse_preference(text):
    matches = re.findall(r"Overall,?\s*Response\s*\(?([AB])\)?\s*is\s*better", text, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"Response\s*\(?([AB])\)?", text)
    return matches[-1].upper() if matches else None


def _parse_rating(text):
    matches = re.findall(r"Rating:?\s*\[*\s*(\d+(?:\.\d+)?)\s*\]*", text, flags=re.IGNORECASE)
    if not matches:
        return None
    return min(max(float(matches[-1]), 1.0), 10.0)


def _grade(doc, model_responses, turn=None, turn_judgements=None):
    """Returns (score, judgement). Pairwise score is 1 if the judge prefers the model over the reference."""
    if GRADING == "pairwise":
        # Seeded per (doc, turn): the A/B order is identical across runs, so compression settings are compared fairly.
        model_is_a = random.Random(f"convbench-{doc['id']}-{turn or 0}").random() < 0.5
        system, user = _pairwise_prompts(doc, model_responses, model_is_a, turn, turn_judgements)
        judgement = _judge(system, user)
        choice = _parse_preference(judgement)
        score = None if choice is None else float(choice == ("A" if model_is_a else "B"))
    else:
        system, user = _direct_prompts(doc, model_responses, turn, turn_judgements)
        judgement = _judge(system, user)
        score = _parse_rating(judgement)
    if score is None:
        eval_logger.warning(f"ConvBench doc {doc['id']} turn {turn or 'overall'}: could not parse judge output; excluded from the mean.")
    return score, judgement


# ---------------------------------------------------------------- results and metrics


def _process(doc, results, n_ref):
    raw = results[0]
    raw = [raw] if isinstance(raw, (str, dict)) else list(raw)
    reference_scores = next((r for r in raw if isinstance(r, dict)), None)
    outputs = [o for o in raw if isinstance(o, str)] + [""] * NUM_TURNS
    responses = [doc[f"reference_{k}"] if k <= n_ref else outputs[k - 1].strip() for k in range(1, NUM_TURNS + 1)]
    task = TASK_NAMES[n_ref]

    out = _judge_metrics(doc, responses, n_ref) if GRADING != "none" else {}
    if TEXT_METRICS:
        pairs = [[k, responses[k - 1], doc[f"reference_{k}"]] for k in range(n_ref + 1, NUM_TURNS + 1)]
        text_item = {"id": doc["id"], "task": task, "pairs": pairs}
        out.update({f"convbench_{m}": text_item for m in TEXT_METRIC_NAMES})
    turns = _turn_records(doc, responses, n_ref, reference_scores)
    if reference_scores:
        ppl_item = {
            "id": doc["id"],
            "task": task,
            "nll": reference_scores["reference_nll"],
            "tokens": reference_scores["reference_tokens"],
            "truncated": reference_scores["reference_truncated"],
        }
        # the record the samples log shows: per-turn question, model output, reference and PPL, then the conversation PPL
        out["convbench_PPL"] = {**ppl_item, "turns": turns, "ppl": _ppl(sum(ppl_item["nll"]), sum(ppl_item["tokens"]))}
        out.update({f"convbench_PPL_turn{k}": ppl_item for k in range(1, NUM_TURNS + 1)})
    if LOG_TURNS:
        _log_turns(doc, task, turns)
    return out


def _ppl(nll, tokens):
    return math.exp(nll / tokens) if tokens else None


def _turn_records(doc, responses, n_ref, reference_scores):
    """One readable record per turn: question, the answer in the history, the reference and (if scored) its PPL."""
    turns = []
    for k in range(1, NUM_TURNS + 1):
        turn = {
            "turn": k,
            "question": doc[f"instruction_{k}"],
            "model_output": None if k <= n_ref else responses[k - 1],
            "reference": doc[f"reference_{k}"],
        }
        if reference_scores and k <= len(reference_scores["reference_nll"]):
            nll, tokens = reference_scores["reference_nll"][k - 1], reference_scores["reference_tokens"][k - 1]
            turn.update(nll=nll, tokens=tokens, ppl=_ppl(nll, tokens), truncated=reference_scores["reference_truncated"][k - 1])
        turns.append(turn)
    return turns


def _log_turns(doc, task, turns):
    lines = [f"ConvBench {task} conversation {doc['id']}:"]
    for t in turns:
        ppl = f" | PPL of the reference {t['ppl']:.3f} over {t['tokens']} tokens" if t.get("ppl") is not None else ""
        output = "(reference given as history)" if t["model_output"] is None else t["model_output"]
        lines += [f"  [turn {t['turn']}]{ppl}", f"    question:  {t['question']}", f"    model:     {output}", f"    reference: {t['reference']}"]
    eval_logger.info("\n".join(lines))


def _judge_metrics(doc, responses, n_ref):
    turn_scores, turn_judgements = {}, {}
    for turn in range(n_ref + 1, NUM_TURNS + 1):
        turn_scores[turn], turn_judgements[turn] = _grade(doc, responses, turn=turn)
    overall, overall_judgement = _grade(doc, responses, turn=None, turn_judgements=turn_judgements)

    out = {}
    for turn in turn_scores:
        out[f"convbench_S{turn}"] = {"id": doc["id"], "grading": GRADING, "score": turn_scores[turn], "judgement": turn_judgements[turn]}
    out["convbench_S0"] = {"id": doc["id"], "grading": GRADING, "score": overall, "judgement": overall_judgement}
    if n_ref == 0:
        summary = {"id": doc["id"], "grading": GRADING, "turns": [turn_scores[k] for k in range(1, NUM_TURNS + 1)], "overall": overall}
        out["convbench_R2"] = summary
        out["convbench_R1"] = summary
    return out


def convbench_process_results(doc, results):
    return _process(doc, results, n_ref=0)


def convbench_ref1_process_results(doc, results):
    return _process(doc, results, n_ref=1)


def convbench_ref2_process_results(doc, results):
    return _process(doc, results, n_ref=2)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _scale(items):
    return 100.0 if items and items[0].get("grading") == "pairwise" else 1.0


def _report(value, n_valid, n_total, name):
    if n_valid < n_total:
        eval_logger.warning(f"ConvBench {name}: {n_total - n_valid}/{n_total} judgements unparsable and excluded.")
    return round(value, 2) if value is not None else None


def convbench_aggregate_score(results):
    """S0-S3: pairwise win rate (%) vs. the human-verified reference, or mean 1-10 rating."""
    valid = [r["score"] for r in results if r.get("score") is not None]
    mean = _mean(valid)
    return _report(None if mean is None else mean * _scale(results), len(valid), len(results), "score")


def _turn_means(results):
    return [_mean([r["turns"][k] for r in results]) for k in range(NUM_TURNS)]


def convbench_aggregate_r2(results):
    """R2 = (S1 + S2 + S3) / 3."""
    means = _turn_means(results)
    if any(m is None for m in means):
        return None
    return round(sum(means) / NUM_TURNS * _scale(results), 2)


def convbench_aggregate_r1(results):
    """R1 = (R2 + S0) / 2."""
    means = _turn_means(results)
    s0 = _mean([r["overall"] for r in results])
    if s0 is None or any(m is None for m in means):
        return None
    return round((sum(means) / NUM_TURNS + s0) / 2 * _scale(results), 2)


# ---------------------------------------------------------------- reference-based metrics

TEXT_METRIC_NAMES = ("BLEU1", "BLEU2", "BLEU3", "BLEU4", "METEOR", "ROUGE_L")
_text_score_cache = {}


def _write_breakdown(args, task, section, data):
    """Adds `section` to <output_path>/submissions/<task>_by_turn_<run tag>.json."""
    path = generate_submission_file(f"{task}_by_turn_{RUN_TAG}.json", args)
    breakdown = json.load(open(path)) if os.path.exists(path) else {}
    breakdown[section] = data
    with open(path, "w") as f:
        json.dump(breakdown, f, indent=2)


def _text_scores(items):
    """BLEU-1..4, METEOR and ROUGE-L (x100), pooled over all scored turns and per turn. Computed once per item set."""
    pairs = [(f"{item['id']}_{turn}", turn, pred, ref) for item in items for turn, pred, ref in item["pairs"]]
    key = hashlib.sha1(json.dumps(pairs).encode()).hexdigest()
    if key in _text_score_cache:
        return _text_score_cache[key]

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    eval_logger.info(f"ConvBench: computing BLEU/METEOR/ROUGE on {len(pairs)} turns...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize({k: [{"caption": ref}] for k, _, _, ref in pairs})
    res = tokenizer.tokenize({k: [{"caption": pred}] for k, _, pred, _ in pairs})
    groups = {"all": [p[0] for p in pairs]}
    for turn in sorted({p[1] for p in pairs}):
        groups[f"turn{turn}"] = [p[0] for p in pairs if p[1] == turn]

    meteor = Meteor()
    scores = {}
    for name, keys in groups.items():
        g, r = {k: gts[k] for k in keys}, {k: res[k] for k in keys}
        bleu, _ = Bleu(4).compute_score(g, r, verbose=0)
        scores[name] = {f"BLEU{n + 1}": bleu[n] * 100 for n in range(4)}
        scores[name]["METEOR"] = meteor.compute_score(g, r)[0] * 100
        scores[name]["ROUGE_L"] = Rouge().compute_score(g, r)[0] * 100
        scores[name]["n_turns"] = len(keys)
    del meteor
    _text_score_cache[key] = scores
    return scores


def _text_metric(results, metric, args):
    scores = _text_scores(results)
    _write_breakdown(args, results[0]["task"], "text_metrics", scores)
    return round(scores["all"][metric], 2)


def convbench_bleu1(results, args=None):
    return _text_metric(results, "BLEU1", args)


def convbench_bleu2(results, args=None):
    return _text_metric(results, "BLEU2", args)


def convbench_bleu3(results, args=None):
    return _text_metric(results, "BLEU3", args)


def convbench_bleu4(results, args=None):
    return _text_metric(results, "BLEU4", args)


def convbench_meteor(results, args=None):
    return _text_metric(results, "METEOR", args)


def convbench_rouge_l(results, args=None):
    return _text_metric(results, "ROUGE_L", args)


def _ppl_breakdown(results):
    """PPL = exp(total NLL / total tokens) over all turns ("all") and per turn ("turn1".."turn3")."""
    breakdown = {}
    for name, turns in [("all", range(NUM_TURNS))] + [(f"turn{k + 1}", [k]) for k in range(NUM_TURNS)]:
        nll = sum(r["nll"][k] for r in results for k in turns if k < len(r["nll"]))
        tokens = sum(r["tokens"][k] for r in results for k in turns if k < len(r["tokens"]))
        truncated = sum(r["truncated"][k] for r in results for k in turns if k < len(r["truncated"]))
        breakdown[name] = {"ppl": _ppl(nll, tokens), "tokens": tokens, "truncated_references": truncated}
    return breakdown


def _rounded(ppl):
    return round(ppl, 3) if ppl is not None else None


def convbench_aggregate_ppl(results, args=None):
    """Overall PPL, pooled over all turns; also writes the per-turn values and every conversation's turns to submissions/."""
    breakdown = _ppl_breakdown(results)
    if breakdown["all"]["truncated_references"]:
        eval_logger.warning(f"ConvBench PPL: {breakdown['all']['truncated_references']} reference(s) were scored only up to the token cap.")
    task = results[0]["task"]
    _write_breakdown(args, task, "perplexity", breakdown)
    if all("turns" in r for r in results):
        with open(generate_submission_file(f"{task}_conversations_{RUN_TAG}.jsonl", args), "w") as f:
            for r in results:
                f.write(json.dumps({"id": r["id"], "ppl": r.get("ppl"), "turns": r["turns"]}) + "\n")
    return _rounded(breakdown["all"]["ppl"])


def convbench_aggregate_ppl_turn1(results, args=None):
    return _rounded(_ppl_breakdown(results)["turn1"]["ppl"])


def convbench_aggregate_ppl_turn2(results, args=None):
    return _rounded(_ppl_breakdown(results)["turn2"]["ppl"])


def convbench_aggregate_ppl_turn3(results, args=None):
    return _rounded(_ppl_breakdown(results)["turn3"]["ppl"])
