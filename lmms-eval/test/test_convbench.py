"""Tests for the ConvBench task and the multi-round generation it relies on.

Run from the TGV-KV repository root:  pytest lmms-eval/test/test_convbench.py
The model tests need torch, transformers, decord and qwen-vl-utils and are skipped without them.
"""

import functools
import json
import math
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]  # TGV-KV root, needed for `utils.*` imports in the models
sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.tasks.convbench import utils as cb  # noqa: E402


def make_doc(doc_id: int = 7) -> Dict[str, Any]:
    doc: Dict[str, Any] = {"id": doc_id, "caption": "A red car.", "focus_points": "1. Is it catchy?", "image_path": ""}
    for k in (1, 2, 3):
        doc[f"instruction_{k}"] = f"instruction {k}"
        doc[f"reference_{k}"] = f"reference {k}"
    return doc


# ---------------------------------------------------------------- prompts / history


def test_round_zero_is_first_instruction() -> None:
    assert cb.convbench_doc_to_text(make_doc()) == "instruction 1"


@pytest.mark.parametrize(
    "n_ref, expected_assistant_turns",
    [
        (0, ["model 1", "model 2"]),
        (1, ["reference 1", "model 2"]),
        (2, ["reference 1", "reference 2"]),
    ],
)
def test_history_uses_model_answers_or_references(n_ref: int, expected_assistant_turns: List[str]) -> None:
    visuals, history, terminal, _, _ = cb.convbench_doc_to_text(make_doc(), {"n_ref_turns": n_ref}, previous_output=["model 1", "model 2"], round_idx=2)
    assert visuals is None and terminal is False
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant", "user"]
    assert [m["content"] for m in history if m["role"] == "user"] == ["instruction 1", "instruction 2", "instruction 3"]
    assert [m["content"] for m in history if m["role"] == "assistant"] == expected_assistant_turns


def test_terminal_after_three_rounds() -> None:
    _, context, terminal, _, _ = cb.convbench_doc_to_text(make_doc(), {}, previous_output=["a", "b", "c"], round_idx=3)
    assert terminal is True and context is None


def test_reference_round_uses_the_reference_history() -> None:
    history, target = cb.convbench_doc_to_text(make_doc(), {}, reference_round=3)
    assert target == "reference 3"
    assert [m["content"] for m in history] == ["instruction 1", "reference 1", "instruction 2", "reference 2", "instruction 3"]
    assert cb.convbench_doc_to_text(make_doc(), {}, reference_round=4) is None


def test_reference_scoring_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cb.convbench_doc_to_text(make_doc(), {"score_references": False}, reference_round=1) is None
    monkeypatch.setattr(cb, "COMPUTE_PPL", False)
    assert cb.convbench_doc_to_text(make_doc(), {}, reference_round=1) is None


# ---------------------------------------------------------------- judging


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replaces the LLM judge; set `judge.reply`, inspect `judge.calls`."""
    state = SimpleNamespace(calls=[], reply="Overall, Response A is better.")

    def fake_judge(system: str, user: str) -> str:
        state.calls.append({"system": system, "user": user})
        return state.reply

    monkeypatch.setattr(cb, "_judge", fake_judge)
    return state


def test_pairwise_score_follows_the_model_position(monkeypatch: pytest.MonkeyPatch, judge: SimpleNamespace) -> None:
    monkeypatch.setattr(cb, "GRADING", "pairwise")
    doc = make_doc()
    out = cb.convbench_process_results(doc, [("model 1", "model 2", "model 3")])
    for key, turn in [("convbench_S1", 1), ("convbench_S2", 2), ("convbench_S3", 3), ("convbench_S0", 0)]:
        model_is_a = random.Random(f"convbench-{doc['id']}-{turn}").random() < 0.5
        assert out[key]["score"] == float(model_is_a), key  # judge always picks A
    assert len(judge.calls) == 4
    assert "1. Is it catchy?" in judge.calls[2]["user"] and "1. Is it catchy?" not in judge.calls[0]["user"]
    assert "The third turn evaluation: Overall, Response A is better." in judge.calls[3]["user"]


def test_reference_settings_only_judge_generated_turns(monkeypatch: pytest.MonkeyPatch, judge: SimpleNamespace) -> None:
    monkeypatch.setattr(cb, "GRADING", "pairwise")
    out = cb.convbench_ref2_process_results(make_doc(), [("model 1", "model 2", "model 3")])
    assert {k for k in out if k.startswith("convbench_S")} == {"convbench_S3", "convbench_S0"}
    assert len(judge.calls) == 2
    assert "reference 2" in judge.calls[0]["user"] and "model 2" not in judge.calls[0]["user"]


def test_unparsable_judgement_is_excluded(monkeypatch: pytest.MonkeyPatch, judge: SimpleNamespace) -> None:
    monkeypatch.setattr(cb, "GRADING", "pairwise")
    judge.reply = "I cannot decide."
    out = cb.convbench_process_results(make_doc(), [("a", "b", "c")])
    assert out["convbench_S1"]["score"] is None
    good = {"grading": "pairwise", "score": 1.0}
    assert cb.convbench_aggregate_score([out["convbench_S1"], good, dict(good, score=0.0)]) == 50.0


def test_direct_mode_prompts_include_the_reference(monkeypatch: pytest.MonkeyPatch, judge: SimpleNamespace) -> None:
    monkeypatch.setattr(cb, "GRADING", "direct")
    judge.reply = "Good. Rating: [[8]]"
    out = cb.convbench_process_results(make_doc(), [("a", "b", "c")])
    assert out["convbench_S2"] == {"id": 7, "grading": "direct", "score": 8.0, "judgement": "Good. Rating: [[8]]"}
    assert all("reference 3" in call["user"] for call in judge.calls)


# ---------------------------------------------------------------- reference-based metrics


def test_judge_free_results_carry_text_pairs_and_perplexity(monkeypatch: pytest.MonkeyPatch, judge: SimpleNamespace) -> None:
    monkeypatch.setattr(cb, "GRADING", "none")
    scores = {"reference_nll": [2.0, 4.0, 6.0], "reference_tokens": [2, 2, 2], "reference_truncated": [False, False, True]}
    out = cb.convbench_process_results(make_doc(), [("a", "b", "c", scores)])
    assert judge.calls == []
    assert not {"convbench_S0", "convbench_S1", "convbench_S2", "convbench_S3", "convbench_R1", "convbench_R2"} & set(out)
    assert out["convbench_BLEU4"]["pairs"] == [[1, "a", "reference 1"], [2, "b", "reference 2"], [3, "c", "reference 3"]]
    assert out["convbench_METEOR"] is out["convbench_BLEU1"]  # one shared item, scored once
    assert out["convbench_PPL"]["nll"] == [2.0, 4.0, 6.0]
    assert out["convbench_PPL"]["ppl"] == pytest.approx(math.exp(12.0 / 6))
    assert [t["model_output"] for t in out["convbench_PPL"]["turns"]] == ["a", "b", "c"]
    assert [t["reference"] for t in out["convbench_PPL"]["turns"]] == ["reference 1", "reference 2", "reference 3"]
    assert [t["ppl"] for t in out["convbench_PPL"]["turns"]] == pytest.approx([math.exp(1.0), math.exp(2.0), math.exp(3.0)])
    assert {f"convbench_PPL_turn{k}" for k in (1, 2, 3)} <= set(out)
    ablation = cb.convbench_ref1_process_results(make_doc(), [("a", "b", "c")])
    assert [p[0] for p in ablation["convbench_ROUGE_L"]["pairs"]] == [2, 3] and "convbench_PPL" not in ablation


def test_perplexity_is_token_weighted_and_written_per_turn(tmp_path: Path) -> None:
    rows = [
        {"task": "convbench", "nll": [2.0, 4.0, 6.0], "tokens": [2, 2, 2], "truncated": [False, False, False]},
        {"task": "convbench", "nll": [0.0, 0.0, 6.0], "tokens": [2, 2, 6], "truncated": [False, False, True]},
    ]
    ppl = cb.convbench_aggregate_ppl(rows, args=SimpleNamespace(output_path=str(tmp_path)))
    assert ppl == pytest.approx(math.exp(18.0 / 16), abs=1e-3)
    breakdown = json.loads(next((tmp_path / "submissions").glob("convbench_by_turn_*.json")).read_text())
    assert breakdown["perplexity"]["turn3"] == {"ppl": pytest.approx(math.exp(12.0 / 8)), "tokens": 8, "truncated_references": 1}
    per_turn = [cb.convbench_aggregate_ppl_turn1(rows), cb.convbench_aggregate_ppl_turn2(rows), cb.convbench_aggregate_ppl_turn3(rows)]
    assert per_turn == [pytest.approx(math.exp(2.0 / 4), abs=1e-3), pytest.approx(math.exp(4.0 / 4), abs=1e-3), pytest.approx(math.exp(12.0 / 8), abs=1e-3)]


def test_every_conversation_is_logged_with_outputs_references_and_ppl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cb, "GRADING", "none")
    monkeypatch.setattr(cb, "LOG_TURNS", True)
    logged: List[str] = []
    monkeypatch.setattr(cb.eval_logger, "info", logged.append)
    scores = {"reference_nll": [2.0, 4.0, 6.0], "reference_tokens": [2, 2, 2], "reference_truncated": [False, False, True]}
    record = cb.convbench_process_results(make_doc(), [("a", "b", "c", scores)])["convbench_PPL"]
    assert "[turn 2] | PPL of the reference 7.389 over 2 tokens" in logged[0]
    assert "    model:     b" in logged[0] and "    reference: reference 2" in logged[0]

    cb.convbench_aggregate_ppl([record], args=SimpleNamespace(output_path=str(tmp_path)))
    saved = json.loads(next((tmp_path / "submissions").glob("convbench_conversations_*.jsonl")).read_text())
    assert saved["id"] == 7 and saved["ppl"] == pytest.approx(math.exp(2.0))
    assert [(t["turn"], t["question"], t["model_output"], t["reference"]) for t in saved["turns"]] == [
        (1, "instruction 1", "a", "reference 1"),
        (2, "instruction 2", "b", "reference 2"),
        (3, "instruction 3", "c", "reference 3"),
    ]


@pytest.mark.skipif(shutil.which("java") is None, reason="METEOR and the PTB tokenizer need Java")
def test_text_metrics_score_100_on_exact_matches_and_survive_empty_answers(tmp_path: Path) -> None:
    ref = "The vehicle is a Hummer H2.\n\nIt has a CPTLISM plate."
    perfect = [{"id": 1, "task": "convbench", "pairs": [[1, ref, ref], [2, ref, ref]]}]
    args = SimpleNamespace(output_path=str(tmp_path))
    assert [cb.convbench_bleu4(perfect, args=args), cb.convbench_meteor(perfect, args=args), cb.convbench_rouge_l(perfect, args=args)] == [100.0, 100.0, 100.0]
    degenerate = [{"id": 2, "task": "convbench", "pairs": [[1, "", ref], [2, "...", ref]]}]
    assert cb.convbench_bleu1(degenerate, args=args) == 0.0
    breakdown = json.loads(next((tmp_path / "submissions").glob("convbench_by_turn_*.json")).read_text())
    assert set(breakdown["text_metrics"]) == {"all", "turn1", "turn2"}


@pytest.mark.parametrize("text, expected", [("... Rating: [[7]]", 7.0), ("Rating:[[12]]", 10.0), ("Rating: 0", 1.0), ("no score", None)])
def test_direct_rating_parsing(text: str, expected: Any) -> None:
    assert cb._parse_rating(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("Overall, Response B is better.", "B"), ("Response A is fine but Overall, Response B is better", "B"), ("Response A", "A"), ("tie", None)],
)
def test_pairwise_preference_parsing(text: str, expected: Any) -> None:
    assert cb._parse_preference(text) == expected


def test_aggregates_follow_the_paper_formulas() -> None:
    rows = [
        {"grading": "pairwise", "turns": [1.0, 0.0, 1.0], "overall": 1.0},
        {"grading": "pairwise", "turns": [0.0, 0.0, 1.0], "overall": 0.0},
    ]
    # S1=50, S2=0, S3=100 -> R2=50; S0=50 -> R1=50
    assert cb.convbench_aggregate_r2(rows) == 50.0
    assert cb.convbench_aggregate_r1(rows) == 50.0
    direct = [dict(r, grading="direct") for r in rows]
    assert cb.convbench_aggregate_r2(direct) == 0.5


# ---------------------------------------------------------------- multi-round generation in the models


def fake_request(doc: Dict[str, Any], n_ref: int = 0) -> SimpleNamespace:
    doc_to_text = functools.partial(cb.convbench_doc_to_text, lmms_eval_specific_kwargs={"n_ref_turns": n_ref})
    return SimpleNamespace(args=("instruction 1", {"max_new_tokens": 8, "until": ["\n\n"]}, lambda d: [Image.new("RGB", (32, 32))], doc_to_text, 0, "convbench", "test"))


def test_llava_hf_multi_round_keeps_history_and_attaches_image_once(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(cb, "COMPUTE_PPL", False)  # scoring is covered by the forced-decoding test below
    monkeypatch.setenv("MULTI_TURN_KV_CARRYOVER", "0")  # the re-encode path; carry-over has its own tests below
    pytest.importorskip("decord")
    from lmms_eval.api.model import CacheHook
    from lmms_eval.models.simple.llava_hf import LlavaHf

    seen_messages: List[List[Dict[str, str]]] = []
    tokenizer = SimpleNamespace(
        chat_template="set",
        eos_token_id=2,
        apply_chat_template=lambda messages, **kw: seen_messages.append([dict(m) for m in messages]) or " ".join(m["content"] for m in messages),
        batch_decode=lambda ids, **kw: [f"answer {len(seen_messages)}"],
    )
    inputs = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    lm = object.__new__(LlavaHf)
    lm.__dict__.update(
        _tokenizer=tokenizer,
        chat_template=None,
        use_cache=True,
        _device="cpu",
        _rank=0,
        max_frames_num=4,
        cache_hook=CacheHook(None),
        _image_processor=lambda **kw: SimpleNamespace(to=lambda *a: inputs),
        _model=SimpleNamespace(dtype=torch.float32, generate=lambda **kw: torch.zeros(1, 9, dtype=torch.long)),
        task_dict={"convbench": {"test": [make_doc()]}},
    )
    out = lm.generate_until_multi_round([fake_request(make_doc())])
    assert out == [("answer 1", "answer 2", "answer 3")]
    assert [len(m) for m in seen_messages] == [1, 3, 5]
    last = seen_messages[-1]
    assert last[0]["content"].startswith("<image>\n") and sum("<image>" in m["content"] for m in last) == 1
    assert [m["content"] for m in last if m["role"] == "assistant"] == ["answer 1", "answer 2"]


def test_reencode_scores_each_reference_after_the_models_own_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """PPL_HISTORY=model (default): reference k is scored on the same history answer k is generated from."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(cb, "COMPUTE_PPL", True)
    monkeypatch.setenv("MULTI_TURN_KV_CARRYOVER", "0")
    monkeypatch.delenv("PPL_HISTORY", raising=False)
    pytest.importorskip("decord")
    import lmms_eval.models.simple.llava_hf as llava_module
    from lmms_eval.api.model import CacheHook

    seen_messages: List[List[Dict[str, str]]] = []
    scored: List[Any] = []

    class Tokenizer:
        chat_template = "set"
        eos_token_id = 2

        def __call__(self, text: str, add_special_tokens: bool = True) -> Dict[str, List[str]]:
            return {"input_ids": text.split()}

        def apply_chat_template(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            seen_messages.append([dict(m) for m in messages])
            return " ".join(m["content"] for m in messages)

        def batch_decode(self, ids: Any, **kw: Any) -> List[str]:
            return [f"answer {len(outputs_so_far)}"]  # generate() appended this turn already

    outputs_so_far: List[str] = []
    monkeypatch.setattr(llava_module, "forced_decode_nll", lambda model, inputs, ids, eos, pad: scored.append((seen_messages[-1], ids)) or (1.0, len(ids)))
    lm = object.__new__(llava_module.LlavaHf)
    lm.__dict__.update(
        _tokenizer=Tokenizer(),
        chat_template=None,
        use_cache=True,
        _device="cpu",
        _rank=0,
        max_frames_num=4,
        cache_hook=CacheHook(None),
        _image_processor=lambda **kw: SimpleNamespace(to=lambda *a: {"input_ids": torch.zeros(1, 5, dtype=torch.long)}),
        _model=SimpleNamespace(dtype=torch.float32, generate=lambda **kw: outputs_so_far.append("x") or torch.zeros(1, 9, dtype=torch.long)),
        task_dict={"convbench": {"test": [make_doc()]}},
    )
    out = lm.generate_until_multi_round([fake_request(make_doc())])
    *answers, ppl = out[0]
    assert answers == ["answer 1", "answer 2", "answer 3"]
    assert ppl == {"reference_nll": [1.0, 1.0, 1.0], "reference_tokens": [2, 2, 2], "reference_truncated": [False, False, False]}
    assert [ids for _, ids in scored] == [["reference", "1"], ["reference", "2"], ["reference", "3"]]
    # reference k after Q1, A1, ..., Qk with the model's own answers, not the references
    assert [[m["content"] for m in history if m["role"] == "assistant"] for history, _ in scored] == [[], ["answer 1"], ["answer 1", "answer 2"]]
    assert [history[-1]["content"] for history, _ in scored] == ["<image>\ninstruction 1", "instruction 2", "instruction 3"]


# A llava-hf/llava-1.5-*-hf style template: it reads typed content parts and renders plain strings as nothing.
HF_STYLE_TEMPLATE = (
    "{% for message in messages %}{{ message['role'].upper() + ': ' }}"
    "{% for c in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}"
    "{% for c in message['content'] | selectattr('type', 'equalto', 'text') %}{{ c['text'] + ' ' }}{% endfor %}"
    "{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)


def _render(messages: List[Dict[str, Any]], add_generation_prompt: bool = False, **kw: Any) -> str:
    import jinja2

    return jinja2.Template(HF_STYLE_TEMPLATE).render(messages=messages, add_generation_prompt=add_generation_prompt)


def _llava_with(processor_template: Any, tokenizer_template: Any) -> Any:
    torch = pytest.importorskip("torch")
    pytest.importorskip("decord")
    from lmms_eval.models.simple.llava_hf import LlavaHf

    class FakeProcessor:
        chat_template = processor_template

        def apply_chat_template(self, messages: List[Dict[str, Any]], tokenize: bool = False, add_generation_prompt: bool = False) -> str:
            return _render(messages, add_generation_prompt)

        def __call__(self, **kw: Any) -> SimpleNamespace:
            return SimpleNamespace(to=lambda *a: {"input_ids": torch.zeros(1, 5, dtype=torch.long)})

    lm = object.__new__(LlavaHf)
    tokenizer = SimpleNamespace(chat_template=tokenizer_template, apply_chat_template=_render)
    lm.__dict__.update(_tokenizer=tokenizer, chat_template=None, _device="cpu", _image_processor=FakeProcessor(), _model=SimpleNamespace(dtype=None))
    return lm


def test_llava_hf_uses_typed_content_for_llava_hf_chat_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: plain-string messages rendered to 'USER: ASSISTANT:' with llava-hf templates (no question, no image token)."""
    lm = _llava_with(processor_template=HF_STYLE_TEMPLATE, tokenizer_template=HF_STYLE_TEMPLATE)
    rendered: List[str] = []
    monkeypatch.setattr(type(lm._image_processor), "apply_chat_template", lambda self, m, **kw: rendered.append(_render(m, **kw)) or rendered[-1])
    history = [{"role": "user", "content": "instruction 1"}, {"role": "assistant", "content": "answer 1"}, {"role": "user", "content": "instruction 2"}]
    lm._chat_inputs(history, [Image.new("RGB", (8, 8))])
    assert rendered == ["USER: <image>\ninstruction 1 ASSISTANT: answer 1 USER: instruction 2 ASSISTANT:"]


def test_llava_hf_raises_when_the_template_drops_the_question() -> None:
    lm = _llava_with(processor_template=None, tokenizer_template=HF_STYLE_TEMPLATE)  # forces the plain-string path
    with pytest.raises(ValueError, match="dropped the user message"):
        lm._chat_inputs([{"role": "user", "content": "What device is shown in the image?"}], [Image.new("RGB", (8, 8))])


def test_qwen3_vl_multi_round_keeps_history_and_attaches_image_once(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(cb, "COMPUTE_PPL", False)
    monkeypatch.setenv("MULTI_TURN_KV_CARRYOVER", "0")
    pytest.importorskip("decord")
    pytest.importorskip("qwen_vl_utils")
    from lmms_eval.api.model import CacheHook
    from lmms_eval.models.simple.qwen3_vl import Qwen3_VL

    seen_chats: List[List[Dict[str, Any]]] = []

    class FakeInputs(dict):
        input_ids = torch.zeros(1, 5, dtype=torch.long)

        def to(self, *a: Any) -> "FakeInputs":
            return self

    class FakeProcessor:
        def apply_chat_template(self, chats: List[Any], **kw: Any) -> List[str]:
            seen_chats.append(chats[0])
            return ["PROMPT"]

        def batch_decode(self, ids: Any, **kw: Any) -> List[str]:
            return [f"answer {len(seen_chats)}"]

        def __call__(self, **kw: Any) -> FakeInputs:
            return FakeInputs()

    lm = object.__new__(Qwen3_VL)
    lm.__dict__.update(
        processor=FakeProcessor(),
        _tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=0, decode=lambda ids: "<|im_end|>"),
        system_prompt="You are a helpful assistant.",
        reasoning_prompt=None,
        max_pixels=64 * 64,
        min_pixels=32 * 32,
        max_num_frames=4,
        device_map="",
        _device="cpu",
        use_cache=True,
        _rank=0,
        cache_hook=CacheHook(None),
        _model=SimpleNamespace(generate=lambda **kw: torch.zeros(1, 9, dtype=torch.long)),
        task_dict={"convbench": {"test": [make_doc()]}},
    )
    out = lm.generate_until_multi_round([fake_request(make_doc(), n_ref=1)])
    assert out == [("answer 1", "answer 2", "answer 3")]
    last = seen_chats[-1]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert [part["type"] for part in last[1]["content"]] == ["image", "text"]
    assert all(part["type"] == "text" for m in last[2:] for part in m["content"])
    # n_ref_turns=1: the reference replaces the model's turn-1 answer in the history
    assert [m["content"][0]["text"] for m in last if m["role"] == "assistant"] == ["reference 1", "answer 2"]


def test_forced_decoding_through_patched_generate_matches_teacher_forcing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no KV compression, scoring via TGV-KV's patched `generate` must equal one teacher-forced forward pass."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import types

    from lmms_eval.models.model_utils.reference_scoring import forced_decode_nll

    import utils.generate_patches as patches

    # Full KV cache. (TGV-KV's own fallback stub for an unset MODEL_TYPE cannot take `prune_ratio`, so stub it here.)
    monkeypatch.setattr(patches, "get_kv_cache", lambda *args, **kwargs: None)
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = transformers.LlamaForCausalLM._from_config(config, attn_implementation="eager").eval()
    model.generate = types.MethodType(patches.generate, model)
    prompt = torch.tensor([[1, 5, 17, 33, 8, 60]])
    target = [11, 42, 7, 90, 3, 25, 64]

    full = torch.cat([prompt, torch.tensor([target])], dim=1)
    with torch.no_grad():
        logits = model(input_ids=full).logits[0, prompt.shape[1] - 1 : -1].float()
    expected = -torch.log_softmax(logits, -1).gather(1, torch.tensor(target)[:, None]).sum().item()

    nll, n = forced_decode_nll(model, {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}, target, eos_token_id=2, pad_token_id=0)
    assert n == len(target)
    assert nll == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------- multi-turn KV carry-over


def test_continuation_ids_are_the_tokens_after_the_previous_answer() -> None:
    from lmms_eval.models.model_utils.kv_carryover import continuation_ids

    tokenize = lambda text: text.split()  # noqa: E731
    prev = "USER: q1 ASSISTANT:"
    nxt = "USER: q1 ASSISTANT: a b USER: q2 ASSISTANT:"
    assert continuation_ids(prev, "a b", nxt, tokenize) == ["USER:", "q2", "ASSISTANT:"]
    with pytest.raises(RuntimeError, match="renders earlier turns differently"):
        continuation_ids("USER: q1  ASSISTANT:", "a b", nxt, tokenize)


@pytest.mark.parametrize("n_ref, fixed", [(0, [None, None]), (1, ["reference 1", None]), (2, ["reference 1", "reference 2"])])
def test_fixed_answers_are_detected_without_generating(n_ref: int, fixed: List[Any]) -> None:
    from lmms_eval.models.model_utils.kv_carryover import _fixed_answer

    doc_to_text = functools.partial(cb.convbench_doc_to_text, lmms_eval_specific_kwargs={"n_ref_turns": n_ref})
    assert [_fixed_answer(doc_to_text, make_doc(), outputs, None) for outputs in ([], ["x"])] == fixed
    assert _fixed_answer(doc_to_text, make_doc(), ["x", "y"], None) is None  # the last turn is always generated


def test_continuation_mask_is_causal_and_aligned_to_the_end() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from utils.continuation_mask import continuation_causal_mask

    mask = continuation_causal_mask(torch.zeros(1, 1, 3, 7))
    visible = (mask == 0).int().tolist()
    assert visible == [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]


class _IdsAdapter:
    """ChatAdapter-like object for a model driven with raw token ids (no tokenizer)."""

    def __init__(self, first_inputs: Dict[str, Any]) -> None:
        self.first = first_inputs
        self.generate_kwargs = {"max_new_tokens": 6, "do_sample": False, "eos_token_id": None, "pad_token_id": 0}
        self.eos_ids: set = set()
        self.pad_id = 0
        self.image_token_ids = {32000}


def _tiny_llava(num_layers: int):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import types

    import utils.generate_patches as patches
    from utils.model_llama_patches import patch_llama

    patch_llama()
    torch.manual_seed(0)
    config = transformers.LlavaConfig(
        vision_config=transformers.CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14, projection_dim=16),
        text_config=transformers.LlamaConfig(
            vocab_size=32064, hidden_size=32, intermediate_size=64, num_hidden_layers=num_layers, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512, bos_token_id=1, eos_token_id=2, pad_token_id=0
        ),
        image_token_index=32000,
        vision_feature_select_strategy="default",
    )
    model = transformers.LlavaForConditionalGeneration._from_config(config, attn_implementation="eager").eval()
    model.generate = types.MethodType(patches.generate, model)
    ids = [1, 5, 6] + [32000] * 4 + list(range(100, 112))  # BOS, text, 4 image tokens (28/14)^2, question
    first = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long), "pixel_values": torch.randn(1, 3, 28, 28)}
    return torch, model, first


def _three_turns(conv, turn_ids):
    produced = [conv.generate()]
    for ids in turn_ids:
        conv.extend(ids)
        produced.append(conv.generate())
    return produced


def test_carryover_without_compression_equals_one_forward_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full KV: every token generated in turns 1-3 is the argmax of a single fresh forward pass over the conversation."""
    monkeypatch.delenv("KV_CACHE_TYPE", raising=False)
    monkeypatch.setattr("utils.generate_patches.get_kv_cache", lambda *a, **k: None, raising=False)
    torch, model, first = _tiny_llava(num_layers=2)
    from lmms_eval.models.model_utils.kv_carryover import CarryOverConversation

    conv = CarryOverConversation(model, _IdsAdapter(first))
    conv.adapter.generate_kwargs["output_logits"] = True
    step_logits, starts = [], []
    generate = model.generate

    def capture(**kwargs):
        out = generate(**kwargs)
        starts.append(kwargs["input_ids"].shape[1])  # the turn's first generated token sits at this position
        step_logits.append(torch.cat(out.logits))
        return out

    model.generate = capture
    conv.start(first)
    _three_turns(conv, [[200, 201, 202, 203], [210, 211, 212]])
    stream = conv.stream + conv.pending
    with torch.no_grad():
        fresh = model(input_ids=torch.tensor([stream]), pixel_values=first["pixel_values"]).logits[0]
    for start, logits in zip(starts, step_logits):  # carried-cache logits == one forward pass over the conversation
        torch.testing.assert_close(logits, fresh[start - 1 : start - 1 + len(logits)], rtol=1e-4, atol=1e-4)
    assert conv.cache.get_seq_length() == len(conv.stream)


def test_carryover_with_tgvkv_turn1_matches_original_and_flags_stay_in_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_TYPE", "llava-7B")  # TGV-KV's settings: image token 32000, 32 layers
    monkeypatch.setenv("KV_CACHE_TYPE", "tgv_kv")
    monkeypatch.setenv("PRUNE_RATIO", "0.5")
    monkeypatch.setenv("MULTI_TURN_DECODE_EVICTION", "1")
    monkeypatch.setenv("MULTI_TURN_BUDGET", "conversation")  # the original's decode budget grows with the tokens
    torch, model, first = _tiny_llava(num_layers=32)
    from lmms_eval.models.model_utils.kv_carryover import CarryOverConversation

    import utils.generate_patches as patches
    from kv_caches import get_kv_cache

    monkeypatch.setattr(patches, "get_kv_cache", get_kv_cache)
    original = model.generate(**first, max_new_tokens=6, do_sample=False, return_dict_in_generate=True, pad_token_id=0)
    conv = CarryOverConversation(model, _IdsAdapter(first))
    conv.start(first)
    assert conv.generate() == original.sequences[0, first["input_ids"].shape[1] :].tolist()
    assert all(torch.equal(a[0], b[0]) for a, b in zip(original.past_key_values, conv.cache))

    seen_positions, seen_inputs = [], []
    language_model = model.model.language_model
    hook = language_model.register_forward_pre_hook(lambda mod, args, kwargs: seen_positions.append(kwargs["position_ids"][0].tolist()), with_kwargs=True)
    hook_ids = model.register_forward_pre_hook(lambda mod, args, kwargs: seen_inputs.append(kwargs["input_ids"][0].tolist()), with_kwargs=True)
    for ids in ([200, 201, 202, 203, 204, 205], [210, 211, 212, 213]):
        start, expected_new = len(conv.stream), conv.pending + ids
        conv.extend(ids)
        seen_positions.clear()
        seen_inputs.clear()
        conv.generate()
        assert seen_inputs[0] == expected_new  # the prefill processes exactly the new tokens
        assert seen_positions[0] == list(range(start, start + len(expected_new)))  # at logical positions
        lengths = [p[0].shape[-2] for p in conv.cache]
        assert [f.numel() for f in conv.criteria.is_image] == lengths
        assert max(lengths) < len(conv.stream)  # compressed, not re-encoded
    hook.remove()
    hook_ids.remove()


def test_carryover_decode_eviction_off_keeps_every_generated_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_TYPE", "llava-7B")
    monkeypatch.setenv("KV_CACHE_TYPE", "tgv_kv")
    monkeypatch.setenv("PRUNE_RATIO", "0.5")
    monkeypatch.setenv("MULTI_TURN_DECODE_EVICTION", "0")
    torch, model, first = _tiny_llava(num_layers=32)
    from lmms_eval.models.model_utils.kv_carryover import CarryOverConversation

    conv = CarryOverConversation(model, _IdsAdapter(first))
    prefill_lengths = []
    fresh = conv.criteria._prefill_fresh
    conv.criteria._prefill_fresh = lambda *a: (lambda out: (prefill_lengths.extend(p[0].shape[-2] for p in out), out)[1])(fresh(*a))
    conv.start(first)
    conv.generate()
    per_layer = [p[0].shape[-2] for p in conv.cache]
    assert max(prefill_lengths) < first["input_ids"].shape[1]  # compressed at prefill
    assert per_layer == [n + 5 for n in prefill_lengths]  # then all 5 decoded tokens kept (6th is not processed yet)


def test_run_conversation_scores_each_reference_before_its_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reference k is scored on a copy of the cache right before answer k, after the model's own answers."""
    from lmms_eval.models.model_utils import kv_carryover as kc

    events: List[Any] = []

    class FakeConversation:
        def __init__(self, model: Any, adapter: Any) -> None:
            self.turn = 0

        def start(self, first_inputs: Any) -> None:
            events.append(("start", first_inputs))

        def extend(self, ids: List[str]) -> None:
            events.append(("extend", ids))

        def generate(self) -> List[str]:
            self.turn += 1
            events.append(("generate",))
            return ["A", str(self.turn)]

        def score_on_fork(self, ids: List[str]) -> Any:
            events.append(("score", ids))
            return 1.5, len(ids)

    monkeypatch.setattr(kc, "CarryOverConversation", FakeConversation)
    chat = lambda messages: " ".join(f"{m['role']}: {m['content']}" for m in messages) + " assistant:"  # noqa: E731
    adapter = SimpleNamespace(chat_text=chat, first_inputs=chat, tokenize=str.split, tokenize_answer=str.split, decode=" ".join)
    doc_to_text = functools.partial(cb.convbench_doc_to_text, lmms_eval_specific_kwargs={"n_ref_turns": 0})
    references = ["R1 x", "R2 y", "R3 z"]
    outputs, scores = kc.run_conversation(None, adapter, make_doc(), "instruction 1", doc_to_text, 8, references=references)

    assert outputs == ["A 1", "A 2", "A 3"]
    assert scores == {"reference_nll": [1.5, 1.5, 1.5], "reference_tokens": [2, 2, 2], "reference_truncated": [False, False, False]}
    assert [e[0] for e in events] == ["start", "score", "generate", "extend", "score", "generate", "extend", "score", "generate"]
    assert [e[1] for e in events if e[0] == "score"] == [r.split() for r in references]
    # the tokens added before scoring R2 follow the model's answer A1 in the history (not R1)
    assert events[3] == ("extend", "user: instruction 2 assistant:".split())


def test_reference_scored_on_a_fork_sees_the_answer_context_and_leaves_it_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_TYPE", "llava-7B")
    monkeypatch.setenv("KV_CACHE_TYPE", "tgv_kv")
    monkeypatch.setenv("PRUNE_RATIO", "0.5")
    monkeypatch.setenv("MULTI_TURN_DECODE_EVICTION", "0")
    torch, model, first = _tiny_llava(num_layers=32)
    from lmms_eval.models.model_utils.kv_carryover import CarryOverConversation

    import utils.generate_patches as patches
    from kv_caches import get_kv_cache

    monkeypatch.setattr(patches, "get_kv_cache", get_kv_cache)
    first_step_logits = []
    generate = model.generate

    def capture(**kwargs):
        out = generate(**kwargs)
        if getattr(out, "logits", None):
            first_step_logits.append(out.logits[0][0])
        return out

    model.generate = capture

    def conversation(score: bool):
        conv = CarryOverConversation(model, _IdsAdapter(first))
        conv.adapter.generate_kwargs["output_logits"] = True
        conv.start(first)
        conv.generate()
        conv.extend([200, 201, 202, 203, 204, 205])
        nll = conv.score_on_fork([300]) if score else None
        first_step_logits.clear()
        return conv, nll, conv.generate()

    plain, _, plain_answer = conversation(score=False)
    conv, (nll, n), answer = conversation(score=True)
    # the fork prefilled and compressed turn 2 exactly as the real turn 2 did: same next-token distribution
    assert n == 1
    assert nll == pytest.approx(-torch.log_softmax(first_step_logits[0].float(), -1)[300].item(), rel=1e-5)
    # and the conversation itself is unchanged by the scoring
    assert answer == plain_answer and conv.stream == plain.stream
    assert all(torch.equal(a[0], b[0]) for a, b in zip(conv.cache, plain.cache))
    assert all(torch.equal(a, b) for a, b in zip(conv.criteria.is_image, plain.criteria.is_image))


def test_debug_log_shows_one_block_per_turn_and_hides_ppl_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    import re

    monkeypatch.setenv("MODEL_TYPE", "llava-7B")
    monkeypatch.setenv("KV_CACHE_TYPE", "tgv_kv")
    monkeypatch.setenv("PRUNE_RATIO", "0.5")
    monkeypatch.setenv("MULTI_TURN_DECODE_EVICTION", "0")
    monkeypatch.setenv("CONVBENCH_DEBUG_PROMPTS", "1")
    torch, model, first = _tiny_llava(num_layers=32)
    from lmms_eval.models.model_utils import kv_carryover as kc

    import utils.generate_patches as patches
    from kv_caches import get_kv_cache

    monkeypatch.setattr(patches, "get_kv_cache", get_kv_cache)
    logged: List[str] = []
    monkeypatch.setattr(kc.eval_logger, "info", logged.append)
    conv = kc.CarryOverConversation(model, _IdsAdapter(first))
    conv.start(first)
    conv.score_on_fork([300, 301])  # PPL scoring: not logged
    conv.generate()
    conv.extend([200, 201, 202, 203])
    conv.score_on_fork([300])
    conv.generate()

    assert [block.splitlines()[0] for block in logged] == ["turn 1", "turn 2"]
    numbers = [[float(x) for x in re.findall(r"\d+(?:\.\d+)?", block)] for block in logged]
    # turn 1: before (image 0, text 0) (image 4 + question 15 new) | after prefill | after answer = text + 5 answer tokens
    assert numbers[0][1:5] == [0, 0, 4, 15]
    after_answer_1 = re.search(r"after answer: \(image ([\d.]+), text ([\d.]+)\)", logged[0]).groups()
    before_2 = re.search(r"before prefill: \(image ([\d.]+), text ([\d.]+)\)", logged[1]).groups()
    assert before_2 == after_answer_1  # the cache the next turn starts from is the one the answer left
    assert "new question tokens 4 + 1 last token of the previous answer" in logged[1]
    prefill_2 = float(re.search(r"after prefill:  \(image [\d.]+, text ([\d.]+)\)", logged[1]).group(1))
    assert f"= text {kc._n(prefill_2)} + 5 answer tokens" in logged[1]


# ---------------------------------------------------------------- Elastic Cache


class _OriginalElasticCache:
    """ElasticCache from github.com/liuzuyan/ElasticCache (kv_cache.py, MIT license), unchanged except that
    `.cuda()` is removed so it runs on CPU. Used to check ELASTIC_SELECTION=official reproduces it exactly."""

    def __init__(self, start_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2, ratio=0.0, distance=-25, layer_num=40):
        import torch

        self.start_size = start_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.score_sum = torch.zeros(layer_num, self.cache_size + 1)
        self.ratio = ratio
        self.protect_size = 1
        self.flag = True
        self.distance = distance
        self.layer_num = layer_num
        self.selected_idx = 0

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        import torch

        attn_score = [attention for attention in attentions]
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False)
        if attn_score.shape[-2] > 1:
            assert self.flag is True
            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[:, idx, : idx + 1]
                self.score_sum[:, : (cur_score.shape[-1])] += cur_score
        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        if forget_num > 1:
            assert self.flag is True
            self.flag = False
            selected_idx_all, merge_idx_all, throw_idx_all = [], [], []
            for idx in range(self.layer_num):
                selected_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size : (seq_len - self.protect_size)]) > forget_num)[0] + self.start_size
                throw_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size : (seq_len - self.protect_size)]) <= forget_num)[0]
                merge_idx = []
                for i in range(len(throw_idx)):
                    merge_idx.append(selected_idx[torch.abs((selected_idx - throw_idx[i])).argmin()].unsqueeze(0))
                merge_idx = torch.cat(merge_idx)
                selected_idx = torch.cat([torch.arange(self.start_size), selected_idx, torch.tensor([seq_len - self.protect_size])], dim=0)
                selected_idx_all.append(selected_idx)
                merge_idx_all.append(merge_idx)
                throw_idx_all.append(throw_idx)
            self.selected_idx = self.distance if self.distance > 0 else seq_len - forget_num + self.distance
            out = []
            for idx, (k, v) in enumerate(past_key_values):
                selected_idx, merge_idx, throw_idx = selected_idx_all[idx], merge_idx_all[idx], throw_idx_all[idx]
                k_forget = k.gather(dim=-2, index=throw_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1]))
                v_forget = v.gather(dim=-2, index=throw_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1]))
                k = k.scatter_reduce(-2, merge_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1]), k_forget, "mean")
                v = v.scatter_reduce(-2, merge_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1]), v_forget, "mean")
                k_new = k.gather(dim=-2, index=selected_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1]))
                v_new = v.gather(dim=-2, index=selected_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1]))
                out.append([k_new, v_new])
            return out
        s = self.selected_idx
        return [[torch.cat([k[:, :, :s], k[:, :, s + 1 : seq_len]], dim=2), torch.cat([v[:, :, :s], v[:, :, s + 1 : seq_len]], dim=2)] for k, v in past_key_values]


def _random_attention(torch: Any, layers: int, heads: int, q: int, k: int, sharpness: float = 1.0) -> List[Any]:
    """Causal softmax attention maps [1, H, q, k] for q new queries at the end of k keys."""
    logits = torch.randn(layers, heads, q, k) * sharpness
    rows, cols = torch.arange(q)[:, None], torch.arange(k)[None, :]
    logits = logits.masked_fill(cols > rows + (k - q), float("-inf"))
    return [a.softmax(-1).unsqueeze(0) for a in logits]


@pytest.mark.parametrize("ratio", [0.5, 0.8])
def test_elastic_official_selection_reproduces_the_original_code(ratio: float) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from kv_caches.elastic_cache import ElasticCache

    torch.manual_seed(0)
    layers, heads, n, dim = 3, 2, 80, 4  # ratio 0.8: fixed point 80 - 64 - 25 < 0, which the original counts from the end
    pkv = [[torch.randn(1, heads, n, dim), torch.randn(1, heads, n, dim)] for _ in range(layers)]
    attn = _random_attention(torch, layers, heads, n, n)
    original = _OriginalElasticCache(start_size=1, recent_size=100, ratio=ratio, layer_num=layers)
    ours = ElasticCache(layer_num=layers, start_size=1, ratio=ratio, selection="official")

    expected, got = original(pkv, n, attn), ours(pkv, n, attn)
    assert all(torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1]) for a, b in zip(expected, got))
    for step in range(1, 6):  # fixed-point elimination while decoding
        expected = [[torch.cat([k, torch.randn(1, heads, 1, dim)], 2), torch.cat([v, torch.randn(1, heads, 1, dim)], 2)] for k, v in expected]
        got = [[a[0].clone(), a[1].clone()] for a in expected]
        decode_attn = _random_attention(torch, layers, heads, 1, expected[0][0].shape[2])
        expected, got = original(expected, n + step, decode_attn), ours(got, n + step, decode_attn)
        assert all(torch.equal(a[0], b[0]) for a, b in zip(expected, got))


def test_elastic_paper_selection_keeps_the_most_important_entries_and_merges_into_the_nearest() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from kv_caches.elastic_cache import ElasticCache

    cache = ElasticCache(layer_num=1, start_size=1, ratio=0.5)
    # 8 entries: sink 0, candidates 1..6, newest 7; importance below; budget 8 * 0.5 = 4 -> 2 candidates kept
    score = torch.tensor([9.0, 0.1, 0.9, 0.05, 0.2, 0.8, 0.3, 0.0])
    keys = torch.arange(8.0).view(1, 1, 8, 1)
    out = cache([[keys, keys.clone()]], 8, {"elastic_online_prefill": True, "scores": (score,)})
    kept = out[0][0].view(-1).tolist()
    # kept: sink 0, candidates 2 and 5 (highest importance), newest 7; 1 and 3 merge into 2, 4 and 6 into 5
    assert kept == [0.0, (2 + 1 + 3) / 3, (5 + 4 + 6) / 3, 7.0]
    assert cache.fixed_point == 1  # 4 kept - 25, clamped to the first non-sink index


def test_elastic_through_patched_generate_compresses_prefill_and_eliminates_at_a_fixed_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_TYPE", "llava-7B")
    torch, model, first = _tiny_llava(num_layers=32)
    import utils.generate_patches as patches
    from kv_caches import get_kv_cache
    from kv_caches.elastic_cache import ElasticCache

    created: List[Any] = []
    monkeypatch.setattr(patches, "get_kv_cache", lambda *a, **k: created.append(get_kv_cache("elastic", prune_ratio=0.5)) or created[-1])
    out = model.generate(**first, max_new_tokens=6, do_sample=False, return_dict_in_generate=True, pad_token_id=0)
    assert isinstance(created[0], ElasticCache)
    n = first["input_ids"].shape[1]  # 19 prompt tokens; 5 generated tokens are in the cache
    lengths = {layer[0].shape[-2] for layer in out.past_key_values}
    assert lengths == {int(0.5 * (n + 5))}  # every layer at the budget, so compression and elimination ran


# ---------------------------------------------------------------- H2O and Local cache


class _OriginalH2OCache:
    """H2OCache from github.com/liuzuyan/ElasticCache (kv_cache.py, MIT license), unchanged except `.cuda()` removed."""

    def __init__(self, start_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2, ratio=0.0):
        import torch

        self.start_size = start_size
        self.k_seq_dim = k_seq_dim
        self.score_sum = torch.zeros(start_size + recent_size + 1)
        self.ratio = ratio
        self.protect_size = 1
        self.flag = True

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        import torch

        attn_score = [attention for attention in attentions]
        past_key_values_new = tuple(x for x in past_key_values)
        seq_len = past_key_values_new[0][0].size(self.k_seq_dim)
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False).mean(dim=0, keepdim=False)
        if attn_score.shape[-2] > 1:
            assert self.flag is True
            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[idx][: idx + 1]
                self.score_sum[: len(cur_score)] += cur_score
        else:
            attn_score = attn_score.squeeze(0)
            self.score_sum[:seq_len] += attn_score
        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values_new
        if forget_num > 1:
            assert self.flag is True
            self.flag = False
            selected_idx = torch.where(torch.argsort(self.score_sum[: (seq_len - self.protect_size)]) > forget_num)[0]
            selected_idx = torch.cat([selected_idx, torch.arange(seq_len - self.protect_size, seq_len)], dim=0)
            out = []
            for k, v in past_key_values_new:
                k_new = k.gather(dim=-2, index=selected_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1]))
                v_new = v.gather(dim=-2, index=selected_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1]))
                out.append([k_new, v_new])
            return out
        selected_idx = self.score_sum[self.start_size : (seq_len - self.protect_size)].argmin() + self.start_size
        self.score_sum[(selected_idx):-1] = self.score_sum[(selected_idx + 1) :].clone()
        return [[torch.cat([k[:, :, :selected_idx], k[:, :, selected_idx + 1 : seq_len]], dim=2), torch.cat([v[:, :, :selected_idx], v[:, :, selected_idx + 1 : seq_len]], dim=2)] for k, v in past_key_values_new]


def _original_local(past_key_values: Any, num_of_token: int, start_size: int, ratio: float) -> Any:
    """LocalCache.__call__ from github.com/liuzuyan/ElasticCache (kv_cache.py, MIT license)."""
    import torch

    seq_len = past_key_values[0][0].size(2)
    forget_num = int(seq_len - num_of_token * (1 - ratio))
    if forget_num <= 0:
        return past_key_values
    return [[torch.cat([k[:, :, :start_size], k[:, :, forget_num + start_size : seq_len]], dim=2), torch.cat([v[:, :, :start_size], v[:, :, forget_num + start_size : seq_len]], dim=2)] for k, v in past_key_values]


def _compare_with_original(torch: Any, original: Any, ours: Any, ratio: float) -> None:
    """Prefill on 80 entries, then 30 decode steps with peaked attention (so decode scores change the ranking);
    the caches must match after every step."""
    torch.manual_seed(0)
    layers, heads, n, dim = 3, 2, 80, 4
    pkv = [[torch.randn(1, heads, n, dim), torch.randn(1, heads, n, dim)] for _ in range(layers)]
    attn = _random_attention(torch, layers, heads, n, n, sharpness=4.0)
    expected, got = original(pkv, n, attn), ours(pkv, n, attn)
    for step in range(1, 31):
        assert [a[0].shape for a in expected] == [b[0].shape for b in got]
        assert all(torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1]) for a, b in zip(expected, got))
        expected = [[torch.cat([k, torch.randn(1, heads, 1, dim)], 2), torch.cat([v, torch.randn(1, heads, 1, dim)], 2)] for k, v in expected]
        got = [[a[0].clone(), a[1].clone()] for a in expected]
        decode_attn = _random_attention(torch, layers, heads, 1, expected[0][0].shape[2], sharpness=4.0)
        expected, got = original(expected, n + step, decode_attn), ours(got, n + step, decode_attn)


@pytest.mark.parametrize("ratio", [0.5, 0.8])
def test_h2o_official_selection_reproduces_the_original_code(ratio: float) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from kv_caches.h2o_cache import H2OCache

    original = _OriginalH2OCache(start_size=1, recent_size=2047, ratio=ratio)
    ours = H2OCache(layer_num=3, start_size=1, recent_size=2047, ratio=ratio, selection="official")
    _compare_with_original(torch, original, ours, ratio)


@pytest.mark.parametrize("ratio", [0.5, 0.8])
def test_local_cache_reproduces_the_original_code(ratio: float) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from kv_caches.local_cache import LocalCache

    ours = LocalCache(layer_num=3, start_size=1, ratio=ratio)
    _compare_with_original(torch, lambda pkv, n, attn: _original_local(pkv, n, 1, ratio), ours, ratio)


def test_h2o_paper_selection_keeps_the_top_scores_and_keeps_them_aligned() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from kv_caches.h2o_cache import H2OCache

    cache = H2OCache(layer_num=1, start_size=1, ratio=0.5)
    score = torch.tensor([9.0, 0.1, 0.9, 0.05, 0.2, 0.8, 0.3, 0.0])
    keys = torch.arange(8.0).view(1, 1, 8, 1)
    out = cache([[keys, keys.clone()]], 8, {"h2o_online_prefill": True, "scores": score})
    assert out[0][0].view(-1).tolist() == [0.0, 2.0, 5.0, 7.0]  # sink, the 2 highest-scoring candidates, newest
    assert cache.scores.tolist() == pytest.approx([9.0, 0.9, 0.8, 0.0])  # each score stays with its entry

    # one decode step: entry 2 (now at index 1) gets no attention, entry 5 a lot -> the lowest score (index 1) goes
    k = torch.cat([out[0][0], torch.full((1, 1, 1, 1), 8.0)], 2)
    attention = torch.tensor([0.0, 0.0, 0.6, 0.2, 0.2]).view(1, 1, 1, 5)
    out = cache([[k, k.clone()]], 9, (attention,))  # 5 entries, budget int(5 - 9 * 0.5) = 0 to forget -> no eviction yet
    assert out[0][0].view(-1).tolist() == [0.0, 2.0, 5.0, 7.0, 8.0]
    k = torch.cat([out[0][0], torch.full((1, 1, 1, 1), 9.0)], 2)
    attention = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.25, 0.25]).view(1, 1, 1, 6)
    out = cache([[k, k.clone()]], 10, (attention,))  # 6 entries, budget 5: the lowest accumulated score among 2, 5, 7, 8 is 8's
    assert out[0][0].view(-1).tolist() == [0.0, 2.0, 5.0, 7.0, 9.0]


@pytest.mark.parametrize("method", ["h2o", "local"])
def test_h2o_and_local_through_patched_generate_hold_the_budget(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    monkeypatch.setenv("MODEL_TYPE", "llava-7B")
    torch, model, first = _tiny_llava(num_layers=32)
    import utils.generate_patches as patches
    from kv_caches import get_kv_cache

    created: List[Any] = []
    monkeypatch.setattr(patches, "get_kv_cache", lambda *a, **k: created.append(get_kv_cache(method, prune_ratio=0.5)) or created[-1])
    out = model.generate(**first, max_new_tokens=6, do_sample=False, return_dict_in_generate=True, pad_token_id=0)
    n = first["input_ids"].shape[1]
    assert {layer[0].shape[-2] for layer in out.past_key_values} == {int(0.5 * (n + 5))}


def test_h2o_official_is_refused_for_multi_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("transformers")
    monkeypatch.setenv("MODEL_TYPE", "llava-7B")
    monkeypatch.setenv("H2O_SELECTION", "official")
    from kv_caches import get_kv_cache

    with pytest.raises(ValueError, match="multi-turn runs need H2O_SELECTION=paper"):
        get_kv_cache("h2o", prune_ratio=0.5, multi_turn=True)


# ---------------------------------------------------------------- multi-turn budget, all methods


def _carried_conversation(monkeypatch: pytest.MonkeyPatch, method: str, decode_eviction: str = "0", budget: str = "fixed", answer_tokens: int = 6) -> Any:
    monkeypatch.setenv("MODEL_TYPE", "llava-7B")
    monkeypatch.setenv("KV_CACHE_TYPE", method)
    monkeypatch.setenv("PRUNE_RATIO", "0.5")
    monkeypatch.setenv("MULTI_TURN_DECODE_EVICTION", decode_eviction)
    monkeypatch.setenv("MULTI_TURN_BUDGET", budget)
    torch, model, first = _tiny_llava(num_layers=32)
    from lmms_eval.models.model_utils.kv_carryover import CarryOverConversation

    import utils.generate_patches as patches
    from kv_caches import get_kv_cache

    monkeypatch.setattr(patches, "get_kv_cache", get_kv_cache)
    adapter = _IdsAdapter(first)
    adapter.generate_kwargs["max_new_tokens"] = answer_tokens
    conv = CarryOverConversation(model, adapter)
    conv.start(first)
    return model, conv, first["input_ids"].shape[1]


def _kept_after_prefill(conv: Any) -> float:
    return sum(conv.criteria.after_prefill)


METHODS = ["tgv_kv", "elastic", "h2o", "local"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("budget", ["fixed", "conversation"])
def test_multi_turn_prefills_compress_to_the_budget(monkeypatch: pytest.MonkeyPatch, method: str, budget: str) -> None:
    """fixed: every prefill keeps the turn-1 budget; conversation: (1 - ratio) x the conversation so far."""
    model, conv, n1 = _carried_conversation(monkeypatch, method, budget=budget)
    seen_inputs: List[List[int]] = []
    hook = model.register_forward_pre_hook(lambda mod, args, kwargs: seen_inputs.append(kwargs["input_ids"][0].tolist()), with_kwargs=True)
    conv.generate()
    for ids in ([200, 201, 202, 203, 204, 205], [210, 211, 212, 213]):
        expected_new = conv.pending + ids
        conv.extend(ids)
        seen_inputs.clear()
        conv.generate()
        assert seen_inputs[0] == expected_new  # the prefill processes only the new tokens
        target = 0.5 * n1 if budget == "fixed" else 0.5 * (len(conv.stream) - 5)
        assert conv.criteria.last_target == pytest.approx(target)
        if method == "tgv_kv":  # per-layer budgets (TVB), each rounded: the mean is within half an entry
            assert abs(_kept_after_prefill(conv) - target) <= 0.5
        else:
            assert _kept_after_prefill(conv) == math.ceil(target)
        lengths = [p[0].shape[-2] for p in conv.cache]
        assert lengths == [f.numel() for f in conv.criteria.is_image]  # flags in sync with every layer
        assert sum(lengths) / len(lengths) == pytest.approx(_kept_after_prefill(conv) + 5)  # all 5 answer tokens kept
    hook.remove()


@pytest.mark.parametrize("method", METHODS)
def test_fixed_budget_does_not_depend_on_the_previous_answer_length(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    kept = []
    for answer_tokens in (3, 9):
        model, conv, n1 = _carried_conversation(monkeypatch, method, answer_tokens=answer_tokens)
        conv.generate()
        conv.extend([200, 201, 202, 203, 204, 205])
        conv.generate()
        kept.append(_kept_after_prefill(conv))
    assert kept[0] == pytest.approx(kept[1], abs=0.5 if method == "tgv_kv" else 0)


@pytest.mark.parametrize("method", METHODS)
def test_multi_turn_decode_eviction_holds_the_fixed_budget(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    model, conv, n1 = _carried_conversation(monkeypatch, method, decode_eviction="1")
    conv.generate()
    conv.extend([200, 201, 202, 203, 204, 205])
    conv.generate()
    lengths = [p[0].shape[-2] for p in conv.cache]
    assert [f.numel() for f in conv.criteria.is_image] == lengths
    if method == "tgv_kv":
        assert abs(sum(lengths) / len(lengths) - 0.5 * n1) <= 1.5  # per-layer budgets, each held while decoding
    else:
        assert set(lengths) == {math.ceil(0.5 * n1)}


def test_tgv_kv_budget_a_full_layer_cannot_use_goes_to_the_other_layers() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("transformers")
    from kv_caches.tgv_kv_multiturn import _allocate

    # 4 layers share 40 entries 70/10/10/10 (28/4/4/4), but the first holds only 20: its other 8 go to the rest
    keep = _allocate(40, [0.7, 0.1, 0.1, 0.1], [20, 50, 50, 50])
    assert keep.tolist() == pytest.approx([20, 4 + 8 / 3, 4 + 8 / 3, 4 + 8 / 3])
    assert keep.sum() == pytest.approx(40)
    assert _allocate(500, [0.25] * 4, [20, 50, 50, 50]).tolist() == [20, 50, 50, 50]  # more than they all hold


def test_tgv_kv_later_prefill_reaches_the_budget_when_a_few_layers_dominate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: with text-to-image attention concentrated in a few layers, those layers were offered more entries
    than they hold and the excess was lost (44 kept per layer instead of 171)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("MODEL_TYPE", "llava-7B")
    from kv_caches import get_kv_cache

    cache = get_kv_cache("tgv_kv", prune_ratio=0.8, multi_turn=True)
    layers, cached, new = cache.layer_num, 236, 25
    cache.budget = 171.2  # as if turn 1 kept 171.2 per layer
    cache.is_image = [torch.cat([torch.ones(36, dtype=torch.bool), torch.zeros(cached - 36, dtype=torch.bool)]) for _ in range(layers)]
    pkv = transformers.cache_utils.DynamicCache([[torch.randn(1, 2, cached + new, 4), torch.randn(1, 2, cached + new, 4)] for _ in range(layers)])
    stats = {
        "new_is_image": torch.zeros(new, dtype=torch.bool),
        "score_sums": [torch.rand(1, cached + new) for _ in range(layers)],
        "text_image_attn_sums": list(torch.tensor([10.0] * 4 + [0.1] * (layers - 4))),
    }
    cache.initial_text_len_list = []
    out = cache._prefill_history(pkv, 856, stats)
    kept = [p[0].shape[-2] for p in out]
    assert kept[:4] == [cached + new] * 4  # the dominant layers keep everything they hold
    assert abs(sum(kept) / layers - 171.2) <= 0.5  # and the rest of the budget goes to the other layers
