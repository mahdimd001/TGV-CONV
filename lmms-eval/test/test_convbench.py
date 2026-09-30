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
