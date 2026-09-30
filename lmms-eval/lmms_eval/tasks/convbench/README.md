# ConvBench

Multi-turn visual conversation benchmark (Liu et al., NeurIPS 2024 D&B, [arXiv:2403.20194](https://arxiv.org/abs/2403.20194)).
577 conversations, each three turns over one image: perception, then reasoning, then creation.
Every turn has a human-verified reference answer, and turn 3 has annotated focus points.

## Setup

The dataset is the spreadsheet and images in the official GitHub repo; nothing is downloaded from Hugging Face at run time.

```bash
git clone --depth 1 https://github.com/shirlyliu64/ConvBench /data/ConvBench
python lmms-eval/lmms_eval/tasks/convbench/prepare_convbench.py --convbench_dir /data/ConvBench
```

This writes `convbench.jsonl` here and points `_convbench_data_yaml` at it. One spreadsheet row (ID 131) has no image in the release and is skipped, leaving the 577 conversations reported in the paper.

BLEU/METEOR/ROUGE need Java on `PATH`, e.g. `conda install -c conda-forge openjdk` or `apt install default-jre`.

## Tasks

| task | chat history given to the model | metrics |
|---|---|---|
| `convbench` | the model's own previous answers | judge R1, R2, S0-S3; BLEU-1..4, METEOR, ROUGE-L (turns 1-3); PPL overall and per turn |
| `convbench_ref1` | reference answer for turn 1 | judge S2, S3, S0 (= Ŝ2, Ŝ3, Ŝ0); text metrics on turns 2-3 |
| `convbench_ref2` | reference answers for turns 1 and 2 | judge S3, S0 (= S̃3, S̃0); text metrics on turn 3 |

The ablation tasks still generate the turns they replace; those outputs are not scored.

## Metrics

**LLM judge** (the paper's ConvEval). A text-only judge sees the instruction-conditioned caption, the three instructions, the model's conversation and the reference. It judges each turn (S1-S3), then the whole conversation (S0) with those turn judgements as input. R2 = (S1+S2+S3)/3, R1 = (R2+S0)/2.
- `CONVBENCH_GRADING=pairwise` (default): the metric is the model's **win rate (%) against the human reference**, the paper's headline number (GPT-4V: 39.51). The A/B order is seeded per conversation, so it is identical across runs.
- `CONVBENCH_GRADING=direct`: mean 1-10 rating with the reference counted as 10.
- `CONVBENCH_GRADING=none`: no judge, no API key needed; only the metrics below are reported.

**BLEU-1..4, METEOR, ROUGE-L** (×100). Each generated turn is compared with its reference using `pycocoevalcap`: PTB tokenizer, corpus BLEU, METEOR 1.5 and ROUGE-L. This is the same toolkit and protocol as lmms-eval's `coco_cap`. The table shows values pooled over all scored turns. Turn off with `CONVBENCH_TEXT_METRICS=0`.

**PPL** (lower is better). How likely the model finds the human reference answers. `PPL` is exp(total NLL / total tokens) over all three turns; `PPL_turn1`, `PPL_turn2` and `PPL_turn3` pool only that turn's references over all conversations. `PPL_HISTORY` sets what each reference R_k is conditioned on:
- `model` (default): the model's own conversation, i.e. image, Q1, A1, ..., Q_k with the model's answers A. R_k is scored at the moment A_k is about to be generated, on a copy of the (compressed) cache, so it sees exactly the context A_k is generated from; the copy is then discarded and A_k is generated as usual.
- `reference`: the reference conversation, image, Q1, R1, ..., Q_k, which does not depend on what the model generated.

The reference tokens are forced one at a time through `model.generate`, so KV-cache compression applies exactly as when the model answers. Without compression, this equals a normal teacher-forced forward pass (tested).
- References are scored up to `max_new_tokens` (1024); only 3 of 1,731 are longer, and they are flagged as truncated.
- Scoring adds roughly one decode step per reference token, plus one extra prefill of each turn's new tokens and a copy of the KV cache per turn (`model` mode).
- Turn off with `CONVBENCH_PPL=0`.
- Reported by `convbench` only.

**Per-turn breakdown.** Per-turn BLEU/METEOR/ROUGE and PPL are written to `<output_path>/submissions/<task>_by_turn_<KV_CACHE_TYPE>_ratio<PRUNE_RATIO>_<timestamp>.json`. The per-sample log (`--log_samples`) also keeps every judgement text, prediction and reference.

**Every conversation, turn by turn.** With PPL on, each conversation's turns are saved in three places:
- `<output_path>/submissions/<task>_conversations_<KV_CACHE_TYPE>_ratio<PRUNE_RATIO>_<timestamp>.jsonl`: one line per conversation, `{"id", "ppl", "turns": [{"turn", "question", "model_output", "reference", "nll", "tokens", "ppl", "truncated"}, ...]}`.
- The `--log_samples` file: the same `turns` list and the conversation's `ppl`, in each sample's `convbench_PPL` field.
- The console log, with `CONVBENCH_LOG_TURNS=1`: question, model output, reference and PPL for every turn, as the results are processed.

## Judge configuration

- `CONVBENCH_JUDGE_MODEL` sets the judge model (default `gpt-4o-mini`). The paper used ChatGPT-3.5, so absolute judge numbers will not match the paper. Keep the judge fixed when comparing runs.
- `OPENAI_API_KEY` and `OPENAI_API_URL` must be set, e.g. `https://api.openai.com/v1`, or a local vLLM server's `http://localhost:8000/v1`. `OPENAI_API_URL` has a broken default in this lmms-eval version.

The task checks the judge settings and Java when it loads and stops *before* generation if something is missing. lmms-eval still exits with code 0 in that case, so read the log.

## Models

The task uses `output_type: generate_until_multi_round`. `llava_hf` and `qwen3_vl` implement it with a real chat history, with the image attached once in the first user turn, and they implement the reference scoring used for PPL. Other models need their own `generate_until_multi_round`.

### KV cache across turns

By default the KV cache is **carried from one turn to the next** (`MULTI_TURN_KV_CARRYOVER=1`):

1. **Turn 1** prefills the full image and question 1. TGV-KV compresses the cache at that prefill exactly as in single-turn tasks (tested to be identical), then answer 1 is generated.
2. **Turn 2** prefills only the new tokens (the end of answer 1, the chat-template glue and question 2) on top of the carried cache. That cache holds the compressed image and question 1 plus answer 1. TGV-KV then compresses the whole cache again and answer 2 is generated.
3. **Turn 3** does the same.

How TGV-KV compresses a carried cache (`kv_caches/tgv_kv_multiturn.py`):

- It tracks, for every layer, which cached entries are image tokens, because each layer keeps a different subset.
- Image entries are ranked by text-weighted attention from the new text tokens (TWR), and text entries are kept first (TPR).
- Layer budgets follow the new tokens' text-to-image attention (TVB).
- The budget is **(1 − `PRUNE_RATIO`) × all tokens of the conversation so far**. This is the rule TGV-KV's decode step already uses, so memory stays at the same fraction of a full cache in every turn.

`MULTI_TURN_DECODE_EVICTION` controls eviction while an answer is being generated:

- `0` (default): no eviction during generation. Answers are kept in full until the next turn's prefill compresses them along with everything else.
- `1`: TGV-KV's usual eviction of one entry per decode step, as in its single-turn code, so answers are compressed while they are generated.

The ablation tasks (`convbench_ref1` and `convbench_ref2`) teacher-force the reference answers as the earlier turns, so the cache holds exactly what the history says. For PPL, each reference is forced on a copy of the conversation's cache, right before the model's answer to that turn (`PPL_HISTORY=model`), or after the compressed reference history (`PPL_HISTORY=reference`).

`MULTI_TURN_KV_CARRYOVER=0` restores the previous behaviour: every turn re-encodes the full history, and compression runs independently per turn.

Without compression (no `KV_CACHE_TYPE`), carrying the cache is exactly equivalent to one forward pass over the whole conversation; the tests check this for LLaVA and Qwen3-VL.

Carrying the cache needs `attn_implementation=eager`, and a chat template that renders earlier turns the same way in later prompts. The llava-hf and Qwen templates do; otherwise the run stops with an error that points to `MULTI_TURN_KV_CARRYOVER=0`.

`CONVBENCH_DEBUG_PROMPTS=1` logs, per turn: the cached image and text entries, the tokens processed at prefill (and their text), the cache size right after the prefill compression, and the number of generated tokens.

With TGV-KV, set `MAX_GENERATED_TOKENS` to 1024 or more. `utils/generate_patches.py` stops decoding at that value regardless of `max_new_tokens`, which would cut both the answers and the PPL scoring. For a full-KV baseline, unset `KV_CACHE_TYPE` but keep `MODEL_TYPE` set.

## Tests

```bash
pytest lmms-eval/test/test_convbench.py   # from the TGV-KV root
```
