# ConvBench (multi-turn) with TGV-KV. Run from the TGV-KV root.
# One-time data setup:
#   git clone --depth 1 https://github.com/shirlyliu64/ConvBench /root/share/ConvBench
#   python lmms-eval/lmms_eval/tasks/convbench/prepare_convbench.py --convbench_dir /root/share/ConvBench
# BLEU/METEOR/ROUGE need Java:  conda install -c conda-forge openjdk   (or: apt install default-jre)
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_TYPE=llava-7B         # keep set even for the full-KV baseline
export HF_HUB_OFFLINE=1
export MAX_GENERATED_TOKENS=1024   # ConvBench answers are long; 32/64 would truncate every turn and the PPL scoring
ckpt=/root/share/llava-1.5-7b-hf

# Metrics
export CONVBENCH_GRADING=pairwise  # pairwise | direct | none (no judge, no API key)
export CONVBENCH_TEXT_METRICS=1    # BLEU-1..4, METEOR, ROUGE-L
export CONVBENCH_PPL=1             # perplexity of the reference answers under the compressed KV cache
export CONVBENCH_LOG_TURNS=1       # log question, model output, reference and PPL of every turn
export PPL_HISTORY=model           # R_k scored after Q1, A1, ..., Q_k (model's own answers) | reference: after Q1, R1, ..., Q_k

# Multi-turn KV cache: carried across turns and re-compressed at every turn's prefill
export MULTI_TURN_KV_CARRYOVER=1   # 0 = re-encode the whole history every turn (previous behaviour)
export MULTI_TURN_DECODE_EVICTION=0  # 1 = also evict one entry per decode step (TGV-KV's single-turn behaviour)

# LLM judge (only needed when CONVBENCH_GRADING is not none)
export CONVBENCH_JUDGE_MODEL=gpt-4o-mini
export OPENAI_API_URL=https://api.openai.com/v1
# export OPENAI_API_KEY=...

run() {
  accelerate launch --num_processes=2 --main_process_port 29508 -m lmms_eval --model llava_hf \
      --model_args "pretrained=$ckpt,attn_implementation=eager" \
      --tasks "convbench" --batch_size 1 --log_samples \
      --log_samples_suffix convbench --output_path ./logs/
}

# Full-KV baseline
unset KV_CACHE_TYPE
run

# TGV-KV at several retention levels (PRUNE_RATIO=0.9 keeps 10% of the KV cache)
export KV_CACHE_TYPE=tgv_kv
for ratio in 0.5 0.8 0.9 0.95
do
  export PRUNE_RATIO=$ratio
  echo ${KV_CACHE_TYPE} ${PRUNE_RATIO}
  run
done
# Error attribution (paper's hierarchical ablation): use --tasks "convbench,convbench_ref1,convbench_ref2".
