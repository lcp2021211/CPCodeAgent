#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-${SCRIPT_DIR}/data/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-${SCRIPT_DIR}/data/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/qwen35-9b-agent-lora}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_data.py" --data-dir "${SCRIPT_DIR}/data"

EXTRA_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  EXTRA_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

export NPROC_PER_NODE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

swift sft \
  --model "${MODEL_ID}" \
  --use_hf true \
  --tuner_type lora \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${EVAL_DATA}" \
  --strict true \
  --torch_dtype bfloat16 \
  --freeze_llm false \
  --freeze_vit true \
  --freeze_aligner true \
  --target_modules all-linear \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout 0.05 \
  --loss_scale last_round+ignore_empty_think \
  --add_non_thinking_prefix true \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --max_length "${MAX_LENGTH}" \
  --truncation_strategy delete \
  --packing false \
  --padding_free false \
  --group_by_length true \
  --attn_impl "${ATTN_IMPL}" \
  --eval_strategy steps \
  --eval_steps 10 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 3 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --logging_steps 1 \
  --report_to tensorboard \
  --dataset_num_proc 4 \
  --dataloader_num_workers 2 \
  --load_from_cache_file false \
  --seed 42 \
  --data_seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
