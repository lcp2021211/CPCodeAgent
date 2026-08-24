#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-$(dirname "${PYTHON_BIN}")/swift}"
LOCAL_MODEL="${SCRIPT_DIR}/../../.cache/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"
MODEL_ID="${MODEL_ID:-${LOCAL_MODEL}}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-${DATA_DIR}/eval.jsonl}"
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
EVAL_STEPS="${EVAL_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_DIR}/tensorboard}"
RUN_NAME="${RUN_NAME:-qwen35-9b-bf16-lora}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is missing: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${SWIFT_BIN}" ]]; then
  echo "ms-swift executable is missing: ${SWIFT_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_ID}/config.json" ]]; then
  echo "Local model config is missing: ${MODEL_ID}/config.json" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_data.py" --data-dir "${DATA_DIR}"

EXTRA_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  EXTRA_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

export NPROC_PER_NODE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${SWIFT_BIN}" sft \
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
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_only_model false \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --logging_steps 1 \
  --report_to tensorboard \
  --logging_dir "${LOGGING_DIR}" \
  --run_name "${RUN_NAME}" \
  --dataset_num_proc 4 \
  --dataloader_num_workers 2 \
  --load_from_cache_file false \
  --seed 42 \
  --data_seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --add_version false \
  --create_checkpoint_symlink true \
  "${EXTRA_ARGS[@]}" \
  "$@"
