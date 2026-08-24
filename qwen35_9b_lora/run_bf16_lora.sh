#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-qwen35-9b-bf16-lora-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/${RUN_ID}}"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"

export PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
export SWIFT_BIN="${SWIFT_BIN:-${SCRIPT_DIR}/.venv/bin/swift}"
export MODEL_ID="${MODEL_ID:-${WORKSPACE_DIR}/.cache/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master}"
export DATA_DIR="${ARTIFACT_DIR}/data"
export TRAIN_DATA="${DATA_DIR}/train.jsonl"
export EVAL_DATA="${DATA_DIR}/eval.jsonl"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/checkpoints}"
export LOGGING_DIR="${LOGGING_DIR:-${RUN_ROOT}/tensorboard}"
export RUN_NAME="${RUN_NAME:-${RUN_ID}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export EPOCHS="${EPOCHS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export EVAL_STEPS="${EVAL_STEPS:-10}"
export SAVE_STEPS="${SAVE_STEPS:-10}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

echo "[preflight] Archiving and validating the exact training dataset in: ${ARTIFACT_DIR}"
mkdir -p "${DATA_DIR}"
cp "${SCRIPT_DIR}/data/train.jsonl" "${DATA_DIR}/"
cp "${SCRIPT_DIR}/data/eval.jsonl" "${DATA_DIR}/"
cp "${SCRIPT_DIR}/data/index.jsonl" "${DATA_DIR}/"
cp "${SCRIPT_DIR}/data/manifest.json" "${DATA_DIR}/"

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_data.py" --data-dir "${DATA_DIR}" \
  | tee "${ARTIFACT_DIR}/data-validation.log"


git -C "${PROJECT_DIR}" rev-parse HEAD > "${ARTIFACT_DIR}/git-commit.txt"
git -C "${PROJECT_DIR}" status --short > "${ARTIFACT_DIR}/git-status.txt"
"${PYTHON_BIN}" -m pip freeze > "${ARTIFACT_DIR}/pip-freeze.txt"
nvidia-smi -q > "${ARTIFACT_DIR}/nvidia-smi.txt"
df -h "${RUN_ROOT}" > "${ARTIFACT_DIR}/disk-usage.txt"

declare -px \
  PYTHON_BIN SWIFT_BIN MODEL_ID DATA_DIR TRAIN_DATA EVAL_DATA \
  OUTPUT_DIR LOGGING_DIR RUN_NAME CUDA_VISIBLE_DEVICES NPROC_PER_NODE \
  DEEPSPEED_CONFIG MAX_LENGTH EPOCHS LEARNING_RATE GRAD_ACCUM \
  LORA_RANK LORA_ALPHA ATTN_IMPL EVAL_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT \
  > "${ARTIFACT_DIR}/train-env.sh"

record_exit_status() {
  local status="$?"
  printf '%s\n' "${status}" > "${ARTIFACT_DIR}/exit-status.txt"
}
trap record_exit_status EXIT

echo "[train] Starting BF16 LoRA; checkpoints: ${OUTPUT_DIR}"
set -o pipefail
bash "${SCRIPT_DIR}/train_lora.sh" "$@" 2>&1 | tee "${RUN_ROOT}/train.log"
