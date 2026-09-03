#!/usr/bin/env bash

# Shared, side-effect-free configuration for the Qwen3.5-9B A/B evaluation.
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

CONFIG_FILE="${CONFIG_FILE:-${EXP_DIR}/config.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

: "${TRAIN_VENV:=${PROJECT_ROOT}/qwen35_9b_lora/.venv}"
: "${EVAL_VENV:=${PROJECT_ROOT}/trajectory_experiments/.venv}"
: "${BASE_MODEL:=${WORKSPACE_ROOT}/.cache/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master}"
: "${LORA_ADAPTER:=${PROJECT_ROOT}/qwen35_9b_lora/runs/qwen35-9b-bf16-lora-20260822T161647Z/checkpoints/best}"
: "${TRAINING_INDEX:=${PROJECT_ROOT}/qwen35_9b_lora/data/index.jsonl}"
: "${EVAL_DATASET:=${PROJECT_ROOT}/trajectory_experiments/data/swesmith_train}"

: "${IN_DOMAIN_REPO:=swesmith/scanny__python-pptx.278b47b1}"
: "${IN_DOMAIN_COUNT:=25}"
: "${OUT_DOMAIN_COUNT:=25}"
: "${EVAL_SEED:=20260903}"
: "${EVAL_MANIFEST:=${EXP_DIR}/eval_set.json}"
: "${TEST_IDS:=${EXP_DIR}/test_ids.txt}"

: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${API_KEY:=local-eval}"
: "${BASE_URL:=http://${HOST}:${PORT}/v1}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${SERVER_MAX_LENGTH:=16384}"
: "${SERVER_MAX_NEW_TOKENS:=2048}"
: "${SERVER_START_TIMEOUT:=300}"

: "${BASE_SERVED_MODEL:=qwen35-9b-base}"
: "${LORA_SERVED_MODEL:=qwen35-9b-lora-best}"
: "${RUN_TAG:=qwen35-9b-ab-n$((IN_DOMAIN_COUNT + OUT_DOMAIN_COUNT))-seed${EVAL_SEED}}"
: "${RUNS_DIR:=${EXP_DIR}/runs}"
: "${LOGS_DIR:=${EXP_DIR}/logs}"
: "${REPORTS_DIR:=${EXP_DIR}/reports}"

: "${TEMPERATURE:=0}"
: "${MAX_STEPS:=40}"
: "${MAX_SECONDS:=1800}"
: "${MAX_TOKENS:=100000}"
: "${WORKERS:=1}"

TRAIN_PYTHON="${TRAIN_VENV}/bin/python"
EVAL_PYTHON="${EVAL_VENV}/bin/python"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

