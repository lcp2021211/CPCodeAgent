#!/usr/bin/env bash

# Shared, side-effect-free configuration for the Qwen3.5-9B A/B evaluation.
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

EXP_CONFIG_FILE="${EXP_CONFIG_FILE:-${EXP_DIR}/config.env}"
if [[ -f "${EXP_CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${EXP_CONFIG_FILE}"
fi

: "${EXP_TRAIN_VENV:=${PROJECT_ROOT}/qwen35_9b_lora/.venv}"
: "${EXP_EVAL_VENV:=${PROJECT_ROOT}/trajectory_experiments/.venv}"
EXP_TRAIN_PYTHON="${EXP_TRAIN_VENV}/bin/python"
EXP_EVAL_PYTHON="${EXP_EVAL_VENV}/bin/python"

# The interactive agent and the evaluator share the same model endpoint by
# default. Parse dotenv with python-dotenv instead of sourcing it as shell code.
EXP_ROOT_ENV="${PROJECT_ROOT}/.env"
dotenv_get() {
  "${EXP_EVAL_PYTHON}" -c \
    'from dotenv import dotenv_values; import sys; print(dotenv_values(sys.argv[1]).get(sys.argv[2]) or "")' \
    "$1" "$2"
}
if [[ -x "${EXP_EVAL_PYTHON}" && -f "${EXP_ROOT_ENV}" ]]; then
  EXP_DOTENV_BASE_URL="$(dotenv_get "${EXP_ROOT_ENV}" OPENAI_BASE_URL)"
  EXP_DOTENV_API_KEY="$(dotenv_get "${EXP_ROOT_ENV}" OPENAI_API_KEY)"
  EXP_DOTENV_MODEL="$(dotenv_get "${EXP_ROOT_ENV}" CPCODEAGENT_MODEL)"
fi

: "${EXP_BASE_MODEL:=${WORKSPACE_ROOT}/.cache/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master}"
: "${EXP_LORA_ADAPTER:=${PROJECT_ROOT}/qwen35_9b_lora/runs/qwen35-9b-bf16-lora-20260822T161647Z/checkpoints/best}"
: "${EXP_TRAINING_INDEX:=${PROJECT_ROOT}/qwen35_9b_lora/data/index.jsonl}"
: "${EXP_EVAL_DATASET:=${PROJECT_ROOT}/trajectory_experiments/data/swesmith_train}"

: "${EXP_IN_DOMAIN_REPO:=swesmith/scanny__python-pptx.278b47b1}"
: "${EXP_IN_DOMAIN_COUNT:=25}"
: "${EXP_OUT_DOMAIN_COUNT:=25}"
: "${EXP_EVAL_SEED:=42}"
: "${EXP_EVAL_MANIFEST:=${EXP_DIR}/eval_set.json}"
: "${EXP_TEST_IDS:=${EXP_DIR}/test_ids.txt}"

: "${EXP_HOST:=127.0.0.1}"
: "${EXP_PORT:=8000}"
: "${EXP_API_KEY:=${EXP_DOTENV_API_KEY:-local-eval}}"
: "${EXP_BASE_URL:=${EXP_DOTENV_BASE_URL:-http://${EXP_HOST}:${EXP_PORT}/v1}}"
: "${EXP_SERVER_URL:=${EXP_BASE_URL%/v1}}"
: "${EXP_CUDA_VISIBLE_DEVICES:=0}"
: "${EXP_SERVER_MAX_LENGTH:=16384}"
: "${EXP_SERVER_MAX_NEW_TOKENS:=2048}"
: "${EXP_SERVER_START_TIMEOUT:=300}"

: "${EXP_BASE_SERVED_MODEL:=${EXP_DOTENV_MODEL:-qwen35-9b-base}}"
: "${EXP_LORA_SERVED_MODEL:=qwen35-9b-lora-best}"
: "${EXP_RUN_TAG:=qwen35-9b-ab-n$((EXP_IN_DOMAIN_COUNT + EXP_OUT_DOMAIN_COUNT))-seed${EXP_EVAL_SEED}}"
: "${EXP_RUNS_DIR:=${EXP_DIR}/runs}"
: "${EXP_LOGS_DIR:=${EXP_DIR}/logs}"
: "${EXP_REPORTS_DIR:=${EXP_DIR}/reports}"

: "${EXP_TEMPERATURE:=0}"
: "${EXP_MAX_STEPS:=40}"
: "${EXP_MAX_SECONDS:=1800}"
: "${EXP_MAX_TOKENS:=100000}"
: "${EXP_CONTEXT_WINDOW_TOKENS:=57344}"
: "${EXP_WORKERS:=1}"

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

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}
