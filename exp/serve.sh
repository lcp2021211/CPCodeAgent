#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "base" && "$1" != "lora" ) ]]; then
  echo "Usage: $0 {base|lora}" >&2
  exit 2
fi
ARM="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_file "${EXP_TRAIN_PYTHON}"
require_dir "${EXP_BASE_MODEL}"

ARGS=(
  --model "${EXP_BASE_MODEL}"
  --use_hf true
  --infer_backend transformers
  --torch_dtype bfloat16
  --attn_impl sdpa
  --host "${EXP_HOST}"
  --port "${EXP_PORT}"
  --api_key "${EXP_API_KEY}"
  --max_length "${EXP_SERVER_MAX_LENGTH}"
  --truncation_strategy delete
  --max_new_tokens "${EXP_SERVER_MAX_NEW_TOKENS}"
  --temperature "${EXP_TEMPERATURE}"
  --max_batch_size 1
  --enable_thinking false
  --add_non_thinking_prefix true
  --no_verbose
)

if [[ "${ARM}" == "base" ]]; then
  ARGS+=(--served_model_name "${EXP_BASE_SERVED_MODEL}")
else
  require_dir "${EXP_LORA_ADAPTER}"
  ARGS+=(
    --adapters "${EXP_LORA_ADAPTER}"
    --served_model_name "${EXP_LORA_SERVED_MODEL}"
  )
fi

export CUDA_VISIBLE_DEVICES="${EXP_CUDA_VISIBLE_DEVICES}"
exec "${EXP_TRAIN_PYTHON}" -m swift.cli.deploy "${ARGS[@]}"
