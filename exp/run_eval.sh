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

require_file "${EVAL_PYTHON}"
require_dir "${EVAL_DATASET}"
require_file "${EVAL_MANIFEST}"
require_file "${TEST_IDS}"
command -v curl >/dev/null 2>&1 || die "curl is required"

if [[ "${ARM}" == "base" ]]; then
  MODEL_NAME="${BASE_SERVED_MODEL}"
else
  MODEL_NAME="${LORA_SERVED_MODEL}"
fi
RUN_ID="${RUN_TAG}-${ARM}"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"

curl -fsS "http://${HOST}:${PORT}/health" >/dev/null \
  || die "model server is not healthy at http://${HOST}:${PORT}"
MODELS_JSON="$(curl -fsS "${BASE_URL}/models" -H "Authorization: Bearer ${API_KEY}")"
grep -Fq "${MODEL_NAME}" <<<"${MODELS_JSON}" \
  || die "server does not expose expected model ${MODEL_NAME}: ${MODELS_JSON}"

INSTANCE_ARGS=()
while IFS= read -r INSTANCE_ID; do
  [[ -z "${INSTANCE_ID}" || "${INSTANCE_ID}" == \#* ]] && continue
  INSTANCE_ARGS+=(--instance-id "${INSTANCE_ID}")
done < "${TEST_IDS}"
[[ ${#INSTANCE_ARGS[@]} -gt 0 ]] || die "no test IDs found in ${TEST_IDS}"

mkdir -p "${RUN_DIR}"
{
  echo "arm=${ARM}"
  echo "model=${MODEL_NAME}"
  echo "base_url=${BASE_URL}"
  echo "temperature=${TEMPERATURE}"
  echo "max_steps=${MAX_STEPS}"
  echo "max_seconds=${MAX_SECONDS}"
  echo "max_tokens=${MAX_TOKENS}"
  echo "workers=${WORKERS}"
  echo "eval_manifest=${EVAL_MANIFEST}"
  echo "test_ids_sha256=$(sha256sum "${TEST_IDS}" | cut -d ' ' -f 1)"
} > "${RUN_DIR}/ab-config.txt"

cd "${PROJECT_ROOT}"
"${EVAL_PYTHON}" -m trajectory_experiments.run_trajectories \
  --dataset "${EVAL_DATASET}" \
  --runs-dir "${RUNS_DIR}" \
  --run-id "${RUN_ID}" \
  --model "${MODEL_NAME}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --language python \
  --temperature "${TEMPERATURE}" \
  --workers "${WORKERS}" \
  --max-steps "${MAX_STEPS}" \
  --max-seconds "${MAX_SECONDS}" \
  --max-tokens "${MAX_TOKENS}" \
  --resume \
  --stop-on-runner-error \
  "${INSTANCE_ARGS[@]}"

