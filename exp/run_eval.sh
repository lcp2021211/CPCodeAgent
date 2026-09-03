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

require_file "${EXP_EVAL_PYTHON}"
require_dir "${EXP_EVAL_DATASET}"
require_file "${EXP_EVAL_MANIFEST}"
require_file "${EXP_TEST_IDS}"
command -v curl >/dev/null 2>&1 || die "curl is required"

if [[ "${ARM}" == "base" ]]; then
  EXP_MODEL_NAME="${EXP_BASE_SERVED_MODEL}"
else
  EXP_MODEL_NAME="${EXP_LORA_SERVED_MODEL}"
fi
EXP_RUN_ID="${EXP_RUN_TAG}-${ARM}"
EXP_RUN_DIR="${EXP_RUNS_DIR}/${EXP_RUN_ID}"

echo "Checking model server at ${EXP_SERVER_URL} for ${EXP_MODEL_NAME}..."
curl -fsS --max-time 10 "${EXP_SERVER_URL}/health" >/dev/null \
  || die "model server is not healthy at ${EXP_SERVER_URL}; check the SSH tunnel"
EXP_MODELS_JSON="$(curl -fsS "${EXP_BASE_URL}/models" -H "Authorization: Bearer ${EXP_API_KEY}")"
grep -Fq "${EXP_MODEL_NAME}" <<<"${EXP_MODELS_JSON}" \
  || die "server does not expose expected model ${EXP_MODEL_NAME}: ${EXP_MODELS_JSON}"

EXP_INSTANCE_ARGS=()
while IFS= read -r EXP_INSTANCE_ID; do
  [[ -z "${EXP_INSTANCE_ID}" || "${EXP_INSTANCE_ID}" == \#* ]] && continue
  EXP_INSTANCE_ARGS+=(--instance-id "${EXP_INSTANCE_ID}")
done < "${EXP_TEST_IDS}"
[[ ${#EXP_INSTANCE_ARGS[@]} -gt 0 ]] || die "no test IDs found in ${EXP_TEST_IDS}"
EXP_TASK_COUNT=$((${#EXP_INSTANCE_ARGS[@]} / 2))

mkdir -p "${EXP_RUN_DIR}"
EXP_CONFIG_SNAPSHOT="${EXP_RUN_DIR}/ab-config.txt"
EXP_CONFIG_CANDIDATE="${EXP_RUN_DIR}/.ab-config.$$.tmp"
trap 'rm -f "${EXP_CONFIG_CANDIDATE}"' EXIT
{
  echo "arm=${ARM}"
  echo "model=${EXP_MODEL_NAME}"
  echo "base_url=${EXP_BASE_URL}"
  echo "temperature=${EXP_TEMPERATURE}"
  echo "seed=${EXP_EVAL_SEED}"
  echo "max_steps=${EXP_MAX_STEPS}"
  echo "max_seconds=${EXP_MAX_SECONDS}"
  echo "max_tokens=${EXP_MAX_TOKENS}"
  echo "context_window_tokens=${EXP_CONTEXT_WINDOW_TOKENS}"
  echo "workers=${EXP_WORKERS}"
  echo "eval_manifest=${EXP_EVAL_MANIFEST}"
  echo "test_ids_sha256=$(sha256_file "${EXP_TEST_IDS}")"
} > "${EXP_CONFIG_CANDIDATE}"
if [[ -f "${EXP_CONFIG_SNAPSHOT}" ]] && ! cmp -s \
  "${EXP_CONFIG_SNAPSHOT}" "${EXP_CONFIG_CANDIDATE}"; then
  diff -u "${EXP_CONFIG_SNAPSHOT}" "${EXP_CONFIG_CANDIDATE}" >&2 || true
  die "run configuration changed; choose a new EXP_RUN_TAG instead of mixing results"
fi
mv "${EXP_CONFIG_CANDIDATE}" "${EXP_CONFIG_SNAPSHOT}"
trap - EXIT

cd "${PROJECT_ROOT}"
echo "Starting ${ARM} evaluation: ${EXP_TASK_COUNT} tasks, seed=${EXP_EVAL_SEED}"
echo "Progress will be shown as [completed/total] with the current evaluation stage."
"${EXP_EVAL_PYTHON}" -u -m trajectory_experiments.run_trajectories \
  --dataset "${EXP_EVAL_DATASET}" \
  --runs-dir "${EXP_RUNS_DIR}" \
  --run-id "${EXP_RUN_ID}" \
  --model "${EXP_MODEL_NAME}" \
  --base-url "${EXP_BASE_URL}" \
  --api-key "${EXP_API_KEY}" \
  --language python \
  --temperature "${EXP_TEMPERATURE}" \
  --seed "${EXP_EVAL_SEED}" \
  --workers "${EXP_WORKERS}" \
  --max-steps "${EXP_MAX_STEPS}" \
  --max-seconds "${EXP_MAX_SECONDS}" \
  --max-tokens "${EXP_MAX_TOKENS}" \
  --context-window-tokens "${EXP_CONTEXT_WINDOW_TOKENS}" \
  --resume \
  --stop-on-runner-error \
  "${EXP_INSTANCE_ARGS[@]}"
