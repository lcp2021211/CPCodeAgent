#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_file "${EVAL_MANIFEST}"
require_file "${TEST_IDS}"
mkdir -p "${LOGS_DIR}" "${RUNS_DIR}" "${REPORTS_DIR}"

SERVER_PID=""
SERVER_LOG=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

expected_model() {
  if [[ "$1" == "base" ]]; then
    echo "${BASE_SERVED_MODEL}"
  else
    echo "${LORA_SERVED_MODEL}"
  fi
}

start_server() {
  local arm="$1"
  local model_name
  local stamp
  model_name="$(expected_model "${arm}")"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  SERVER_LOG="${LOGS_DIR}/${RUN_TAG}-${arm}-${stamp}.log"

  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    die "port ${PORT} already has a healthy service; stop it before running the A/B experiment"
  fi

  bash "${EXP_DIR}/serve.sh" "${arm}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  echo "Starting ${arm} server (PID ${SERVER_PID}); log: ${SERVER_LOG}"

  local attempts=$((SERVER_START_TIMEOUT / 2))
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 80 "${SERVER_LOG}" >&2 || true
      die "${arm} server exited during startup"
    fi
    if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      local models
      models="$(curl -fsS "${BASE_URL}/models" -H "Authorization: Bearer ${API_KEY}")"
      if grep -Fq "${model_name}" <<<"${models}"; then
        echo "${arm} server is ready: ${model_name}"
        return 0
      fi
    fi
    sleep 2
  done
  tail -n 80 "${SERVER_LOG}" >&2 || true
  die "timed out waiting for ${arm} server after ${SERVER_START_TIMEOUT}s"
}

for ARM in base lora; do
  start_server "${ARM}"
  bash "${EXP_DIR}/run_eval.sh" "${ARM}"
  stop_server
done

BASE_SUMMARY="${RUNS_DIR}/${RUN_TAG}-base/summary.json"
LORA_SUMMARY="${RUNS_DIR}/${RUN_TAG}-lora/summary.json"
OUTPUT_PREFIX="${REPORTS_DIR}/${RUN_TAG}"
"${TRAIN_PYTHON}" "${EXP_DIR}/compare_results.py" \
  --eval-manifest "${EVAL_MANIFEST}" \
  --base-summary "${BASE_SUMMARY}" \
  --lora-summary "${LORA_SUMMARY}" \
  --output-prefix "${OUTPUT_PREFIX}"

echo "A/B evaluation complete."
echo "Markdown report: ${OUTPUT_PREFIX}.md"
echo "JSON report: ${OUTPUT_PREFIX}.json"

