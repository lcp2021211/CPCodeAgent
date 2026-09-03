#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_file "${EXP_EVAL_MANIFEST}"
require_file "${EXP_TEST_IDS}"
mkdir -p "${EXP_LOGS_DIR}" "${EXP_RUNS_DIR}" "${EXP_REPORTS_DIR}"

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
    echo "${EXP_BASE_SERVED_MODEL}"
  else
    echo "${EXP_LORA_SERVED_MODEL}"
  fi
}

start_server() {
  local arm="$1"
  local model_name
  local stamp
  model_name="$(expected_model "${arm}")"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  SERVER_LOG="${EXP_LOGS_DIR}/${EXP_RUN_TAG}-${arm}-${stamp}.log"

  if curl -fsS "http://${EXP_HOST}:${EXP_PORT}/health" >/dev/null 2>&1; then
    die "port ${EXP_PORT} already has a healthy service; stop it before running the A/B experiment"
  fi

  bash "${EXP_DIR}/serve.sh" "${arm}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  echo "Starting ${arm} server (PID ${SERVER_PID}); log: ${SERVER_LOG}"

  local attempts=$((EXP_SERVER_START_TIMEOUT / 2))
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 80 "${SERVER_LOG}" >&2 || true
      die "${arm} server exited during startup"
    fi
    if curl -fsS "http://${EXP_HOST}:${EXP_PORT}/health" >/dev/null 2>&1; then
      local models
      models="$(curl -fsS "${EXP_BASE_URL}/models" -H "Authorization: Bearer ${EXP_API_KEY}")"
      if grep -Fq "${model_name}" <<<"${models}"; then
        echo "${arm} server is ready: ${model_name}"
        return 0
      fi
    fi
    sleep 2
  done
  tail -n 80 "${SERVER_LOG}" >&2 || true
  die "timed out waiting for ${arm} server after ${EXP_SERVER_START_TIMEOUT}s"
}

for ARM in base lora; do
  start_server "${ARM}"
  bash "${EXP_DIR}/run_eval.sh" "${ARM}"
  stop_server
done

BASE_SUMMARY="${EXP_RUNS_DIR}/${EXP_RUN_TAG}-base/summary.json"
LORA_SUMMARY="${EXP_RUNS_DIR}/${EXP_RUN_TAG}-lora/summary.json"
OUTPUT_PREFIX="${EXP_REPORTS_DIR}/${EXP_RUN_TAG}"
"${EXP_TRAIN_PYTHON}" "${EXP_DIR}/compare_results.py" \
  --eval-manifest "${EXP_EVAL_MANIFEST}" \
  --base-summary "${BASE_SUMMARY}" \
  --lora-summary "${LORA_SUMMARY}" \
  --output-prefix "${OUTPUT_PREFIX}"

echo "A/B evaluation complete."
echo "Markdown report: ${OUTPUT_PREFIX}.md"
echo "JSON report: ${OUTPUT_PREFIX}.json"
