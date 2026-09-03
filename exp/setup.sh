#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

command -v python3.12 >/dev/null 2>&1 || die "python3.12 is required"
require_file "${PROJECT_ROOT}/trajectory_experiments/requirements.txt"
require_file "${TRAINING_INDEX}"

if [[ ! -x "${EVAL_PYTHON}" ]]; then
  python3.12 -m venv "${EVAL_VENV}"
fi

cd "${PROJECT_ROOT}"
"${EVAL_PYTHON}" -m pip install -r trajectory_experiments/requirements.txt
"${EVAL_PYTHON}" -m trajectory_experiments.prepare_swesmith

"${EVAL_PYTHON}" "${EXP_DIR}/prepare_eval_set.py" \
  --dataset "${EVAL_DATASET}" \
  --training-index "${TRAINING_INDEX}" \
  --output "${EVAL_MANIFEST}" \
  --ids-output "${TEST_IDS}" \
  --in-domain-repo "${IN_DOMAIN_REPO}" \
  --in-domain-count "${IN_DOMAIN_COUNT}" \
  --out-domain-count "${OUT_DOMAIN_COUNT}" \
  --seed "${EVAL_SEED}"

echo "Evaluation setup is ready."
echo "Manifest: ${EVAL_MANIFEST}"
echo "Task IDs: ${TEST_IDS}"

