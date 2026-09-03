#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

command -v python3.12 >/dev/null 2>&1 || die "python3.12 is required"
require_file "${PROJECT_ROOT}/trajectory_experiments/requirements.txt"
require_file "${EXP_TRAINING_INDEX}"

if [[ ! -x "${EXP_EVAL_PYTHON}" ]]; then
  python3.12 -m venv "${EXP_EVAL_VENV}"
fi

cd "${PROJECT_ROOT}"
"${EXP_EVAL_PYTHON}" -m pip install -r trajectory_experiments/requirements.txt
"${EXP_EVAL_PYTHON}" -m trajectory_experiments.prepare_swesmith

"${EXP_EVAL_PYTHON}" "${EXP_DIR}/prepare_eval_set.py" \
  --dataset "${EXP_EVAL_DATASET}" \
  --training-index "${EXP_TRAINING_INDEX}" \
  --output "${EXP_EVAL_MANIFEST}" \
  --ids-output "${EXP_TEST_IDS}" \
  --in-domain-repo "${EXP_IN_DOMAIN_REPO}" \
  --in-domain-count "${EXP_IN_DOMAIN_COUNT}" \
  --out-domain-count "${EXP_OUT_DOMAIN_COUNT}" \
  --seed "${EXP_EVAL_SEED}"

echo "Evaluation setup is ready."
echo "Manifest: ${EXP_EVAL_MANIFEST}"
echo "Task IDs: ${EXP_TEST_IDS}"
