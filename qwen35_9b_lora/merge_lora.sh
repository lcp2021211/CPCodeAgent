#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/checkpoint-N" >&2
  exit 2
fi

swift export \
  --adapters "$1" \
  --merge_lora true \
  --use_hf true

