# Qwen3.5-9B Base vs LoRA Agent evaluation

This directory runs a paired, end-to-end CPCodeAgent comparison on SWE-smith. The two
arms differ only in weights:

- `base`: untouched local Qwen3.5-9B;
- `lora`: the validation-loss-best adapter (`checkpoint-40` through the `best` symlink).

Both arms use the same frozen tasks, agent code, tools, official grader, greedy decoding,
context/output limits, and budgets. The single GPU runs the arms sequentially.

## Evaluation set

The default frozen set contains **50 tasks**:

- 25 in-domain tasks from `swesmith/scanny__python-pptx.278b47b1`;
- 25 out-of-domain Python tasks, balanced across repositories where possible.

Selection is deterministic with seed `20260903`. It excludes combined-mutation tasks,
tasks without problem statements, and every `instance_id` present in the LoRA training
index. This removes all 46 training trajectories and all 5 development trajectories used
to choose the best checkpoint. `prepare_eval_set.py` writes only IDs and problem hashes,
then refuses to silently replace a frozen set with different settings.

Fifty tasks are suitable for an initial comparison, not a high-precision leaderboard.
For a stronger conclusion use at least 100-200 frozen tasks; for a cheap plumbing smoke
test use a separate 5+5 manifest and run tag rather than changing the final test set after
seeing results.

## Metrics

The primary metric is official SWE-smith **Resolve@1** (`evaluation.resolved`). The report
also includes:

- paired outcomes: both solved, LoRA-only, Base-only, neither;
- 95% Wilson intervals and a paired-bootstrap interval for the Resolve@1 difference;
- exact McNemar p-value for discordant task pairs;
- in-domain and out-of-domain Resolve@1;
- agent-success, false-success, empty-patch, and runner-error rates;
- mean/median steps, agent tokens, model calls, prompt/completion tokens, model time,
  wall time, and patch size;
- token/time/call cost per officially resolved task in the JSON report.

Training loss and teacher-token accuracy are intentionally not primary metrics because a
coding agent can take multiple valid action sequences to produce the same correct patch.

## Prerequisites and dataset freeze

Docker must be running. Dataset preparation downloads a pinned SWE-smith revision, and
the first tasks may download Docker images.

```bash
cd /root/autodl-tmp/LXP/CPCodeAgent
bash exp/setup.sh
```

This installs the dedicated trajectory environment, prepares the dataset, and writes:

```text
exp/eval_set.json
exp/test_ids.txt
```

Copy `exp/config.env.example` to `exp/config.env` only if paths, counts, port, or budgets
need to be changed before the evaluation set is frozen.

## Run the complete A/B experiment

Make sure port 8000 is free, then run:

```bash
cd /root/autodl-tmp/LXP/CPCodeAgent
bash exp/run_ab.sh
```

The script performs these steps:

1. starts the BF16 base-model API;
2. runs and officially grades all frozen tasks;
3. stops the base server and releases GPU memory;
4. starts the same BF16 model with the best LoRA adapter;
5. runs the identical tasks;
6. writes the paired report.

Interrupted task runs are resumable: execute `bash exp/run_ab.sh` again. Completed task
summaries are retained and skipped by the runner.

## Manual operation

Each phase can also be run in separate terminals:

```bash
# Terminal A
bash exp/serve.sh base

# Terminal B
bash exp/run_eval.sh base
```

Stop Terminal A, then repeat with LoRA:

```bash
# Terminal A
bash exp/serve.sh lora

# Terminal B
bash exp/run_eval.sh lora
```

Generate or regenerate only the comparison report:

```bash
qwen35_9b_lora/.venv/bin/python exp/compare_results.py \
  --eval-manifest exp/eval_set.json \
  --base-summary exp/runs/qwen35-9b-ab-n50-seed20260903-base/summary.json \
  --lora-summary exp/runs/qwen35-9b-ab-n50-seed20260903-lora/summary.json \
  --output-prefix exp/reports/qwen35-9b-ab-n50-seed20260903
```

## Outputs

All evidence is kept under `exp/`:

```text
exp/eval_set.json                 frozen selection policy and IDs
exp/test_ids.txt                  exact paired task order
exp/logs/                         timestamped model-server logs
exp/runs/<run-id>/<instance-id>/  prompts, streams, tools, journal, patch, grader evidence
exp/runs/<run-id>/summary.json    arm-level durable summary
exp/reports/<run-tag>.md          human-readable paired report
exp/reports/<run-tag>.json        full machine-readable metrics
```

The local API key is only a loopback authentication token and is never written to the
experiment configuration snapshot.

