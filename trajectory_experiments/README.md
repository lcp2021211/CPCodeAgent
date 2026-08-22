# SWE-smith trajectory experiments

This directory collects reproducible CPCodeAgent trajectories without modifying
anything under `cpcodeagent/`.

## What is recorded

Each task gets its own directory containing:

- `manifest.json`: task/repository/image/model/runtime metadata, with gold fields omitted;
- `model_calls.jsonl`: every provider attempt (including retries/fallbacks), with exact
  messages, tool schemas, responses or partial failures, hashes, token usage and latency;
- `journal.jsonl`: the original append-only CPCodeAgent journal;
- `events.jsonl` and `stream.txt`: tool/verifier progress and streamed text;
- `patch.diff` and `prediction.json`: the portable model patch;
- `official_evaluation/` and `evaluation_report.json`: SWE-smith grader evidence;
- `summary.json`: final status, usage, timing and resolved result.

Known credentials are recursively redacted. During agent execution the task container
has no network, and its original Git history is replaced by a fresh baseline so hidden
parent commits cannot be inspected. Official grading happens afterward in a clean
SWE-smith evaluation container.

## Setup

The local checkout is pinned in `SWE_SMITH_REVISION`; runtime dependencies are pinned in
`requirements.txt`. To recreate the virtual environment:

```bash
python3.12 -m venv trajectory_experiments/.venv
trajectory_experiments/.venv/bin/python -m pip install \
  -r trajectory_experiments/requirements.txt
```

Download and pin the task dataset:

```bash
trajectory_experiments/.venv/bin/python -m \
  trajectory_experiments.prepare_swesmith
```

Docker Desktop must be running. On Apple Silicon the runner defaults to the official
`linux/x86_64` images through Docker emulation.

## Run trajectories

One deterministic Python task:

```bash
trajectory_experiments/.venv/bin/python -m \
  trajectory_experiments.run_trajectories \
  --env-file /absolute/path/to/.env \
  --run-id qwen38-smoke \
  --count 1 --seed 42
```

One hundred tasks, with two task containers/API calls in parallel:

```bash
trajectory_experiments/.venv/bin/python -m \
  trajectory_experiments.run_trajectories \
  --env-file /absolute/path/to/.env \
  --run-id qwen37-python-pptx-100 \
  --repo swesmith/scanny__python-pptx.278b47b1 \
  --count 100 --seed 42 --workers 2 --resume --stop-on-runner-error
```

Constraining one repository reuses its Docker image and avoids downloading many large
images. Omit `--repo` for a diverse cross-repository sample.

Use `--instance-id ID` for an exact task, `--repo REPO` to constrain the repository,
`--no-evaluate` to collect patches without running the official grader, and
`--request-options-json '{"reasoning_effort":"high"}'` for provider-specific options.
Random sampling excludes `combine_file` instances by default because some combine
multiple mutations while the issue text describes only one. Use `--include-combined`
only if that behavior is intentional; exact `--instance-id` requests remain available
for auditing those tasks.

Each model call and journal event is flushed and fsynced. Each finished task gets an
atomic `summary.json`, and the run-level summary is refreshed after every task. With
`--stop-on-runner-error`, an exhausted API quota stops pending work. Re-running the same
command preserves completed tasks, archives failed or interrupted attempts under the
task's `attempts/` directory, and retries only unfinished tasks.
