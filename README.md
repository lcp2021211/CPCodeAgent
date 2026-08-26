# CPCodeAgent

CPCodeAgent is a compact coding-agent harness implemented in Python. Its goal is not to
maximize framework surface area, but to make four difficult properties explicit:

1. a session is a durable sequence of turns, while model context is only a replayable view;
2. one semantic `Action` classification drives scheduling, permission, and retry;
3. read calls may run concurrently while side effects cross deterministic barriers;
4. every side effect follows a durable intent/start/commit protocol, so interrupted
   actions are reconciled from evidence and never replayed blindly.

## Architecture

```text
Session
   ├── Turn 1        INPUT → THINK → ACT → CHECK → FINAL
   ├── Turn 2        INPUT → THINK → ACT → CHECK → FINAL
   └── Journal       append-only history, checkpoints, recovery
          │
          ├── Context       replay projection, compaction, active skill
          └── ActionRuntime classify → policy → intent → start → commit
                                      └── recovery contract → reconcile
```

The implementation deliberately has no generic hook bus, planner DAG, or built-in
multi-agent layer. Extension happens through a few direct interfaces: `Model`, `Tool`,
`Policy`, `Executor`, `SkillRegistry`, and `Verifier`.

## Data flow

1. `Harness` creates a session with a fixed workspace and policy, then appends one user
   input for each turn.
2. `ContextEngine` derives a model view from the journal. Old history may become a
   deterministic memory snapshot, while the original events stay intact.
3. The model returns text and/or `ToolCall` values.
4. Every tool first runs `classify(arguments) -> Action`. The action contains only its
   required capabilities, concrete targets, and idempotency.
5. `RunPolicy` returns `ALLOW`, `ASK`, or `DENY`; an authorized tool then creates a small
   `EffectContract` describing safe recovery.
6. The runtime fsyncs a `TOOL_CALL` intent before execution, fsyncs `TOOL_STARTED` before
   handing control to the tool, and commits one `TOOL_RESULT` with observed revisions.
7. Consecutive read-only actions execute in parallel. A write or external effect is a
   serial barrier.
8. Results and a workspace checkpoint are durably appended before the next model call.
9. A final model answer passes through one optional `Verifier` before the turn succeeds.
   The next user message starts a new turn with fresh budgets and the same durable history.

## Package layout

```text
cpcodeagent/
  kernel.py          four-state run loop and budgets
  session.py         session identity, turn projection, safe journal lookup
  context.py         replayable context views and deterministic compaction
  skills.py          SKILL.md discovery and progressive loading
  tools.py           Tool contract and barrier scheduler
  builtin_tools.py   small workspace coding toolset
  policy.py          one ALLOW / ASK / DENY decision
  executor.py        confined file I/O, local execution, Docker sandbox
  journal.py         durable append-only JSONL event journal
  recovery.py        action-ledger projection and effect contracts
  model.py           provider-neutral streaming, retry, one fallback
  ui.py              Rich spinner, token stream, tool and verifier progress
  verifier.py        explicit completion gate
```

## Skill model

At startup, only a skill's name, description, and version enter the context. The model
must call `use_skill` to snapshot the full instructions into the journal and activate
them. A skill can require tools but cannot grant authority. Supporting files are read
through `read_skill_resource`; scripts still go through the normal command tool and
sandbox.

Project skills live under `.cpcodeagent/skills/<name>/SKILL.md`. A minimal example:

```markdown
---
name: debugging
description: Diagnose a reproducible failure using evidence-first iterations.
requires-tools: [read_file, search_text, edit_file, run_command]
---

Reproduce the failure, trace the smallest relevant path, make a focused change, and verify it.
```

## Failure semantics

The action journal is a write-ahead log with three externally meaningful states:

```text
INTENT --durable TOOL_STARTED--> STARTED --durable TOOL_RESULT--> COMMITTED
```

Recovery makes only three decisions:

- **Retry** when there is no `TOOL_STARTED` marker, or when a started operation is
  explicitly retry-safe.
- **Reconcile** deterministic file writes against their recorded before/after content
  digests. A matching postcondition is committed as recovered; a matching precondition
  proves the atomic replace did not happen and permits one safe retry.
- **Stop** with `RECOVERY_CONFLICT` when a file matches neither state, or
  `UNKNOWN_COMMIT` when an opaque side effect such as a command started but cannot be
  verified. Neither case is automatically replayed.

Runtime-owned file writes use per-action staging names, fsync file contents, atomically
replace the target, and fsync the containing directory. An interrupted final model
response is also settled from the journal instead of being requested from the provider
a second time. See [`docs/crash-consistency.md`](docs/crash-consistency.md) for the
invariants and extension contract.

Model API retries are owned by `ResilientModel`, so provider SDK retries should remain
disabled. Context overflow triggers one forced compaction. A provider can have one
explicit fallback model. Three identical action batches trigger one recovery instruction;
if the loop repeats, the turn stops with its checkpoint intact.

## Streaming terminal UX

Every provider call uses `stream=True`. Text deltas are rendered immediately instead of
waiting for the complete response. Function-call names and JSON arguments are also
streamed by providers, so `model.py` reassembles their fragments by call index before the
normal policy and tool runtime sees them. Providers that reject OpenAI's optional usage
stream field are retried once without that field while keeping streaming enabled.

The terminal uses one neutral progress-event stream rather than UI hooks spread through
the harness:

```text
⠋ Thinking…
agent › I will inspect the relevant files first.
▸ search_text {"query": "SessionState", "pattern": "**/*.py"}
⠋ Running 1 tool…
✓ search_text cpcodeagent/session.py:32:class SessionState:
agent › The session boundary is enforced in two places…
succeeded turn=turn-0002  steps=3  tokens=1842
```

If a connection fails before any text arrives, the existing bounded retry/fallback policy
applies. Once partial text has reached the terminal, automatic retry is suppressed so the
same paragraph is not printed twice. Only a completed model response is committed to the
durable Journal.

## Quick start

Requires Python 3.11+.

```bash
python -m pip install -e '.[dev]'

# Edit the existing .env file and fill in OPENAI_API_KEY first.
# Environment variables and CLI arguments can still override it.

# Start a durable multi-turn session.
cpcodeagent
```

CPCodeAgent automatically finds the nearest `.env` from the current directory upward.
The repository includes `.env.example`, while the real `.env` is ignored by Git. The
configuration precedence is:

```text
CLI argument > shell environment > .env > built-in default
```

The supported `.env` settings are:

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=
CPCODEAGENT_MODEL=gpt-4.1
CPCODEAGENT_FALLBACK_MODEL=

CPCODEAGENT_WORKSPACE=.
CPCODEAGENT_EXECUTOR=local
CPCODEAGENT_DOCKER_IMAGE=python:3.12-slim
CPCODEAGENT_JOURNAL_DIR=~/.cpcodeagent/runs

CPCODEAGENT_MAX_STEPS=40
CPCODEAGENT_MAX_SECONDS=1800
CPCODEAGENT_MAX_TOKENS=200000
CPCODEAGENT_VERIFY=
```

`OPENAI_BASE_URL` makes the same adapter usable with OpenAI-compatible providers. A
resumed session still restores its original workspace, policy, and executor; model and
per-turn budget values are read when the CLI starts.

The interactive shell keeps one session open until `/exit` or EOF:

```text
CPCodeAgent session: a83f219d2c10
Workspace: /project
Commands: /status, /help, /exit

you> Inspect this repository and explain its architecture
agent> ...

you> Now fix the first issue you found and run its tests
agent> ...

you> /exit
Session saved: a83f219d2c10
```

For scripts and one-off automation, passing a task still runs one turn and exits:

```bash
cpcodeagent "Inspect this repository and fix the failing tests" \
  --workspace . \
  --verify "python -m unittest discover -s tests" \
  --executor docker
```

`DockerExecutor` runs commands with no network, dropped Linux capabilities, resource
limits, and only the workspace mounted writable. `LocalExecutor` is useful for trusted
development: its Python file tools are path-confined, but local subprocesses are not an
OS sandbox, so the CLI prints an explicit warning.

Resume a durable session. If its latest turn was interrupted, CPCodeAgent reconciles that
turn first; otherwise it immediately returns to the interactive prompt:

```bash
cpcodeagent --resume <session-id>
```

Session journals live at `~/.cpcodeagent/runs/<session-id>.jsonl` by default. The journal
starts with immutable workspace, policy, and executor metadata, then stores any number of
sequential turns. Resume restores those boundaries instead of silently falling back to a
different permission set or executor.
Every turn has its own step/token/time budget. Context compaction affects only the model
view; raw events remain available for audit and replay.

Run the offline example and tests without an API key:

```bash
python -m examples.offline_demo
python -m unittest discover -s tests -v
```

## Current boundary

The core intentionally excludes multi-agent coordination, MCP discovery, and automatic
skill evolution. The append-only trajectories and skill hashes provide a clean base for
adding those experimentally without making the execution kernel more complicated.
