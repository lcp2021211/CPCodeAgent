# Crash consistency

CPCodeAgent treats the Journal as a write-ahead action log. Recovery does not infer that
an action is safe merely because the process restarted or the workspace changed; it uses
durable lifecycle markers plus a tool-owned effect contract.

## State machine

```text
ModelResponse
    │
    ▼
TOOL_CALL (INTENT, fsynced)
    │  contains Action, policy decision, EffectContract, model response sequence
    ▼
TOOL_STARTED (STARTED, fsynced)
    │  executor may now receive control
    ▼
TOOL_RESULT (COMMITTED, fsynced)
       contains result and observed before/after workspace revisions
```

`action_id` is the lifecycle identity. Model-provided call IDs remain in provider
messages, but are not trusted to be globally unique across turns.

## Invariants

1. No tool receives control before its intent and started marker are durable.
2. One action has at most one committed result.
3. An intent without a started marker is safe to execute: the executor never received it.
4. A started action without a result is never replayed merely because it is pending.
5. Model-visible tool results come only from committed Journal events.
6. A completed model response is reused after restart; the provider is not called again
   just to recreate a missing `FINAL` event.
7. Recovery preserves the model's original tool-call order; a later unstarted action
   cannot cross an earlier unresolved side-effect barrier.

The `ActionLedger` is a pure projection of these events. It stores no parallel mutable
database, so restart and audit use the same source of truth as context reconstruction.

## Effect contracts

Tools choose one recovery mode before `TOOL_STARTED`:

| Mode | Intended use | Started action with no result |
|---|---|---|
| `retry_safe` | reads and genuinely idempotent operations | execute again |
| `verify_files` | deterministic atomic file transition | inspect before/after digests |
| `manual` | commands, external writes, opaque side effects | return `UNKNOWN_COMMIT` |

`write_file` and `edit_file` record a workspace-relative file digest before the action
and the exact desired digest after it. On recovery:

```text
current == after   → effect is present; commit a recovered success
current == before  → atomic replace did not commit; retry safely
otherwise          → RECOVERY_CONFLICT; preserve the current file
```

The comparison is target-specific. An unrelated workspace change cannot turn a completed
write into an unknown result, and cannot authorize overwriting a divergent target.

## Atomic file boundary

Runtime-owned writes use a staging file tied to `action_id`, fsync its contents, replace
the destination atomically, then fsync the destination directory. Recovery removes only
staging files belonging to that interrupted action. Internal staging files are excluded
from workspace revisions.

This gives a single-file transition a recoverable before-or-after boundary. Arbitrary
commands deliberately do not claim this property: they can touch processes, databases,
or many files, so a missing result after `TOOL_STARTED` requires confirmation.

## Extension tools

A custom tool may override `Tool.effect_contract(...)`. It should use `retry_safe` only
when repeating a started operation is semantically safe. Deterministic single-file tools
may return a `verify_files` contract. Everything else should keep the default `manual`
mode.

The contract does not grant capabilities and does not bypass policy. It is captured only
after classification and approval, and it controls recovery only.

## Legacy journals

Older journals contain `TOOL_CALL` and `TOOL_RESULT` but no `TOOL_STARTED` or `action_id`.
They remain readable. A pending legacy read may be retried, but a pending legacy
non-idempotent action is reported as `UNKNOWN_COMMIT`, because its start state cannot be
proven.
