"""Export strict, officially resolved trajectories as step-level Agent SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = (
    ROOT.parent / "trajectory_experiments" / "runs" / "qwen37-python-pptx-100"
)
DEFAULT_OUTPUT = ROOT / "data"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_arguments(value: Any) -> str:
    """Return OpenAI-compatible JSON-string function arguments."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return canonical_json(value if value is not None else {})


def normalize_tool_call(raw: dict[str, Any]) -> dict[str, Any]:
    function = raw.get("function")
    if isinstance(function, dict):
        name = function.get("name", "")
        arguments = function.get("arguments", {})
    else:
        name = raw.get("name", "")
        arguments = raw.get("arguments", {})
    if not name:
        raise ValueError(f"Tool call has no function name: {raw}")
    call = {
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": normalize_arguments(arguments),
        },
    }
    if raw.get("id"):
        call["id"] = str(raw["id"])
    return call


def normalize_message(raw: dict[str, Any]) -> dict[str, Any]:
    role = str(raw.get("role", ""))
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"Unexpected message role: {role!r}")
    message: dict[str, Any] = {"role": role, "content": raw.get("content")}
    if raw.get("tool_calls"):
        message["tool_calls"] = [normalize_tool_call(item) for item in raw["tool_calls"]]
    if role == "tool" and raw.get("tool_call_id"):
        message["tool_call_id"] = str(raw["tool_call_id"])
    return message


def response_message(response: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.get("content") or None,
    }
    if response.get("tool_calls"):
        message["tool_calls"] = [
            normalize_tool_call(item) for item in response["tool_calls"]
        ]
    if message["content"] is None and not message.get("tool_calls"):
        raise ValueError("Model response has neither content nor tool calls")
    return message


def is_strict_positive(summary: dict[str, Any]) -> bool:
    return (
        summary.get("agent", {}).get("status") == "succeeded"
        and summary.get("evaluation", {}).get("resolved") is True
    )


def stable_key(seed: int, instance_id: str) -> str:
    return sha256_bytes(f"{seed}:{instance_id}".encode())


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def export(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    strict_tasks: list[tuple[Path, dict[str, Any]]] = []
    for summary_path in sorted(run_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if is_strict_positive(summary):
            strict_tasks.append((summary_path.parent, summary))
    if not strict_tasks:
        raise ValueError("No strict positive trajectories were found")

    task_ids = [summary["instance_id"] for _, summary in strict_tasks]
    eval_count = max(1, round(len(task_ids) * args.eval_ratio))
    eval_ids = set(sorted(task_ids, key=lambda value: stable_key(args.seed, value))[:eval_count])

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    index_rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    duplicates = 0
    teacher_token_estimates: list[int] = []
    tool_call_samples = 0

    for task_dir, summary in strict_tasks:
        instance_id = summary["instance_id"]
        split = "eval" if instance_id in eval_ids else "train"
        calls_path = task_dir / "model_calls.jsonl"
        if not calls_path.is_file():
            raise FileNotFoundError(f"Missing model calls: {calls_path}")
        for line_number, line in enumerate(calls_path.read_text(encoding="utf-8").splitlines(), 1):
            call = json.loads(line)
            if "error" in call or "response" not in call:
                raise ValueError(f"Strict trajectory contains failed call: {calls_path}:{line_number}")
            prompt = [normalize_message(item) for item in call["request"]["messages"]]
            completion = response_message(call["response"])
            tools = call["request"].get("tools", [])
            sample = {
                "messages": [*prompt, completion],
                # ms-swift requires the agent `tools` column to have string type.
                "tools": canonical_json(tools),
            }
            fingerprint = sha256_bytes(canonical_json(sample).encode())
            if fingerprint in fingerprints:
                duplicates += 1
                continue
            fingerprints.add(fingerprint)
            target_line = len(split_rows[split]) + 1
            split_rows[split].append(sample)
            response = call["response"]
            token_estimate = int(response.get("prompt_tokens", 0)) + int(
                response.get("completion_tokens", 0)
            )
            teacher_token_estimates.append(token_estimate)
            has_tool_call = bool(completion.get("tool_calls"))
            tool_call_samples += has_tool_call
            index_rows.append(
                {
                    "sample_id": fingerprint[:20],
                    "split": split,
                    "line": target_line,
                    "instance_id": instance_id,
                    "source_call_index": call.get("call_index"),
                    "source_request_hash": call.get("request_hash"),
                    "teacher_model": response.get("model"),
                    "teacher_token_estimate": token_estimate,
                    "has_tool_call": has_tool_call,
                }
            )

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    index_path = output_dir / "index.jsonl"
    write_jsonl_atomic(train_path, split_rows["train"])
    write_jsonl_atomic(eval_path, split_rows["eval"])
    write_jsonl_atomic(index_path, index_rows)

    ordered_tokens = sorted(teacher_token_estimates)

    def percentile(fraction: float) -> int:
        index = round((len(ordered_tokens) - 1) * fraction)
        return ordered_tokens[index]

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "format": "ms-swift/OpenAI messages with JSON-string tools",
        "unit": "one exact teacher decision point per sample",
        "quality_filter": "agent.status == succeeded AND evaluation.resolved == true",
        "source": {
            "run_dir": str(run_dir),
            "run_id": run_manifest.get("run_id"),
            "teacher_model": run_manifest.get("model"),
            "agent_commit": run_manifest.get("agent_commit"),
        },
        "split_policy": {
            "by": "trajectory instance_id (no cross-split trajectory leakage)",
            "seed": args.seed,
            "eval_ratio": args.eval_ratio,
        },
        "counts": {
            "source_strict_trajectories": len(strict_tasks),
            "train_trajectories": len(task_ids) - len(eval_ids),
            "eval_trajectories": len(eval_ids),
            "train_samples": len(split_rows["train"]),
            "eval_samples": len(split_rows["eval"]),
            "tool_call_samples": tool_call_samples,
            "final_answer_samples": len(fingerprints) - tool_call_samples,
            "deduplicated_samples": duplicates,
        },
        "teacher_token_estimates": {
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(ordered_tokens),
            "samples_over_16384": sum(value > 16_384 for value in ordered_tokens),
        },
        "files": {
            "train.jsonl": file_sha256(train_path),
            "eval.jsonl": file_sha256(eval_path),
            "index.jsonl": file_sha256(index_path),
        },
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 < args.eval_ratio < 1:
        raise SystemExit("--eval-ratio must be between 0 and 1")
    manifest = export(args)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote curated SFT data to {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

