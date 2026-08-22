"""Validate exported Agent SFT JSONL without downloading the target model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(value)
    return rows


def validate_sample(sample: dict[str, Any], location: str) -> tuple[int, int]:
    if set(sample) != {"messages", "tools"}:
        raise ValueError(f"{location}: expected only messages/tools, got {sorted(sample)}")
    messages = sample["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{location}: messages must be a non-empty list")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{location}: final message must be assistant")
    if not isinstance(sample["tools"], str):
        raise ValueError(f"{location}: tools must be a JSON string for ms-swift")
    tools = json.loads(sample["tools"])
    tool_names = {
        item.get("function", {}).get("name")
        for item in tools
        if isinstance(item, dict)
    }
    calls = 0
    final_calls = 0
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{location}: invalid role at message {message_index}: {role}")
        if message.get("content") is not None and not isinstance(message.get("content"), str):
            raise ValueError(f"{location}: non-string content at message {message_index}")
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            name = function.get("name")
            if name not in tool_names:
                raise ValueError(f"{location}: undefined tool call {name!r}")
            arguments = function.get("arguments")
            if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
                raise ValueError(f"{location}: invalid arguments for {name!r}")
            calls += 1
            if message_index == len(messages) - 1:
                final_calls += 1
    return calls, final_calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    index = load_jsonl(data_dir / "index.jsonl")
    counts = Counter()
    split_instances: dict[str, set[str]] = {"train": set(), "eval": set()}
    seen_samples: set[str] = set()
    total_rows = 0
    for split in ("train", "eval"):
        path = data_dir / f"{split}.jsonl"
        expected_hash = manifest["files"][path.name]
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Checksum mismatch: {path}")
        rows = load_jsonl(path)
        total_rows += len(rows)
        for line_number, sample in enumerate(rows, 1):
            encoded = json.dumps(sample, ensure_ascii=False, sort_keys=True)
            fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
            if fingerprint in seen_samples:
                raise ValueError(f"Duplicate sample at {path}:{line_number}")
            seen_samples.add(fingerprint)
            calls, final_calls = validate_sample(sample, f"{path}:{line_number}")
            counts["tool_calls"] += calls
            counts["tool_call_targets"] += bool(final_calls)
            counts["final_answer_targets"] += not bool(final_calls)
    if len(index) != total_rows:
        raise ValueError(f"Index has {len(index)} rows but datasets have {total_rows}")
    for item in index:
        split_instances[item["split"]].add(item["instance_id"])
    overlap = split_instances["train"] & split_instances["eval"]
    if overlap:
        raise ValueError(f"Trajectory leakage across splits: {sorted(overlap)}")
    print(
        json.dumps(
            {
                "valid": True,
                "samples": total_rows,
                "train_trajectories": len(split_instances["train"]),
                "eval_trajectories": len(split_instances["eval"]),
                **counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

