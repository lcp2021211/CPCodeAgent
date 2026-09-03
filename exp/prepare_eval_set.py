#!/usr/bin/env python3
"""Freeze a leakage-free, deterministic SWE-smith test set for Base/LoRA A/B runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict, deque
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from swesmith.profiles import registry


def stable_key(seed: int, *parts: str) -> str:
    payload = ":".join((str(seed), *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_combined(instance_id: str) -> bool:
    return ".combine_file__" in instance_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def profile_for(instance: dict[str, Any]) -> tuple[str, Any] | None:
    instance_id = str(instance.get("instance_id", ""))
    key = str(instance.get("repo") or instance_id.rsplit(".", 1)[0])
    try:
        return key, registry.get(key)
    except KeyError:
        return None


def profile_language(profile: Any) -> str:
    return profile.__class__.__module__.rsplit(".", 1)[-1]


def repo_matches(key: str, profile: Any, requested: str) -> bool:
    aliases = {key, str(profile.repo_name), str(profile.mirror_name)}
    return requested in aliases


def balanced_across_repos(
    rows: Iterable[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, deque[dict[str, Any]]] = {}
    raw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_groups[row["repo"]].append(row)
    for repo, values in raw_groups.items():
        values.sort(key=lambda item: stable_key(seed, "out", item["instance_id"]))
        grouped[repo] = deque(values)

    repo_order = sorted(grouped, key=lambda repo: stable_key(seed, "repo", repo))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        added = False
        for repo in repo_order:
            if grouped[repo]:
                selected.append(grouped[repo].popleft())
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
    return selected


def load_existing(
    output: Path,
    ids_output: Path,
    expected: dict[str, Any],
) -> bool:
    if not output.exists():
        return False
    manifest = json.loads(output.read_text(encoding="utf-8"))
    policy = manifest.get("selection", {})
    mismatches = {
        key: (policy.get(key), value) for key, value in expected.items() if policy.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Frozen evaluation manifest uses different settings: {mismatches}. "
            "Choose new output paths, or explicitly pass --overwrite before inspecting results."
        )
    ids = [item["instance_id"] for item in manifest["instances"]]
    if not ids_output.exists():
        atomic_text(ids_output, "\n".join(ids) + "\n")
    print(f"Using existing frozen evaluation set: {output} ({len(ids)} tasks)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids-output", type=Path, required=True)
    parser.add_argument("--in-domain-repo", required=True)
    parser.add_argument("--in-domain-count", type=int, default=25)
    parser.add_argument("--out-domain-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="python")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.in_domain_count < 0 or args.out_domain_count < 0:
        parser.error("task counts must be non-negative")
    if args.in_domain_count + args.out_domain_count < 1:
        parser.error("at least one task is required")

    dataset_path = args.dataset.expanduser().resolve()
    index_path = args.training_index.expanduser().resolve()
    output = args.output.expanduser().resolve()
    ids_output = args.ids_output.expanduser().resolve()
    if not dataset_path.exists():
        parser.error(f"dataset does not exist: {dataset_path}")
    if not index_path.is_file():
        parser.error(f"training index does not exist: {index_path}")

    selection = {
        "seed": args.seed,
        "language": args.language,
        "in_domain_repo": args.in_domain_repo,
        "in_domain_count": args.in_domain_count,
        "out_domain_count": args.out_domain_count,
        "combine_file_instances_included": False,
    }
    if not args.overwrite and load_existing(output, ids_output, selection):
        return 0

    excluded_ids = sorted({str(row["instance_id"]) for row in read_jsonl(index_path)})
    excluded = set(excluded_ids)
    dataset = load_from_disk(str(dataset_path))
    in_pool: list[dict[str, Any]] = []
    out_pool: list[dict[str, Any]] = []

    for raw in dataset:
        instance = dict(raw)
        instance_id = str(instance.get("instance_id", ""))
        if (
            not instance_id
            or instance_id in excluded
            or is_combined(instance_id)
            or not str(instance.get("problem_statement", "")).strip()
        ):
            continue
        resolved = profile_for(instance)
        if resolved is None:
            continue
        key, profile = resolved
        if args.language and profile_language(profile) != args.language:
            continue
        row = {
            "instance_id": instance_id,
            "repo": key,
            "problem_sha256": hashlib.sha256(
                str(instance["problem_statement"]).encode("utf-8")
            ).hexdigest(),
        }
        if repo_matches(key, profile, args.in_domain_repo):
            row["group"] = "in_domain"
            in_pool.append(row)
        else:
            row["group"] = "out_domain"
            out_pool.append(row)

    in_pool.sort(key=lambda item: stable_key(args.seed, "in", item["instance_id"]))
    selected_in = in_pool[: args.in_domain_count]
    selected_out = balanced_across_repos(out_pool, args.out_domain_count, args.seed)
    if len(selected_in) != args.in_domain_count:
        raise ValueError(
            f"Only {len(selected_in)} eligible in-domain tasks remain; "
            f"requested {args.in_domain_count}."
        )
    if len(selected_out) != args.out_domain_count:
        raise ValueError(
            f"Only {len(selected_out)} eligible out-of-domain tasks remain; "
            f"requested {args.out_domain_count}."
        )

    selected = [*selected_in, *selected_out]
    selected_ids = [item["instance_id"] for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise AssertionError("duplicate instance IDs were selected")

    dataset_manifest_path = dataset_path.parent / "dataset_manifest.json"
    dataset_manifest = (
        json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest_path.exists()
        else {"path": str(dataset_path), "task_count": len(dataset)}
    )
    dataset_manifest["path"] = str(dataset_path)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "frozen": True,
        "selection": selection,
        "counts": {
            "total": len(selected),
            "in_domain": len(selected_in),
            "out_domain": len(selected_out),
            "excluded_training_and_dev_instances": len(excluded_ids),
            "out_domain_repositories": len({item["repo"] for item in selected_out}),
        },
        "dataset": dataset_manifest,
        "training_exclusion": {
            "index_path": str(index_path),
            "instance_count": len(excluded_ids),
            "instance_ids_sha256": hashlib.sha256(
                ("\n".join(excluded_ids) + "\n").encode("utf-8")
            ).hexdigest(),
        },
        "test_ids_sha256": hashlib.sha256(
            ("\n".join(selected_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "instances": selected,
    }
    atomic_text(output, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_text(ids_output, "\n".join(selected_ids) + "\n")
    print(
        f"Frozen {len(selected)} tasks: {len(selected_in)} in-domain + "
        f"{len(selected_out)} out-of-domain"
    )
    print(f"Excluded {len(excluded_ids)} training/dev instance IDs")
    print(f"Manifest: {output}")
    print(f"Task IDs: {ids_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
