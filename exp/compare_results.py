#!/usr/bin/env python3
"""Create a paired Base-vs-LoRA report from durable SWE-smith run summaries."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

Z_95 = 1.959963984540054


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


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


def task_map(summary: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task in summary.get("tasks", []):
        instance_id = str(task.get("instance_id", ""))
        if not instance_id:
            raise ValueError(f"{label} summary contains a task without instance_id")
        if instance_id in result:
            raise ValueError(f"{label} summary contains duplicate task {instance_id}")
        result[instance_id] = task
    return result


def is_resolved(task: dict[str, Any]) -> bool:
    return bool(task.get("evaluation", {}).get("resolved"))


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def values(tasks: Iterable[dict[str, Any]], getter: Callable[[dict[str, Any]], Any]) -> list[float]:
    result: list[float] = []
    for task in tasks:
        value = number(getter(task))
        if value is not None:
            result.append(value)
    return result


def describe(data: list[float]) -> dict[str, float | None]:
    if not data:
        return {"mean": None, "median": None, "sum": None}
    return {
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "sum": sum(data),
    }


def wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + Z_95**2 / total
    center = (p + Z_95**2 / (2 * total)) / denominator
    radius = Z_95 * math.sqrt(p * (1 - p) / total + Z_95**2 / (4 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def arm_metrics(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = sum(is_resolved(task) for task in tasks)
    agent_success = sum(task.get("agent", {}).get("status") == "succeeded" for task in tasks)
    false_success = sum(
        task.get("agent", {}).get("status") == "succeeded" and not is_resolved(task)
        for task in tasks
    )
    empty_patch = sum(number(task.get("agent", {}).get("patch_bytes")) == 0 for task in tasks)
    runner_errors = sum(
        task.get("agent", {}).get("status") == "runner_error"
        or task.get("evaluation", {}).get("status") == "runner_error"
        for task in tasks
    )
    metrics = {
        "tasks": len(tasks),
        "resolved": resolved,
        "resolve_rate": resolved / len(tasks) if tasks else None,
        "resolve_rate_wilson_95": wilson(resolved, len(tasks)),
        "agent_success": agent_success,
        "agent_success_rate": agent_success / len(tasks) if tasks else None,
        "false_success": false_success,
        "false_success_rate": false_success / len(tasks) if tasks else None,
        "empty_patch": empty_patch,
        "empty_patch_rate": empty_patch / len(tasks) if tasks else None,
        "runner_errors": runner_errors,
        "runner_error_rate": runner_errors / len(tasks) if tasks else None,
        "agent_statuses": dict(
            sorted(
                Counter(
                    str(task.get("agent", {}).get("status", "missing")) for task in tasks
                ).items()
            )
        ),
        "evaluation_statuses": dict(
            sorted(
                Counter(
                    str(task.get("evaluation", {}).get("status", "missing")) for task in tasks
                ).items()
            )
        ),
    }
    field_getters: dict[str, Callable[[dict[str, Any]], Any]] = {
        "steps": lambda task: task.get("agent", {}).get("steps"),
        "agent_tokens": lambda task: task.get("agent", {}).get("tokens"),
        "model_calls": lambda task: task.get("model_usage", {}).get("model_calls"),
        "prompt_tokens": lambda task: task.get("model_usage", {}).get("prompt_tokens"),
        "completion_tokens": lambda task: task.get("model_usage", {}).get("completion_tokens"),
        "model_seconds": lambda task: task.get("model_usage", {}).get("model_seconds"),
        "wall_seconds": lambda task: task.get("wall_seconds"),
        "patch_bytes": lambda task: task.get("agent", {}).get("patch_bytes"),
    }
    metrics["efficiency"] = {
        name: describe(values(tasks, getter)) for name, getter in field_getters.items()
    }
    resolved_tasks = [task for task in tasks if is_resolved(task)]
    metrics["resolved_task_efficiency"] = {
        name: describe(values(resolved_tasks, getter)) for name, getter in field_getters.items()
    }
    if resolved:
        metrics["cost_per_resolved"] = {
            name: sum(values(tasks, getter)) / resolved
            for name, getter in field_getters.items()
            if name
            in {
                "model_calls",
                "prompt_tokens",
                "completion_tokens",
                "model_seconds",
                "wall_seconds",
            }
        }
    else:
        metrics["cost_per_resolved"] = None
    return metrics


def quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of empty data")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap(
    base: list[int], lora: list[int], samples: int, seed: int
) -> list[float] | None:
    if not base:
        return None
    rng = random.Random(seed)
    size = len(base)
    differences: list[float] = []
    for _ in range(samples):
        total = 0
        for _ in range(size):
            index = rng.randrange(size)
            total += lora[index] - base[index]
        differences.append(total / size)
    differences.sort()
    return [quantile(differences, 0.025), quantile(differences, 0.975)]


def mcnemar_exact(base_only: int, lora_only: int) -> float:
    discordant = base_only + lora_only
    if discordant == 0:
        return 1.0
    smaller = min(base_only, lora_only)
    probability = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


def fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def fmt_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def markdown(report: dict[str, Any]) -> str:
    base = report["arms"]["base"]
    lora = report["arms"]["lora"]
    paired = report["paired"]
    lines = [
        "# Qwen3.5-9B Base vs LoRA evaluation",
        "",
        f"Frozen paired tasks: **{report['task_count']}**",
        "",
        "## Primary metrics",
        "",
        "| Metric | Base | LoRA | LoRA - Base |",
        "|---|---:|---:|---:|",
        f"| Resolve@1 | {fmt_percent(base['resolve_rate'])} | {fmt_percent(lora['resolve_rate'])} | {fmt_percent(paired['resolve_rate_delta'])} |",
        f"| Agent success rate | {fmt_percent(base['agent_success_rate'])} | {fmt_percent(lora['agent_success_rate'])} | {fmt_percent(lora['agent_success_rate'] - base['agent_success_rate'])} |",
        f"| False-success rate | {fmt_percent(base['false_success_rate'])} | {fmt_percent(lora['false_success_rate'])} | {fmt_percent(lora['false_success_rate'] - base['false_success_rate'])} |",
        f"| Empty-patch rate | {fmt_percent(base['empty_patch_rate'])} | {fmt_percent(lora['empty_patch_rate'])} | {fmt_percent(lora['empty_patch_rate'] - base['empty_patch_rate'])} |",
        f"| Runner-error rate | {fmt_percent(base['runner_error_rate'])} | {fmt_percent(lora['runner_error_rate'])} | {fmt_percent(lora['runner_error_rate'] - base['runner_error_rate'])} |",
        "",
        f"Base Resolve@1 95% Wilson CI: `{base['resolve_rate_wilson_95']}`",
        "",
        f"LoRA Resolve@1 95% Wilson CI: `{lora['resolve_rate_wilson_95']}`",
        "",
        f"Paired bootstrap 95% CI for Resolve@1 delta: `{paired['resolve_rate_delta_bootstrap_95']}`",
        "",
        f"Exact McNemar p-value: `{paired['mcnemar_exact_p']:.6g}`",
        "",
        "## Paired outcomes",
        "",
        f"- Both resolved: {paired['both_resolved']}",
        f"- LoRA only: {paired['lora_only']}",
        f"- Base only: {paired['base_only']}",
        f"- Neither resolved: {paired['neither_resolved']}",
        "",
        "## Mean efficiency over all tasks",
        "",
        "| Metric | Base | LoRA |",
        "|---|---:|---:|",
    ]
    for key in (
        "steps",
        "agent_tokens",
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "model_seconds",
        "wall_seconds",
        "patch_bytes",
    ):
        lines.append(
            f"| {key} | {fmt_number(base['efficiency'][key]['mean'])} | "
            f"{fmt_number(lora['efficiency'][key]['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Resolve@1 by group",
            "",
            "| Group | Tasks | Base | LoRA | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in ("in_domain", "out_domain"):
        group_result = report["groups"][group]
        lines.append(
            f"| {group} | {group_result['tasks']} | "
            f"{fmt_percent(group_result['base']['resolve_rate'])} | "
            f"{fmt_percent(group_result['lora']['resolve_rate'])} | "
            f"{fmt_percent(group_result['delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic task IDs",
            "",
            "LoRA-only:",
            "",
            *([f"- `{item}`" for item in paired["lora_only_ids"]] or ["- None"]),
            "",
            "Base-only:",
            "",
            *([f"- `{item}`" for item in paired["base_only_ids"]] or ["- None"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--lora-summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    manifest = load_json(args.eval_manifest.expanduser().resolve())
    base_summary = load_json(args.base_summary.expanduser().resolve())
    lora_summary = load_json(args.lora_summary.expanduser().resolve())
    expected_ids = [str(item["instance_id"]) for item in manifest["instances"]]
    group_by_id = {str(item["instance_id"]): str(item["group"]) for item in manifest["instances"]}
    base_by_id = task_map(base_summary, "base")
    lora_by_id = task_map(lora_summary, "lora")

    missing_base = [item for item in expected_ids if item not in base_by_id]
    missing_lora = [item for item in expected_ids if item not in lora_by_id]
    if (missing_base or missing_lora) and not args.allow_incomplete:
        raise ValueError(
            f"Incomplete paired runs. Missing base={missing_base}, missing lora={missing_lora}. "
            "Resume both runs or pass --allow-incomplete for a provisional report."
        )
    paired_ids = [item for item in expected_ids if item in base_by_id and item in lora_by_id]
    if not paired_ids:
        raise ValueError("No paired task results are available")

    base_tasks = [base_by_id[item] for item in paired_ids]
    lora_tasks = [lora_by_id[item] for item in paired_ids]
    base_binary = [int(is_resolved(task)) for task in base_tasks]
    lora_binary = [int(is_resolved(task)) for task in lora_tasks]
    both = sum(base and lora for base, lora in zip(base_binary, lora_binary))
    base_only_ids = [
        item for item, base, lora in zip(paired_ids, base_binary, lora_binary) if base and not lora
    ]
    lora_only_ids = [
        item for item, base, lora in zip(paired_ids, base_binary, lora_binary) if lora and not base
    ]
    neither = sum(not base and not lora for base, lora in zip(base_binary, lora_binary))

    arms = {"base": arm_metrics(base_tasks), "lora": arm_metrics(lora_tasks)}
    delta = arms["lora"]["resolve_rate"] - arms["base"]["resolve_rate"]
    groups: dict[str, Any] = {}
    for group in ("in_domain", "out_domain"):
        ids = [item for item in paired_ids if group_by_id[item] == group]
        group_base = arm_metrics([base_by_id[item] for item in ids])
        group_lora = arm_metrics([lora_by_id[item] for item in ids])
        groups[group] = {
            "tasks": len(ids),
            "base": group_base,
            "lora": group_lora,
            "delta": group_lora["resolve_rate"] - group_base["resolve_rate"] if ids else None,
        }

    report = {
        "schema_version": 1,
        "eval_manifest": str(args.eval_manifest.expanduser().resolve()),
        "task_count": len(paired_ids),
        "expected_task_count": len(expected_ids),
        "missing_base_ids": missing_base,
        "missing_lora_ids": missing_lora,
        "arms": arms,
        "groups": groups,
        "paired": {
            "both_resolved": both,
            "lora_only": len(lora_only_ids),
            "base_only": len(base_only_ids),
            "neither_resolved": neither,
            "lora_only_ids": lora_only_ids,
            "base_only_ids": base_only_ids,
            "resolve_rate_delta": delta,
            "resolve_rate_delta_bootstrap_95": paired_bootstrap(
                base_binary, lora_binary, args.bootstrap_samples, args.seed
            ),
            "mcnemar_exact_p": mcnemar_exact(len(base_only_ids), len(lora_only_ids)),
        },
    }

    prefix = args.output_prefix.expanduser().resolve()
    atomic_text(
        prefix.with_suffix(".json"),
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_text(prefix.with_suffix(".md"), markdown(report))
    print(f"Base Resolve@1: {fmt_percent(arms['base']['resolve_rate'])}")
    print(f"LoRA Resolve@1: {fmt_percent(arms['lora']['resolve_rate'])}")
    print(f"Delta: {fmt_percent(delta)}")
    print(f"Markdown: {prefix.with_suffix('.md')}")
    print(f"JSON: {prefix.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
