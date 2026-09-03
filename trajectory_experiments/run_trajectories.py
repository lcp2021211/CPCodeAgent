"""Collect CPCodeAgent trajectories on isolated SWE-smith tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import docker
import swesmith.harness.utils as swesmith_utils
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
)
from swebench.harness.docker_build import close_logger
from swesmith.harness.grading import get_eval_report
from swesmith.profiles import registry

from cpcodeagent.context import ContextEngine
from cpcodeagent.journal import Journal
from cpcodeagent.kernel import Harness
from cpcodeagent.model import OpenAICompatibleModel, ResilientModel
from cpcodeagent.policy import RunPolicy
from cpcodeagent.types import RunLimits
from trajectory_experiments.container_runtime import (
    PersistentContainerExecutor,
    RequirePatchVerifier,
    build_container_runtime,
    disconnect_network,
    replace_git_history_with_baseline,
)
from trajectory_experiments.recording import (
    EventRecorder,
    RecordedModelGroup,
    RecordingModel,
    RecordingState,
    Redactor,
    utc_now,
    write_json,
)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_DATASET = ROOT / "data" / "swesmith_train"
DEFAULT_RUNS = ROOT / "runs"
GOLD_FIELDS = {
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "FAIL_TO_FAIL",
    "PASS_TO_FAIL",
}
EVALUATION_LOCK = threading.Lock()


def int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def stable_order(instance_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()


def is_combined_instance(instance_id: str) -> bool:
    """Return whether SWE-smith merged multiple mutations into one task."""
    return ".combine_file__" in instance_id


def profile_language(profile: Any) -> str:
    return profile.__class__.__module__.rsplit(".", 1)[-1]


def select_instances(
    dataset: Dataset,
    count: int,
    seed: int,
    language: str | None,
    repo: str | None,
    instance_ids: list[str] | None,
    include_combined: bool,
) -> list[dict[str, Any]]:
    requested = set(instance_ids or ())
    selected: list[dict[str, Any]] = []
    for raw in dataset:
        instance = dict(raw)
        instance_id = str(instance.get(KEY_INSTANCE_ID, ""))
        if requested and instance_id not in requested:
            continue
        # Combined mutations can contain held-out failures not described by the
        # generated issue statement. Keep exact-ID requests possible for audit,
        # but exclude them from normal trajectory sampling by default.
        if not requested and not include_combined and is_combined_instance(instance_id):
            continue
        key = instance.get("repo", instance_id.rsplit(".", 1)[0])
        try:
            profile = registry.get(key)
        except KeyError:
            continue
        if language and profile_language(profile) != language:
            continue
        if repo and repo not in {str(key), profile.repo_name, profile.mirror_name}:
            continue
        if not str(instance.get("problem_statement", "")).strip():
            continue
        selected.append(instance)

    if requested:
        found = {item[KEY_INSTANCE_ID] for item in selected}
        missing = requested - found
        if missing:
            raise ValueError(f"Requested instances were not found/eligible: {sorted(missing)}")
        selected.sort(key=lambda item: (instance_ids or []).index(item[KEY_INSTANCE_ID]))
        return selected

    selected.sort(key=lambda item: stable_order(item[KEY_INSTANCE_ID], seed))
    return selected[:count]


def public_task_metadata(instance: dict[str, Any], image: str) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in instance.items()
        if key not in GOLD_FIELDS and key not in {"tests_status"}
    }
    metadata["image"] = image
    return metadata


def task_prompt(instance: dict[str, Any]) -> str:
    return (
        "Fix the software issue described below in the current repository. "
        "Inspect the implementation before editing, make the smallest correct change, "
        "and run relevant visible tests. Do not inspect benchmark metadata, git history, "
        "or hidden evaluation assets. When the patch is ready, give a concise summary.\n\n"
        f"Issue:\n{instance['problem_statement']}"
    )


def summary_has_runner_error(summary: dict[str, Any]) -> bool:
    return (
        summary.get("agent", {}).get("status") == "runner_error"
        or summary.get("evaluation", {}).get("status") == "runner_error"
    )


def archive_previous_attempt(task_dir: Path) -> Path | None:
    """Preserve a failed/interrupted attempt before starting a clean retry."""
    if not task_dir.exists():
        return None
    existing = [path for path in task_dir.iterdir() if path.name != "attempts"]
    if not existing:
        return None
    archive_root = task_dir / "attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("attempt-%Y%m%dT%H%M%S.%fZ")
    destination = archive_root / stamp
    destination.mkdir()
    for path in existing:
        path.replace(destination / path.name)
    return destination


def force_profile_arch(profile: Any, arch: str) -> None:
    profile.arch = arch
    profile.__dict__.pop("_cache_image_exists", None)


def create_model(args: argparse.Namespace, task_dir: Path, redactor: Redactor) -> RecordedModelGroup:
    options = json.loads(args.request_options_json)
    if args.temperature is not None:
        options["temperature"] = args.temperature
    output_path = task_dir / "model_calls.jsonl"
    state = RecordingState(output_path, redactor)
    primary = RecordingModel(
        OpenAICompatibleModel(
            args.model,
            args.api_key,
            args.base_url,
            **options,
        ),
        output_path,
        {
            "provider_role": "primary",
            "model": args.model,
            "base_url": args.base_url,
            "request_options": options,
            "adapter": "cpcodeagent.model.OpenAICompatibleModel",
        },
        redactor,
        state,
    )
    fallback = (
        RecordingModel(
            OpenAICompatibleModel(
                args.fallback_model,
                args.api_key,
                args.base_url,
                **options,
            ),
            output_path,
            {
                "provider_role": "fallback",
                "model": args.fallback_model,
                "base_url": args.base_url,
                "request_options": options,
                "adapter": "cpcodeagent.model.OpenAICompatibleModel",
            },
            redactor,
            state,
        )
        if args.fallback_model
        else None
    )
    resilient = ResilientModel(primary, fallback)
    return RecordedModelGroup(resilient, state)


def evaluate_patch(
    instance: dict[str, Any],
    patch: str,
    model_name: str,
    task_dir: Path,
    profile: Any,
    f2p_only: bool,
) -> dict[str, Any]:
    if not patch.strip():
        report = {"patch_exists": False, "resolved": False, "status": "empty_patch"}
        write_json(task_dir / "evaluation_report.json", report)
        return report

    evaluation_root = task_dir / "official_evaluation"
    evaluation_run_id = "grader"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    prediction = {
        KEY_INSTANCE_ID: instance[KEY_INSTANCE_ID],
        KEY_PREDICTION: patch,
        KEY_MODEL: model_name,
    }
    # SWE-smith uses identity with this module global to select its evaluation
    # checkout path (including restoration of held-out tests). Guard that small
    # upstream global so parallel agents cannot cross-wire grader directories.
    with EVALUATION_LOCK:
        swesmith_utils.RUN_EVALUATION_LOG_DIR = evaluation_root
        result = swesmith_utils.run_patch_in_container(
            dict(instance),
            evaluation_run_id,
            evaluation_root,
            profile.timeout,
            patch=patch,
            commit=instance[KEY_INSTANCE_ID],
            f2p_only=f2p_only,
            is_gold=False,
        )
    if result is None:
        report = {"patch_exists": True, "resolved": False, "status": "grader_error"}
        write_json(task_dir / "evaluation_report.json", report)
        return report

    logger, timed_out = result
    test_log = evaluation_root / evaluation_run_id / instance[KEY_INSTANCE_ID] / LOG_TEST_OUTPUT
    try:
        if timed_out:
            report = {
                "patch_exists": True,
                "resolved": False,
                "status": "timeout",
                "timeout": profile.timeout,
            }
        elif not test_log.exists():
            report = {"patch_exists": True, "resolved": False, "status": "grader_error"}
        else:
            report = get_eval_report(
                prediction,
                dict(instance),
                str(test_log),
                f2p_only=f2p_only,
            )
            report["status"] = "completed"
            report[KEY_MODEL] = model_name
    finally:
        close_logger(logger)

    report_path = evaluation_root / evaluation_run_id / instance[KEY_INSTANCE_ID] / LOG_REPORT
    write_json(report_path, report)
    write_json(task_dir / "evaluation_report.json", report)
    return report


def run_one(
    instance: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    dataset_manifest: dict[str, Any],
    task_number: int,
    task_total: int,
) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    progress = f"[{task_number}/{task_total}][{instance_id}]"
    task_dir = run_dir / instance_id
    summary_path = task_dir / "summary.json"
    if args.resume and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {"agent": {"status": "runner_error"}}
        if not summary_has_runner_error(summary):
            print(f"{progress} skipped (completed summary)", flush=True)
            return summary
        archived = archive_previous_attempt(task_dir)
        print(f"{progress} retrying; previous attempt archived at {archived}", flush=True)
    elif args.resume and task_dir.exists() and any(task_dir.iterdir()):
        archived = archive_previous_attempt(task_dir)
        print(f"{progress} retrying interrupted attempt; archived at {archived}", flush=True)

    task_dir.mkdir(parents=True, exist_ok=True)
    redactor = Redactor.from_environment()
    profile = registry.get_from_inst(instance)
    force_profile_arch(profile, args.container_arch)
    image = profile.image_name
    manifest = {
        "schema_version": 1,
        "started_at": utc_now(),
        "instance": public_task_metadata(instance, image),
        "dataset": dataset_manifest,
        "agent": {
            "package": "cpcodeagent",
            "source_commit": args.agent_commit,
            "tools": [
                "plan_write",
                "read_file",
                "list_files",
                "search_text",
                "write_file",
                "edit_file",
                "run_command",
            ],
        },
        "runtime": {
            "host_platform": platform.platform(),
            "container_arch": args.container_arch,
            "network_disabled_during_agent": not args.allow_container_network,
            "git_history_replaced": True,
        },
    }
    write_json(task_dir / "manifest.json", manifest, redactor)
    print(f"{progress} starting; image={image}", flush=True)

    container = None
    started = time.monotonic()
    recording_model: RecordedModelGroup | None = None
    try:
        container = profile.get_container(instance)
        if not args.allow_container_network:
            disconnect_network(container)
        replace_git_history_with_baseline(container)
        executor = PersistentContainerExecutor(container, image, REPO_ROOT)
        tools = build_container_runtime(executor)
        recording_model = create_model(args, task_dir, redactor)
        events = EventRecorder(
            task_dir,
            redactor,
            verbose=not args.quiet,
            prefix=progress,
        )
        harness = Harness(
            model=recording_model,
            tools=tools,
            context=ContextEngine(max_context_tokens=args.context_window_tokens),
            policy=RunPolicy(),
            verifier=RequirePatchVerifier(executor),
            limits=RunLimits(args.max_steps, args.max_seconds, args.max_tokens),
            event_sink=events,
        )
        outcome = harness.run(
            task_prompt(instance),
            executor,
            Journal(task_dir / "journal.jsonl"),
            re.sub(r"[^A-Za-z0-9_-]", "_", instance_id)[:80],
        )
        patch = executor.final_diff()
        (task_dir / "patch.diff").write_text(patch, encoding="utf-8")
        prediction = {
            KEY_INSTANCE_ID: instance_id,
            KEY_PREDICTION: patch,
            KEY_MODEL: args.model,
        }
        write_json(task_dir / "prediction.json", prediction, redactor)
        agent_result = {
            "status": outcome.status.value,
            "answer": outcome.answer,
            "steps": outcome.steps,
            "tokens": outcome.tokens,
            "patch_bytes": len(patch.encode("utf-8")),
        }
    except Exception as exc:  # noqa: BLE001 - task failures must become durable records
        patch = ""
        agent_result = {
            "status": "runner_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "patch_bytes": 0,
        }
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary task result
                agent_result["container_cleanup_error"] = str(cleanup_exc)

    evaluation: dict[str, Any] = {"status": "not_run", "resolved": False}
    if args.evaluate and patch.strip():
        print(f"{progress} grading patch", flush=True)
        try:
            evaluation = evaluate_patch(
                instance,
                patch,
                args.model,
                task_dir,
                profile,
                args.f2p_only,
            )
        except Exception as exc:  # noqa: BLE001 - grader failures must become durable records
            evaluation = {
                "status": "runner_error",
                "resolved": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(task_dir / "evaluation_report.json", evaluation, redactor)

    summary = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "model": args.model,
        "started_at": manifest["started_at"],
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": time.monotonic() - started,
        "agent": agent_result,
        "model_usage": recording_model.totals() if recording_model else {},
        "evaluation": evaluation,
    }
    write_json(summary_path, summary, redactor)
    print(
        f"{progress} finished; agent={agent_result['status']} "
        f"resolved={evaluation.get('resolved', False)} "
        f"seconds={summary['wall_seconds']:.1f}",
        flush=True,
    )
    return summary


def load_task_summaries(
    run_dir: Path, instances: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for instance in instances:
        path = run_dir / instance[KEY_INSTANCE_ID] / "summary.json"
        if not path.exists():
            continue
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    summaries.sort(key=lambda item: item["instance_id"])
    return summaries


def write_aggregate(
    run_dir: Path,
    run_id: str,
    summaries: list[dict[str, Any]],
    selected: int,
) -> dict[str, Any]:
    ordered = sorted(summaries, key=lambda item: item["instance_id"])
    resolved = sum(bool(item.get("evaluation", {}).get("resolved")) for item in ordered)
    aggregate = {
        "run_id": run_id,
        "selected": selected,
        "processed": len(ordered),
        "remaining": selected - len(ordered),
        "resolved": resolved,
        "resolve_rate": resolved / len(ordered) if ordered else 0.0,
        "runner_errors": sum(summary_has_runner_error(item) for item in ordered),
        "wall_seconds_sum": sum(float(item.get("wall_seconds", 0)) for item in ordered),
        "tasks": ordered,
    }
    write_json(run_dir / "summary.json", aggregate, Redactor.from_environment())
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="dotenv file containing model settings")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="python")
    parser.add_argument("--repo")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="allow combine_file tasks, whose issue text may omit some mutations",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-on-runner-error",
        action="store_true",
        help="stop pending tasks after an API, container, or grader failure",
    )
    parser.add_argument("--no-evaluate", action="store_false", dest="evaluate")
    parser.add_argument("--f2p-only", action="store_true")
    parser.add_argument("--allow-container-network", action="store_true")
    parser.add_argument("--container-arch", choices=("x86_64", "arm64"), default="x86_64")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--fallback-model")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--request-options-json", default="{}")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--context-window-tokens", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # Load the requested dotenv before resolving environment-backed defaults.
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.env_file:
        env_path = preliminary.env_file.expanduser().resolve()
        if not env_path.is_file():
            parser.error(f"env file does not exist: {env_path}")
        load_dotenv(env_path, override=False)
    else:
        local_env = REPO_ROOT / ".env"
        if local_env.is_file():
            load_dotenv(local_env, override=False)

    args = parser.parse_args(argv)
    args.model = args.model or os.getenv("CPCODEAGENT_MODEL", "qwen3.8-max")
    args.fallback_model = args.fallback_model or os.getenv("CPCODEAGENT_FALLBACK_MODEL") or None
    args.api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    args.base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or None
    args.max_steps = args.max_steps or int_env("CPCODEAGENT_MAX_STEPS", 40)
    args.max_seconds = args.max_seconds or float_env("CPCODEAGENT_MAX_SECONDS", 1_800)
    args.max_tokens = args.max_tokens or int_env("CPCODEAGENT_MAX_TOKENS", 200_000)
    args.context_window_tokens = args.context_window_tokens or int_env(
        "CPCODEAGENT_CONTEXT_WINDOW_TOKENS", 128_000
    )
    if not args.api_key:
        parser.error("OPENAI_API_KEY is missing; set it in --env-file or the environment")
    if args.count < 1 or args.workers < 1 or args.context_window_tokens < 1:
        parser.error("--count, --workers, and --context-window-tokens must be positive")
    try:
        json.loads(args.request_options_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--request-options-json is invalid JSON: {exc}")

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.exists():
        parser.error(f"prepared dataset not found: {dataset_path}; run prepare_swesmith first")
    try:
        docker.from_env().ping()
    except docker.errors.DockerException as exc:
        parser.error(f"Docker daemon is not available: {exc}")

    dataset = load_from_disk(str(dataset_path))
    manifest_path = dataset_path.parent / "dataset_manifest.json"
    dataset_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"path": str(dataset_path), "task_count": len(dataset)}
    )
    instances = select_instances(
        dataset,
        args.count,
        args.seed,
        args.language or None,
        args.repo,
        args.instance_ids,
        args.include_combined,
    )
    if not instances:
        parser.error("No eligible SWE-smith tasks matched the selection")
    expected_count = len(set(args.instance_ids)) if args.instance_ids else args.count
    if len(instances) < expected_count:
        parser.error(
            f"Only {len(instances)} eligible task(s) matched; "
            f"requested {expected_count}. Relax the filters or choose another repo."
        )

    try:
        commit_result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        args.agent_commit = commit_result.stdout.strip() or "unknown"
    except OSError:
        args.agent_commit = "unknown"
    run_dir = args.runs_dir.expanduser().resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "model": args.model,
        "count": len(instances),
        "seed": args.seed,
        "language": args.language,
        "include_combined": args.include_combined,
        "workers": args.workers,
        "context_window_tokens": args.context_window_tokens,
        "stop_on_runner_error": args.stop_on_runner_error,
        "instance_ids": [item[KEY_INSTANCE_ID] for item in instances],
        "dataset": dataset_manifest,
        "agent_commit": args.agent_commit,
    }
    write_json(run_dir / "run_manifest.json", run_manifest, Redactor.from_environment())
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Selected {len(instances)} task(s); seed={args.seed}", flush=True)

    summaries: list[dict[str, Any]] = []
    if args.workers == 1:
        for task_number, instance in enumerate(instances, start=1):
            summary = run_one(
                instance,
                args,
                run_dir,
                dataset_manifest,
                task_number,
                len(instances),
            )
            summaries.append(summary)
            write_aggregate(run_dir, args.run_id, summaries, len(instances))
            if args.stop_on_runner_error and summary_has_runner_error(summary):
                print("Stopping after runner error; completed task data is preserved.", flush=True)
                break
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_one,
                    instance,
                    args,
                    run_dir,
                    dataset_manifest,
                    task_number,
                    len(instances),
                ): instance
                for task_number, instance in enumerate(instances, start=1)
            }
            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                write_aggregate(run_dir, args.run_id, summaries, len(instances))
                if args.stop_on_runner_error and summary_has_runner_error(summary):
                    cancelled = sum(item.cancel() for item in futures if not item.done())
                    print(
                        "Stopping after runner error; "
                        f"cancelled {cancelled} pending task(s). Completed task data is preserved.",
                        flush=True,
                    )
                    break

    # A second worker may finish while cancellation is happening. Rebuild from
    # durable per-task summaries so the run overview remains accurate.
    summaries = load_task_summaries(run_dir, instances)
    aggregate = write_aggregate(run_dir, args.run_id, summaries, len(instances))
    print(f"Resolved {aggregate['resolved']}/{aggregate['processed']} processed tasks")
    return 0 if aggregate["runner_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
