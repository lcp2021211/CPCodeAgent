"""Download and pin the SWE-smith task dataset for offline experiment runs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "swesmith_train"
DATASET_ID = "SWE-bench/SWE-smith"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", help="Hugging Face dataset revision; defaults to HEAD")
    args = parser.parse_args()
    target = args.output.expanduser().resolve()
    manifest_path = target.parent / "dataset_manifest.json"

    if target.exists():
        dataset = load_from_disk(str(target))
        print(f"Dataset already prepared: {target} ({len(dataset)} tasks)")
        if manifest_path.exists():
            print(manifest_path.read_text(encoding="utf-8"))
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    revision = args.revision or HfApi().dataset_info(DATASET_ID).sha
    dataset = load_dataset(DATASET_ID, split="train", revision=revision)

    temporary = Path(tempfile.mkdtemp(prefix="swesmith_train.", dir=target.parent))
    try:
        dataset.save_to_disk(str(temporary))
        os.replace(temporary, target)
    finally:
        # save_to_disk should leave no sibling data outside the temporary directory.
        if temporary.exists() and not any(temporary.iterdir()):
            temporary.rmdir()

    manifest = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "split": "train",
        "task_count": len(dataset),
        "fingerprint": dataset._fingerprint,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "path": str(target),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(dataset)} tasks at {target}")
    print(f"Pinned revision: {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
