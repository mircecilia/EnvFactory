"""Fixed CPU-prep Git base for the two-cycle V2 GPU experiment."""
from __future__ import annotations

import subprocess
from pathlib import Path

GPU_EXPERIMENT_BASE_SHA = "582108da1f0a192c6ba049decdb2a9250f742be5"
ROOT = Path(__file__).resolve().parents[3]


def checked_base_sha() -> str:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", GPU_EXPERIMENT_BASE_SHA, "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("GPU experiment CPU-prep base SHA is not an ancestor of HEAD")
    return GPU_EXPERIMENT_BASE_SHA
