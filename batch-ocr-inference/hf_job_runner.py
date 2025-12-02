# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub[hf_transfer,hf_xet]",
#     "numpy",
#     "datasets",
#     "pillow",
#     "requests",
#     "openai",
#     "torch",
# ]
# ///

"""
Minimal entrypoint for Hugging Face Jobs.

It downloads the job code repository (containing the `ds_batch_ocr` package)
using `huggingface_hub.snapshot_download` and then delegates to
`ds_batch_ocr.cli.main`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def ensure_code_checkout() -> Path:
    repo_id = os.environ.get("JOB_CODE_REPO")
    if not repo_id:
        raise RuntimeError("JOB_CODE_REPO environment variable must be set.")

    repo_type = os.environ.get("JOB_CODE_REPO_TYPE", "dataset")
    revision = os.environ.get("JOB_CODE_REVISION")
    local_dir = Path(os.environ.get("JOB_CODE_LOCAL_DIR", "/tmp/deepseek-ocr-job-code"))
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return local_dir


def main() -> None:
    code_dir = ensure_code_checkout()
    sys.path.insert(0, str(code_dir))

    from ds_batch_ocr.cli import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()

