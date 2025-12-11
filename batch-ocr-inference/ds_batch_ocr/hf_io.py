"""Hugging Face Hub upload utilities."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import CommitOperationAdd, HfApi, create_repo

LOGGER = logging.getLogger(__name__)

DEFAULT_CHUNK_MAX_FILES = 200
DEFAULT_CHUNK_MAX_BYTES = 512 * 1024 * 1024


def _gather_files(output_dir: Path, path_in_repo: str) -> List[Tuple[Path, str, int]]:
    """Collect all files from output_dir with their repo paths and sizes."""
    base = output_dir.resolve()
    prefix = path_in_repo.strip("/")
    entries = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rel = p.relative_to(base).as_posix()
            entries.append((p, f"{prefix}/{rel}" if prefix else rel, p.stat().st_size))
    return entries


def _make_batches(
    files: List[Tuple[Path, str, int]],
    max_files: int,
    max_bytes: int,
) -> List[List[Tuple[Path, str, int]]]:
    """Split files into batches respecting max_files and max_bytes limits."""
    if not files:
        return []

    batches: List[List[Tuple[Path, str, int]]] = []
    current: List[Tuple[Path, str, int]] = []
    current_bytes = 0

    for entry in files:
        current.append(entry)
        current_bytes += max(entry[2], 0)
        if len(current) >= max_files or current_bytes >= max_bytes:
            batches.append(current)
            current = []
            current_bytes = 0

    if current:
        batches.append(current)

    return batches


def maybe_upload_dataset(
    *,
    output_dir: Path,
    repo_id: Optional[str],
    path_in_repo: str,
    commit_message: Optional[str],
    revision: Optional[str],
) -> None:
    """Upload local files to a HuggingFace dataset repository."""
    if not repo_id:
        LOGGER.info("No dataset repo provided; skipping upload.")
        return

    commit_message = commit_message or (
        "Add assembled DeepSeek OCR dataset " + datetime.utcnow().isoformat() + "Z"
    )

    token = os.environ.get("HF_TOKEN") or None  # Treat empty string as None
    api = HfApi(token=token)

    max_files = int(os.environ.get("HF_UPLOAD_CHUNK_MAX_FILES", DEFAULT_CHUNK_MAX_FILES))
    max_bytes = int(os.environ.get("HF_UPLOAD_CHUNK_MAX_BYTES", DEFAULT_CHUNK_MAX_BYTES))

    files = _gather_files(output_dir, path_in_repo or "")
    if not files:
        LOGGER.info("Nothing to upload from %s", output_dir)
        return

    batches = _make_batches(files, max_files=max_files, max_bytes=max_bytes)
    total_batches = len(batches) or 1
    LOGGER.info("Uploading %d files to %s in %d commit(s)", len(files), repo_id, total_batches)

    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)

    for index, batch in enumerate(batches, start=1):
        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path)
            for local_path, repo_path, _ in batch
        ]
        message = commit_message
        if total_batches > 1:
            message = f"{commit_message} (batch {index}/{total_batches})"

        LOGGER.info("Commit %d/%d | files=%d", index, total_batches, len(batch))
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            operations=operations,
            commit_message=message,
        )


__all__ = ["maybe_upload_dataset"]
