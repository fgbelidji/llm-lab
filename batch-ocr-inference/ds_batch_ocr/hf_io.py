from __future__ import annotations

import logging
import os
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from huggingface_hub import CommitOperationAdd, HfApi, create_repo, snapshot_download
from .config import ArtifactLocator

LOGGER = logging.getLogger(__name__)

DEFAULT_CHUNK_MAX_FILES = 200
DEFAULT_CHUNK_MAX_BYTES = 512 * 1024 * 1024


def _gather_files(output_dir: Path, path_in_repo: str) -> List[Tuple[Path, str, int]]:
    base = output_dir.resolve()
    entries: List[Tuple[Path, str, int]] = []
    prefix = path_in_repo.strip("/")
    for local_path in sorted(base.rglob("*")):
        if not local_path.is_file():
            continue
        rel_path = local_path.relative_to(base).as_posix()
        repo_path = f"{prefix}/{rel_path}" if prefix else rel_path
        try:
            size = local_path.stat().st_size
        except OSError:
            size = 0
        entries.append((local_path, repo_path, size))
    return entries


def _make_batches(
    files: List[Tuple[Path, str, int]],
    max_files: int,
    max_bytes: int,
) -> List[List[Tuple[Path, str, int]]]:
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


def unpack_archives(target_dir: Path) -> None:
    for archive in list(target_dir.glob("**/*.tar.gz")):
        LOGGER.info("Extracting archive %s", archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(archive.parent)
        archive.unlink()


def download_job_artifact(repo_id: str, target_dir: Path) -> None:
    LOGGER.info("Downloading job artifact %s -> %s", repo_id, target_dir)
    actual_repo_id = repo_id

    if repo_id.startswith("jobs/"):
        parts = repo_id.split("/", 2)
        if len(parts) == 3:
            actual_repo_id = f"{parts[1]}/{parts[2]}"
        else:
            LOGGER.warning("Unexpected jobs repo format: %s", repo_id)
    elif repo_id.startswith("datasets/"):
        actual_repo_id = repo_id.split("/", 1)[1]
    elif repo_id.startswith("models/"):
        actual_repo_id = repo_id.split("/", 1)[1]

    snapshot_download(
        repo_id=actual_repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=("logs/**",),
    )
    unpack_archives(target_dir)


def resolve_stage_dir(base_dir: Path, locator: ArtifactLocator) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    def locate_manifest(candidate: Path) -> Optional[Path]:
        manifest_name = locator.manifest_name or "manifest.json"
        manifest_path = candidate / manifest_name
        return manifest_path if manifest_path.exists() else None

    manifest_path = locate_manifest(base_dir)
    if manifest_path:
        locator.manifest_name = manifest_path.name
        return manifest_path.parent

    strategy = (locator.strategy or "local").lower()
    if strategy == "local":
        LOGGER.debug("Using local artifact locator for %s", base_dir)
    else:
        handler = _ARTIFACT_HANDLERS.get(strategy)
        if handler is None:
            raise ValueError(
                f"Unsupported artifact locator strategy '{strategy}' in HF Jobs mode."
            )
        handler(locator, base_dir)

    manifest_path = locate_manifest(base_dir)
    if manifest_path:
        locator.manifest_name = manifest_path.name
        return manifest_path.parent

    outputs_dir = base_dir / "outputs"
    outputs_manifest = locate_manifest(outputs_dir)
    if outputs_manifest:
        locator.manifest_name = outputs_manifest.name
        return outputs_manifest.parent

    return base_dir


def _handle_hf_hub(locator: ArtifactLocator, base_dir: Path) -> None:
    repo_id = locator.repo_id or locator.uri
    if repo_id:
        download_job_artifact(repo_id, base_dir)
        return
    if locator.job_id and locator.job_owner:
        download_job_artifact(f"jobs/{locator.job_owner}/{locator.job_id}", base_dir)
        return
    LOGGER.debug("HF locator missing repo/job information; treating as local artifacts.")


_ARTIFACT_HANDLERS: Dict[str, Callable[[ArtifactLocator, Path], None]] = {
    "hf-hub": _handle_hf_hub,
    "huggingface": _handle_hf_hub,
    "hub": _handle_hf_hub,
}


def maybe_upload_dataset(
    *,
    output_dir: Path,
    repo_id: Optional[str],
    path_in_repo: str,
    commit_message: Optional[str],
    revision: Optional[str],
) -> None:
    if not repo_id:
        LOGGER.info("No dataset repo provided; skipping upload.")
        return

    commit_message = commit_message or (
        "Add assembled DeepSeek OCR dataset " + datetime.utcnow().isoformat() + "Z"
    )

    token = os.environ.get("HF_TOKEN", None)
    api = HfApi(token=token)

    max_files = int(os.environ.get("HF_UPLOAD_CHUNK_MAX_FILES", DEFAULT_CHUNK_MAX_FILES))
    max_bytes = int(os.environ.get("HF_UPLOAD_CHUNK_MAX_BYTES", DEFAULT_CHUNK_MAX_BYTES))

    files = _gather_files(output_dir, path_in_repo or "")
    if not files:
        LOGGER.info("Nothing to upload from %s", output_dir)
        return

    batches = _make_batches(files, max_files=max_files, max_bytes=max_bytes)
    total_batches = len(batches) or 1
    LOGGER.info(
        "Uploading %s files to %s in %s commit(s)",
        len(files),
        repo_id,
        total_batches,
    )

    LOGGER.info("Ensuring dataset repo exists: repo_id=%s", repo_id)
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=token,
    )

    for index, batch in enumerate(batches, start=1):
        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path)
            for local_path, repo_path, _ in batch
        ]
        message = commit_message
        if total_batches > 1:
            message = f"{commit_message} (batch {index}/{total_batches})"

        LOGGER.info(
            "Commit %s/%s | files=%s | path_in_repo=%s",
            index,
            total_batches,
            len(batch),
            path_in_repo or ".",
        )
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            operations=operations,
            commit_message=message,
        )

__all__ = [
    "unpack_archives",
    "download_job_artifact",
    "resolve_stage_dir",
    "maybe_upload_dataset",
]


