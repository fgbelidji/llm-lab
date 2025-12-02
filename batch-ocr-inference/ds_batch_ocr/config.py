from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)

SUPPORTED_ARTIFACT_STRATEGIES = {"local", "hf-hub"}


@dataclass
class FigureMetadata:
    figure_id: str
    label: str
    image_path: str
    document_relative_path: str
    bounding_box_pixels: Dict[str, int]
    description: Optional[str] = None


@dataclass
class DocumentMetadata:
    sample_id: str
    dataset_index: int
    document_path: str
    raw_response_path: str
    source_image_path: str
    document_with_boxes_path: str
    document_markdown_text: str
    document_final_markdown_path: Optional[str] = None
    document_final_markdown_text: Optional[str] = None
    extracted_figures: List[str] = field(default_factory=list)
    extracted_figures_metadata: List[FigureMetadata] = field(default_factory=list)


@dataclass
class InferenceSettings:
    max_batch_size: int = 4
    max_concurrency: int = 4
    request_timeout: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    max_retry_wait_seconds: float = 60.0

    @classmethod
    def from_env(cls, stage: str) -> "InferenceSettings":
        stage = stage.upper()
        default = cls()

        def read_int(*keys: str, default_value: int) -> int:
            return _read_env(*keys, default=default_value, cast=int)

        def read_float(*keys: str, default_value: float) -> float:
            return _read_env(*keys, default=default_value, cast=float)

        return cls(
            max_batch_size=max(
                1,
                read_int(f"{stage}_BATCH_SIZE", "PIPELINE_BATCH_SIZE", default_value=default.max_batch_size),
            ),
            max_concurrency=max(
                1,
                read_int(
                    f"{stage}_MAX_CONCURRENCY",
                    "PIPELINE_MAX_CONCURRENCY",
                    default_value=default.max_concurrency,
                ),
            ),
            request_timeout=max(
                1,
                read_int(
                    f"{stage}_REQUEST_TIMEOUT",
                    "PIPELINE_REQUEST_TIMEOUT",
                    default_value=default.request_timeout,
                ),
            ),
            max_retries=max(
                0,
                read_int(
                    f"{stage}_MAX_RETRIES",
                    "PIPELINE_MAX_RETRIES",
                    default_value=default.max_retries,
                ),
            ),
            retry_backoff_seconds=max(
                0.0,
                read_float(
                    f"{stage}_RETRY_BACKOFF_SECONDS",
                    "PIPELINE_RETRY_BACKOFF_SECONDS",
                    default_value=default.retry_backoff_seconds,
                ),
            ),
            max_retry_wait_seconds=max(
                1.0,
                read_float(
                    f"{stage}_MAX_RETRY_WAIT_SECONDS",
                    "PIPELINE_MAX_RETRY_WAIT_SECONDS",
                    default_value=default.max_retry_wait_seconds,
                ),
            ),
        )


@dataclass
class ArtifactLocator:
    strategy: str = "local"
    repo_id: Optional[str] = None
    job_id: Optional[str] = None
    job_owner: Optional[str] = None
    uri: Optional[str] = None
    manifest_name: str = "manifest.json"

    @classmethod
    def from_env(cls, stage: str, *, manifest_name: str) -> "ArtifactLocator":
        stage = stage.upper()

        env = os.environ

        repo_id = (env.get(f"{stage}_JOB_REPO") or "").strip() or (env.get(f"{stage}_REPO_ID") or "").strip() or None
        job_id = (env.get(f"{stage}_JOB_ID") or "").strip() or None
        job_owner = (env.get(f"{stage}_JOB_OWNER") or "").strip() or None
        uri = (env.get(f"{stage}_ARTIFACT_URI") or "").strip() or None
        manifest_override = (env.get(f"{stage}_MANIFEST_NAME") or "").strip() or None
        explicit_strategy = (env.get(f"{stage}_ARTIFACT_STRATEGY") or "").strip() or None
        pipeline_strategy = (env.get("PIPELINE_ARTIFACT_STRATEGY") or "").strip() or None

        requested_strategy = (explicit_strategy or pipeline_strategy or "").lower()

        if requested_strategy and requested_strategy not in SUPPORTED_ARTIFACT_STRATEGIES:
            raise ValueError(
                f"Unsupported artifact strategy '{requested_strategy}'. "
                "This build only supports HF Jobs via 'hf-hub' or local artifacts."
            )

        if requested_strategy:
            strategy = requested_strategy
        elif repo_id or (job_id and job_owner) or uri:
            strategy = "hf-hub"
        else:
            strategy = "local"

        locator = cls(
            strategy=strategy,
            repo_id=repo_id,
            job_id=job_id,
            job_owner=job_owner,
            uri=uri,
            manifest_name=manifest_override or manifest_name,
        )

        LOGGER.debug(
            "Artifact locator for %s: %s",
            stage,
            {
                "strategy": locator.strategy,
                "repo_id": locator.repo_id,
                "job_id": locator.job_id,
                "job_owner": locator.job_owner,
                "uri": locator.uri,
                "manifest": locator.manifest_name,
            },
        )
        return locator


@dataclass
class ExtractSettings:
    dataset_name: str
    dataset_config: str
    dataset_split: str
    max_samples: Optional[int]
    prompt: str
    max_tokens: int
    temperature: float
    output_dir: Path
    stream_dataset: bool
    served_model_name: str
    client: "DeepSeekClient"
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    upload_repo_id: Optional[str] = None
    upload_path_in_repo: str = ""
    upload_commit_message: Optional[str] = None
    upload_revision: Optional[str] = None


@dataclass
class DescribeSettings:
    stage1_dir: Path
    output_dir: Path
    prompt: str
    max_tokens: int
    temperature: float
    client: "DeepSeekClient"
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    source_locator: ArtifactLocator = field(default_factory=ArtifactLocator)
    upload_repo_id: Optional[str] = None
    upload_path_in_repo: str = ""
    upload_commit_message: Optional[str] = None
    upload_revision: Optional[str] = None


@dataclass
class AssembleSettings:
    stage1_dir: Path
    stage2_dir: Path
    output_dir: Path
    dataset_repo_id: Optional[str]
    dataset_path_in_repo: str
    dataset_commit_message: Optional[str]
    dataset_branch: Optional[str]
    stage1_locator: ArtifactLocator = field(default_factory=ArtifactLocator)
    stage2_locator: ArtifactLocator = field(default_factory=ArtifactLocator)


__all__ = [
    "FigureMetadata",
    "DocumentMetadata",
    "InferenceSettings",
    "ArtifactLocator",
    "ExtractSettings",
    "DescribeSettings",
    "AssembleSettings",
    "SUPPORTED_ARTIFACT_STRATEGIES",
]


def _read_env(*keys: str, default, cast):
    for key in keys:
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return cast(raw)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid value for %s=%s; using default=%s", key, raw, default)
    return default


