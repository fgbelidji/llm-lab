from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    # Prefer rich for readable, colored log output when available.
    from rich.console import Console
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _RICH_AVAILABLE = False

from .config import ArtifactLocator, AssembleSettings, DescribeSettings, ExtractSettings, InferenceSettings
from .server import (
    DeepSeekClient,
    base_url_from_env,
    launch_vllm,
    should_launch_server,
    shutdown_server,
    wait_for_server,
)
from .stages import (
    run_stage_assemble,
    run_stage_describe,
    run_stage_extract,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek OCR HF Jobs pipeline")
    parser.add_argument("--stage", choices=["extract", "describe", "assemble"], help="Pipeline stage to run")
    parser.add_argument("--output-dir", help="Output directory for the current stage")
    parser.add_argument("--stage1-dir", help="Path to stage1 outputs (for describe/assemble)")
    parser.add_argument("--stage2-dir", help="Path to stage2 outputs (for assemble)")
    parser.add_argument("--dataset-name", help="Dataset name for extract stage")
    parser.add_argument("--dataset-config", help="Dataset config for extract stage")
    parser.add_argument("--dataset-split", help="Dataset split for extract stage")
    parser.add_argument("--max-samples", type=int, help="Max samples to process in extract stage")
    parser.add_argument("--doc-prompt", help="Prompt for document extraction stage")
    parser.add_argument("--figure-prompt", help="Prompt for figure description stage")
    parser.add_argument("--doc-max-tokens", type=int, help="Max tokens for extraction stage")
    parser.add_argument("--figure-max-tokens", type=int, help="Max tokens for description stage")
    parser.add_argument("--doc-temperature", type=float, help="Sampling temperature for extraction stage")
    parser.add_argument("--figure-temperature", type=float, help="Sampling temperature for description stage")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable dataset streaming in extract stage",
    )
    parser.add_argument("--dataset-repo-id", help="Hugging Face dataset repo to upload assembled outputs")
    parser.add_argument("--dataset-path-in-repo", help="Target path inside the dataset repo")
    parser.add_argument("--dataset-branch", help="Dataset repo branch or revision to push to")
    parser.add_argument("--dataset-commit-message", help="Commit message for dataset upload")
    return parser.parse_args(argv)


def getenv_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid float for %s=%s. Using default=%s", name, value, default)
        return default


def getenv_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("Invalid int for %s=%s. Using default=%s", name, value, default)
        return default


def _token_margin_for_stage(stage: str, default: int = 512) -> int:
    stage_key = f"{stage.upper()}_TOKEN_MARGIN"
    value = os.environ.get(stage_key) or os.environ.get("PIPELINE_TOKEN_MARGIN")
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed < 0:
            raise ValueError
        return parsed
    except ValueError:
        LOGGER.warning(
            "Invalid token margin for %s=%s. Using default=%s",
            stage_key,
            value,
            default,
        )
        return default


def safe_max_tokens(desired: int, stage: str) -> int:
    max_context = getenv_int("MAX_MODEL_LEN", 4096)
    margin = _token_margin_for_stage(stage)
    allowed = max(1, max_context - margin)
    clamped = min(desired, allowed)
    if clamped < desired:
        LOGGER.info(
            "Clamping %s max tokens from %s to %s to respect context window (MAX_MODEL_LEN=%s, margin=%s)",
            stage,
            desired,
            clamped,
            max_context,
            margin,
        )
    return clamped


def main(argv: Optional[Sequence[str]] = None) -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    if _RICH_AVAILABLE:
        console = Console(force_terminal=os.environ.get("FORCE_COLOR", "").lower() in {"1", "true"})
        handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            rich_tracebacks=True,
        )
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%m/%d/%y %H:%M:%S]",
            handlers=[handler],
            force=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            force=True,
        )
    args = parse_arguments(argv)

    stage = (args.stage or os.environ.get("PIPELINE_STAGE", "extract")).lower()
    if stage not in {"extract", "describe", "assemble"}:
        raise ValueError(f"Unsupported stage: {stage}")

    served_model_name = os.environ.get("SERVED_MODEL_NAME", "deepseek-ocr")
    base_url = base_url_from_env()

    launch_server = should_launch_server() and stage in {"extract", "describe"}
    server_process: Optional["subprocess.Popen"] = None

    try:
        if launch_server:
            server_process = launch_vllm()

        if stage in {"extract", "describe"}:
            health_url = os.environ.get("HEALTH_URL", f"{base_url}/health")
            LOGGER.info("Waiting for server at %s", health_url)
            if not wait_for_server(health_url):
                raise RuntimeError("vLLM server did not become ready in time")

        if stage == "extract":
            dataset_name = args.dataset_name or os.environ.get(
                "DATASET_NAME", "HuggingFaceM4/FineVision"
            )
            dataset_config = args.dataset_config or os.environ.get(
                "DATASET_CONFIG", "olmOCR-mix-0225-documents"
            )
            dataset_split = args.dataset_split or os.environ.get(
                "DATASET_SPLIT", "train"
            )
            max_samples = args.max_samples
            if max_samples is None:
                max_samples = getenv_int("MAX_SAMPLES", 3)

            doc_prompt = args.doc_prompt or os.environ.get(
                "DOC_PROMPT",
                "<image>\n<|grounding|>Convert this document to Markdown.",
            )
            output_dir = Path(
                args.output_dir
                or os.environ.get("STAGE1_OUTPUT_DIR")
                or os.environ.get("OUTPUT_DIR", "./outputs/stage1")
            )
            doc_max_tokens_requested = args.doc_max_tokens or getenv_int("DOC_MAX_TOKENS", 2048)
            doc_max_tokens = safe_max_tokens(doc_max_tokens_requested, stage="extract")
            doc_temperature = (
                args.doc_temperature
                if args.doc_temperature is not None
                else getenv_float("DOC_TEMPERATURE", 0.0)
            )

            extract_inference = InferenceSettings.from_env("extract")

            client = DeepSeekClient(
                base_url=base_url,
                model_name=served_model_name,
                max_tokens=doc_max_tokens,
                temperature=doc_temperature,
                request_timeout=extract_inference.request_timeout,
                max_retries=extract_inference.max_retries,
                retry_backoff_seconds=extract_inference.retry_backoff_seconds,
                max_retry_wait_seconds=extract_inference.max_retry_wait_seconds,
            )

            stage1_upload_repo = os.environ.get("STAGE1_UPLOAD_REPO") or os.environ.get("STAGE1_REPO_ID")

            stage1_upload_path = (
                os.environ.get("STAGE1_UPLOAD_PATH_IN_REPO")
                or os.environ.get("STAGE1_PATH_IN_REPO")
                or ""
            )
            stage1_upload_commit = os.environ.get("STAGE1_UPLOAD_COMMIT_MESSAGE")
            stage1_upload_branch = (
                os.environ.get("STAGE1_UPLOAD_BRANCH")
                or os.environ.get("STAGE1_REPO_REVISION")
            )

            settings = ExtractSettings(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                dataset_split=dataset_split,
                max_samples=max_samples,
                prompt=doc_prompt,
                max_tokens=doc_max_tokens,
                temperature=doc_temperature,
                output_dir=output_dir,
                stream_dataset=not args.no_streaming,
                served_model_name=served_model_name,
                inference=extract_inference,
                client=client,
                upload_repo_id=stage1_upload_repo,
                upload_path_in_repo=stage1_upload_path,
                upload_commit_message=stage1_upload_commit,
                upload_revision=stage1_upload_branch,
            )
            run_stage_extract(settings)

        elif stage == "describe":
            stage1_dir = Path(
                args.stage1_dir
                or os.environ.get("STAGE1_DIR")
                or os.environ.get("STAGE1_OUTPUT_DIR", "./outputs/stage1")
            )
            output_dir = Path(
                args.output_dir
                or os.environ.get("STAGE2_OUTPUT_DIR")
                or os.environ.get("OUTPUT_DIR", "./outputs/stage2")
            )
            figure_prompt = args.figure_prompt or os.environ.get(
                "FIGURE_PROMPT",
                "<image>\nDescribe this image in detail",
            )
            figure_max_tokens_requested = (
                args.figure_max_tokens or getenv_int("FIGURE_MAX_TOKENS", 512)
            )
            figure_max_tokens = safe_max_tokens(
                figure_max_tokens_requested, stage="describe"
            )
            figure_temperature = (
                args.figure_temperature
                if args.figure_temperature is not None
                else getenv_float("FIGURE_TEMPERATURE", 0.0)
            )

            describe_inference = InferenceSettings.from_env("describe")

            client = DeepSeekClient(
                base_url=base_url,
                model_name=served_model_name,
                max_tokens=figure_max_tokens,
                temperature=figure_temperature,
                request_timeout=describe_inference.request_timeout,
                max_retries=describe_inference.max_retries,
                retry_backoff_seconds=describe_inference.retry_backoff_seconds,
                max_retry_wait_seconds=describe_inference.max_retry_wait_seconds,
            )

            stage1_locator = ArtifactLocator.from_env("stage1", manifest_name="manifest.json")

            stage2_upload_repo = os.environ.get("STAGE2_UPLOAD_REPO") or os.environ.get("STAGE2_REPO_ID")

            stage2_upload_path = (
                os.environ.get("STAGE2_UPLOAD_PATH_IN_REPO")
                or os.environ.get("STAGE2_PATH_IN_REPO")
                or ""
            )
            stage2_upload_commit = os.environ.get("STAGE2_UPLOAD_COMMIT_MESSAGE")
            stage2_upload_branch = (
                os.environ.get("STAGE2_UPLOAD_BRANCH")
                or os.environ.get("STAGE2_REPO_REVISION")
            )

            settings = DescribeSettings(
                stage1_dir=stage1_dir,
                output_dir=output_dir,
                prompt=figure_prompt,
                max_tokens=figure_max_tokens,
                temperature=figure_temperature,
                client=client,
                inference=describe_inference,
                source_locator=stage1_locator,
                upload_repo_id=stage2_upload_repo,
                upload_path_in_repo=stage2_upload_path,
                upload_commit_message=stage2_upload_commit,
                upload_revision=stage2_upload_branch,
            )
            run_stage_describe(settings)

        elif stage == "assemble":
            stage1_dir = Path(
                args.stage1_dir
                or os.environ.get("STAGE1_DIR")
                or os.environ.get("STAGE1_OUTPUT_DIR", "./outputs/stage1")
            )
            stage2_dir = Path(
                args.stage2_dir
                or os.environ.get("STAGE2_DIR")
                or os.environ.get("STAGE2_OUTPUT_DIR", "./outputs/stage2")
            )
            output_dir = Path(
                args.output_dir
                or os.environ.get("STAGE3_OUTPUT_DIR")
                or os.environ.get("OUTPUT_DIR", "./outputs/stage3")
            )

            dataset_repo_id = args.dataset_repo_id or os.environ.get("ASSEMBLED_DATASET_REPO")
            if dataset_repo_id:
                dataset_repo_id = dataset_repo_id.strip() or None

            dataset_path_in_repo = (
                args.dataset_path_in_repo
                or os.environ.get("ASSEMBLED_DATASET_PATH_IN_REPO")
                or "data"
            )
            dataset_commit_message = (
                args.dataset_commit_message
                or os.environ.get("ASSEMBLED_DATASET_COMMIT_MESSAGE")
            )
            dataset_branch = args.dataset_branch or os.environ.get("ASSEMBLED_DATASET_BRANCH")
            stage1_locator = ArtifactLocator.from_env("stage1", manifest_name="manifest.json")
            stage2_locator = ArtifactLocator.from_env(
                "stage2", manifest_name="figure_descriptions.json"
            )

            settings = AssembleSettings(
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                output_dir=output_dir,
                dataset_repo_id=dataset_repo_id,
                dataset_path_in_repo=dataset_path_in_repo,
                dataset_commit_message=dataset_commit_message,
                dataset_branch=dataset_branch,
                stage1_locator=stage1_locator,
                stage2_locator=stage2_locator,
            )
            run_stage_assemble(settings)

    finally:
        if server_process is not None:
            shutdown_server(server_process)


__all__ = ["main", "parse_arguments", "getenv_float", "getenv_int"]


