"""Amazon S3 utilities for SageMaker jobs."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import boto3
from botocore.config import Config

if TYPE_CHECKING:
    from datasets import Dataset

LOGGER = logging.getLogger(__name__)


def get_s3_client():
    """Get S3 client with retry configuration."""
    config = Config(
        retries={"max_attempts": 3, "mode": "standard"},
        max_pool_connections=50,
    )
    return boto3.client("s3", config=config)


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def upload_files_to_s3(
    *,
    output_dir: Path,
    s3_uri: str,
    path_prefix: str = "",
) -> None:
    """Upload local files to S3.
    
    Args:
        output_dir: Local directory containing files to upload
        s3_uri: S3 URI (s3://bucket/prefix)
        path_prefix: Additional prefix to add to S3 keys
    """
    if not s3_uri:
        LOGGER.info("No S3 URI provided; skipping upload.")
        return

    bucket, base_prefix = parse_s3_uri(s3_uri)
    
    full_prefix = base_prefix.rstrip("/")
    if path_prefix:
        full_prefix = f"{full_prefix}/{path_prefix.strip('/')}" if full_prefix else path_prefix.strip("/")

    s3 = get_s3_client()
    base = output_dir.resolve()
    
    files = sorted(p for p in base.rglob("*") if p.is_file())
    if not files:
        LOGGER.info("Nothing to upload from %s", output_dir)
        return

    LOGGER.info("Uploading %d files to s3://%s/%s", len(files), bucket, full_prefix)
    
    for local_path in files:
        rel = local_path.relative_to(base).as_posix()
        s3_key = f"{full_prefix}/{rel}" if full_prefix else rel
        try:
            s3.upload_file(str(local_path), bucket, s3_key)
        except Exception as exc:
            LOGGER.error("Failed to upload %s to s3://%s/%s: %s", local_path, bucket, s3_key, exc)
            raise


def save_dataset_to_s3(
    dataset,
    s3_uri: str,
    name: str = "dataset",
) -> str:
    """Save HF dataset to S3 using Arrow format (preserves Image columns).
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict to save
        s3_uri: Base S3 URI (s3://bucket/prefix)
        name: Name for the dataset folder
        
    Returns:
        S3 URI of the saved dataset
    """
    from datasets import DatasetDict
    
    # Handle DatasetDict by extracting the first split
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            LOGGER.info("Using split '%s' from DatasetDict", split_name)
    
    bucket, prefix = parse_s3_uri(s3_uri)
    full_prefix = prefix.rstrip("/")
    
    # Save to local temp directory using Arrow format
    local_dir = Path(f"/tmp/{name}_arrow_temp")
    if local_dir.exists():
        shutil.rmtree(local_dir)
    
    LOGGER.info("Saving dataset to Arrow format...")
    dataset.save_to_disk(str(local_dir))
    
    # Upload entire directory to S3
    s3_prefix = f"{full_prefix}/{name}" if full_prefix else name
    upload_files_to_s3(output_dir=local_dir, s3_uri=f"s3://{bucket}/{s3_prefix}")
    
    # Cleanup
    shutil.rmtree(local_dir)
    
    result_uri = f"s3://{bucket}/{s3_prefix}"
    LOGGER.info("Saved dataset to %s", result_uri)
    return result_uri


def get_dataset_features():
    """Get the dataset feature schema."""
    from datasets import Features, Sequence, Value, Image as HfImage
    
    return Features({
        "sample_id": Value("string"),
        "dataset_index": Value("int64"),
        "source_image": HfImage(),
        "document_with_boxes_image": HfImage(),
        "document_markdown": Value("string"),
        "extracted_figures": Sequence(HfImage()),
        "extracted_figures_metadata": Sequence(Value("string")),
        "document_final_markdown": Value("string"),
    })


def load_dataset_from_s3(s3_uri: str, split: str = "train") -> "Dataset":
    """Load HF dataset directly from S3 (saved with save_to_disk).
    
    Downloads files locally first to avoid s3fs caching issues.
    
    Args:
        s3_uri: S3 URI to dataset directory (s3://bucket/path/to/dataset/)
        split: Unused, kept for API compatibility
        
    Returns:
        Loaded Dataset
        
    Requires:
        pip install datasets boto3
    """
    from datasets import load_from_disk
    import tempfile
    
    LOGGER.info("Loading dataset from %s", s3_uri)
    
    # Parse S3 URI
    bucket_name, prefix = parse_s3_uri(s3_uri)
    
    # Download to local temp directory (bypasses s3fs cache)
    s3 = get_s3_client()
    local_dir = tempfile.mkdtemp(prefix="s3_dataset_")
    
    # List and download all objects
    paginator = s3.get_paginator('list_objects_v2')
    download_count = 0
    for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{prefix}/"):
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = key.split('/')[-1]
            if filename:  # Skip directory markers
                local_path = f"{local_dir}/{filename}"
                s3.download_file(bucket_name, key, local_path)
                download_count += 1
    
    LOGGER.info("Downloaded %d files to %s", download_count, local_dir)
    
    # Load from local
    ds = load_from_disk(local_dir)
    
    return ds


__all__ = [
    "save_dataset_to_s3",
    "load_dataset_from_s3",
    "parse_s3_uri",
    "get_s3_client",
]
