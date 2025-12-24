"""Google Cloud Storage utilities for Cloud Run jobs."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

LOGGER = logging.getLogger(__name__)


def get_gcs_client():
    """Get GCS client."""
    from google.cloud import storage
    return storage.Client()


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse gs://bucket/key into (bucket, key)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def upload_files_to_gcs(
    *,
    output_dir: Path,
    gcs_uri: str,
    path_prefix: str = "",
) -> None:
    """Upload local files to GCS.
    
    Args:
        output_dir: Local directory containing files to upload
        gcs_uri: GCS URI (gs://bucket/prefix)
        path_prefix: Additional prefix to add to GCS keys
    """
    if not gcs_uri:
        LOGGER.info("No GCS URI provided; skipping upload.")
        return

    bucket_name, base_prefix = parse_gcs_uri(gcs_uri)
    
    full_prefix = base_prefix.rstrip("/")
    if path_prefix:
        full_prefix = f"{full_prefix}/{path_prefix.strip('/')}" if full_prefix else path_prefix.strip("/")

    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    base = output_dir.resolve()
    
    files = sorted(p for p in base.rglob("*") if p.is_file())
    if not files:
        LOGGER.info("Nothing to upload from %s", output_dir)
        return

    LOGGER.info("Uploading %d files to gs://%s/%s", len(files), bucket_name, full_prefix)
    
    for local_path in files:
        rel = local_path.relative_to(base).as_posix()
        gcs_key = f"{full_prefix}/{rel}" if full_prefix else rel
        try:
            blob = bucket.blob(gcs_key)
            blob.upload_from_filename(str(local_path))
        except Exception as exc:
            LOGGER.error("Failed to upload %s to gs://%s/%s: %s", local_path, bucket_name, gcs_key, exc)
            raise


def save_dataset_to_gcs(
    dataset,
    gcs_uri: str,
    name: str = "dataset",
) -> str:
    """Save HF dataset to GCS using Arrow format (preserves Image columns).
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict to save
        gcs_uri: Base GCS URI (gs://bucket/prefix)
        name: Name for the dataset folder
        
    Returns:
        GCS URI of the saved dataset
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
    
    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    full_prefix = prefix.rstrip("/")
    
    # Save to local temp directory using Arrow format
    local_dir = Path(f"/tmp/{name}_arrow_temp")
    if local_dir.exists():
        shutil.rmtree(local_dir)
    
    LOGGER.info("Saving dataset to Arrow format...")
    dataset.save_to_disk(str(local_dir))
    
    # Upload entire directory to GCS
    gcs_prefix = f"{full_prefix}/{name}" if full_prefix else name
    upload_files_to_gcs(output_dir=local_dir, gcs_uri=f"gs://{bucket_name}/{gcs_prefix}")
    
    # Cleanup
    shutil.rmtree(local_dir)
    
    result_uri = f"gs://{bucket_name}/{gcs_prefix}"
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


def load_dataset_from_gcs(gcs_uri: str, split: str = "train") -> "Dataset":
    """Load HF dataset directly from GCS (saved with save_to_disk).
    
    Downloads files locally first to avoid gcsfs caching issues.
    
    Args:
        gcs_uri: GCS URI to dataset directory (gs://bucket/path/to/dataset/)
        split: Unused, kept for API compatibility
        
    Returns:
        Loaded Dataset
        
    Requires:
        pip install datasets google-cloud-storage
    """
    from datasets import load_from_disk
    import tempfile
    
    LOGGER.info("Loading dataset from %s", gcs_uri)
    
    # Parse GCS URI
    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    
    # Download to local temp directory (bypasses gcsfs cache)
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    local_dir = tempfile.mkdtemp(prefix="gcs_dataset_")
    
    blobs = list(bucket.list_blobs(prefix=f"{prefix}/"))
    for blob in blobs:
        filename = blob.name.split('/')[-1]
        if filename:  # Skip directory markers
            local_path = f"{local_dir}/{filename}"
            blob.download_to_filename(local_path)
    
    LOGGER.info("Downloaded %d files to %s", len(blobs), local_dir)
    
    # Load from local
    ds = load_from_disk(local_dir)
    
    return ds


__all__ = [
    "save_dataset_to_gcs",
    "load_dataset_from_gcs",
    "parse_gcs_uri",
    "get_gcs_client",
]


