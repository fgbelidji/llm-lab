#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-cloud-storage",
#     "gcsfs",
#     "torch",
#     "datasets>=4.0.0",
#     "pyarrow>=12.0.0",
#     "numpy",
#     "pillow",
#     "requests",
#     "openai",
#     "huggingface-hub[hf_transfer]",
#     "rich",
# ]
# ///
"""
Cloud Run GPU job entry point for the DeepSeek OCR pipeline.

This script is the entrypoint for Cloud Run jobs.
It starts the vLLM server and runs the OCR pipeline.

Environment Variables:
    PIPELINE_STAGE: Stage to run (extract, describe, assemble)
    GCS_OUTPUT_URI: GCS URI for output (gs://bucket/prefix)
    GCS_INPUT_URI: GCS URI for input data (for describe/assemble stages)
    
    # Model/inference settings
    MODEL_NAME: Model name for vLLM
    
    # Dataset settings
    DATASET_NAME: Source dataset name
    DATASET_CONFIG: Dataset config name
    NUM_SAMPLES: Number of samples to process
    
    # HuggingFace settings (optional)
    HF_TOKEN: HuggingFace token
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_logging():
    """Configure logging."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def write_success_marker():
    """Write a success marker file."""
    marker_path = Path("/tmp/pipeline_success")
    marker_path.write_text("Pipeline completed successfully\n")
    logging.info("Wrote success marker to %s", marker_path)


def main() -> None:
    """Main entry point for Cloud Run job."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    stage = os.environ.get("PIPELINE_STAGE", "extract").lower()
    
    logger.info("Starting Cloud Run OCR pipeline job")
    logger.info("Python version: %s", sys.version)
    logger.info("Working directory: %s", os.getcwd())
    logger.info("Pipeline stage: %s", stage)
    
    # Log environment
    logger.info("Environment:")
    logger.info("  GCS_OUTPUT_URI: %s", os.environ.get("GCS_OUTPUT_URI", "not set"))
    logger.info("  GCS_INPUT_URI: %s", os.environ.get("GCS_INPUT_URI", "not set"))
    logger.info("  MODEL_NAME: %s", os.environ.get("MODEL_NAME", "not set"))
    logger.info("  DATASET_NAME: %s", os.environ.get("DATASET_NAME", "not set"))
    logger.info("  NUM_SAMPLES: %s", os.environ.get("NUM_SAMPLES", "not set"))
    
    # Import and run pipeline
    try:
        from llm_ocr.cli import main as pipeline_main
        pipeline_main()
        write_success_marker()
        logger.info("Pipeline completed successfully")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        # Write failure info for debugging
        failure_file = Path("/tmp/pipeline_failure")
        failure_file.write_text(str(exc))
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
