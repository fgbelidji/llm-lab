# Batch Document OCR Pipeline

A scalable 3-stage OCR pipeline using **DeepSeek-OCR** for document processing. Supports multiple cloud platforms: **Hugging Face Jobs**, **AWS SageMaker**, and **Google Cloud Run GPU**.

## Overview

This pipeline processes document images through three stages:

1. **Extract** – Run DeepSeek OCR to convert documents to Markdown and detect/crop figures
2. **Describe** – Generate captions for extracted figures using vision-language inference
3. **Assemble** – Enrich the Markdown with figure captions to produce the final document

All stages use a shared HuggingFace Dataset format, enabling seamless handoff between stages and platforms.

## Architecture

### Stage 1: Extract

![Extract Stage](assets/extract.png)

### Stage 2: Describe

![Describe Stage](assets/describe.png)

### Stage 3: Assemble

![Assemble Stage](assets/assemble.png)

## Project Structure

```
batch-ocr-inference/
├── llm_ocr/                    # Core pipeline library
│   ├── config.py               # Configuration dataclasses
│   ├── stages.py               # Extract, Describe, Assemble implementations
│   ├── document.py             # Markdown parsing, figure extraction, rendering
│   ├── server.py               # vLLM server management & DeepSeek client
│   ├── storage.py              # Abstract storage (HF Hub, S3, GCS)
│   ├── sm_io.py                # S3 dataset I/O for SageMaker
│   ├── gcr_io.py               # GCS dataset I/O for Cloud Run
│   └── cli.py                  # Command-line interface
│
├── hf-jobs/                    # Hugging Face Jobs deployment
│   ├── hf-jobs-pipeline.ipynb  # Interactive notebook
│   └── hf_job_runner.py        # Job entrypoint script
│
├── sagemaker/                  # AWS SageMaker deployment
│   ├── sm-jobs-pipeline.ipynb  # Interactive notebook
│   ├── sm_job_runner.py        # Job entrypoint script
│   └── entry.sh                # Container entrypoint
│
└── google-cloud-run/           # Google Cloud Run GPU deployment
    ├── gcr-jobs-pipeline.ipynb # Interactive notebook
    ├── gcr_job_runner.py       # Job entrypoint script
    └── Dockerfile.cloudrun     # Container definition
```

## Quick Start

### Option 1: Hugging Face Jobs

The simplest way to run the pipeline with GPU access:

```bash
cd hf-jobs/
jupyter notebook hf-jobs-pipeline.ipynb
```

Requires:
- Hugging Face account with Jobs access
- `HF_TOKEN` environment variable set

### Option 2: AWS SageMaker

Run on SageMaker Training Jobs with vLLM DLC:

```bash
cd sagemaker/
jupyter notebook sm-jobs-pipeline.ipynb
```

Requires:
- AWS credentials configured
- SageMaker execution role with S3 access
- `HF_TOKEN` for accessing source datasets

### Option 3: Google Cloud Run GPU

Run on Cloud Run with L4 GPUs:

```bash
cd google-cloud-run/
jupyter notebook gcr-jobs-pipeline.ipynb
```

Requires:
- GCP credentials configured
- Cloud Run GPU quota in your project
- Container built and pushed to Artifact Registry

## Configuration

All stages are configured via environment variables:

### Common Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `deepseek-ai/DeepSeek-OCR` | Model to use for inference |
| `HF_TOKEN` | - | HuggingFace token for private repos |
| `PIPELINE_STAGE` | - | Stage to run: `extract`, `describe`, `assemble` |

### Extract Stage

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_NAME` | `HuggingFaceM4/FineVision` | Source dataset name |
| `DATASET_CONFIG` | `olmOCR-mix-0225-documents` | Dataset configuration |
| `DATASET_SPLIT` | `train` | Dataset split |
| `MAX_SAMPLES` | - | Limit number of samples (for testing) |
| `DOC_PROMPT` | `<image>\n<\|grounding\|>Convert...` | OCR prompt |
| `EXTRACT_BATCH_SIZE` | `4` | Batch size for inference |
| `EXTRACT_MAX_CONCURRENCY` | `4` | Max concurrent requests |

### Describe Stage

| Variable | Default | Description |
|----------|---------|-------------|
| `FIGURE_PROMPT` | `<image>\nDescribe this image...` | Figure description prompt |
| `FIGURE_MAX_TOKENS` | `512` | Max tokens for descriptions |
| `DESCRIBE_BATCH_SIZE` | `4` | Batch size for inference |

### Output Settings

| Variable | Description |
|----------|-------------|
| `HF_REPO_ID` | HuggingFace repo to push results |
| `S3_OUTPUT_URI` | S3 path for SageMaker output |
| `GCS_OUTPUT_URI` | GCS path for Cloud Run output |

## Dataset Schema

The pipeline produces a HuggingFace Dataset with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | `string` | Unique identifier |
| `source_image` | `Image` | Original document image |
| `document_markdown` | `string` | Extracted Markdown (with `figure:` URIs) |
| `extracted_figures` | `Sequence[Image]` | Cropped figure images |
| `extracted_figures_metadata` | `Sequence[string]` | JSON metadata with descriptions |
| `document_final_markdown` | `string` | Final Markdown with captions |

## Rendering Results

To display the final markdown with embedded images in a Jupyter notebook:

```python
from llm_ocr.document import display_markdown

# Load your dataset
ds = load_dataset("your-repo/ocr-results", split="train")

# Display a sample with rendered images
display_markdown(ds[0])
```

## Development

### Running Locally

```bash
# Install dependencies
pip install vllm datasets pillow huggingface_hub

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-OCR \
    --port 8000

# Run a stage
PIPELINE_STAGE=extract \
DATASET_NAME=HuggingFaceM4/FineVision \
MAX_SAMPLES=5 \
python -m llm_ocr.cli
```

### Using the CLI

```bash
# Extract stage
python -m llm_ocr.cli extract --max-samples 10

# Describe stage  
python -m llm_ocr.cli describe --source-repo user/dataset

# Assemble stage
python -m llm_ocr.cli assemble --source-repo user/dataset
```

