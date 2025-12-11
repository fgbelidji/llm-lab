"""Pipeline stages: extract, describe, assemble."""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from datasets import Features, Sequence, Value, load_dataset, Image as HfImage
from PIL import Image
from torch.utils.data import DataLoader

from .config import AssembleSettings, DescribeSettings, ExtractSettings, env
from .document import build_document_markdown, enrich_markdown_with_captions, write_json, write_text
from .hf_io import maybe_upload_dataset

LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _dataset_features() -> Features:
    return Features({
        "sample_id": Value("string"),
        "dataset_index": Value("int64"),
        "source_image_path": HfImage(),
        "document_with_boxes_image_path": HfImage(),
        "document_markdown_text": Value("string"),
        "extracted_figures": Sequence(HfImage()),
        "extracted_figures_metadata": Sequence(Value("string")),
        "document_markdown_path": Value("string"),
        "document_final_markdown_path": Value("string"),
        "document_final_markdown_text": Value("string"),
        "raw_response_path": Value("string"),
    })


def run_stage_extract(settings: ExtractSettings) -> None:
    """Run OCR extraction on dataset samples."""
    dataset = load_dataset(
        settings.dataset_name,
        settings.dataset_config,
        split=settings.dataset_split,
        streaming=settings.stream_dataset,
    )

    # Setup iterator with optional DataLoader for streaming
    if settings.stream_dataset:
        num_workers = env("DATALOADER_WORKERS", 2, int)
        prefetch = env("DATALOADER_PREFETCH", 2, int)
        kwargs = {"batch_size": 1, "num_workers": num_workers, "collate_fn": lambda b: b[0]}
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch
        sample_iter = iter(DataLoader(dataset, **kwargs))
    else:
        sample_iter = iter(dataset)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = settings.output_dir / "document_batches"
    if batches_dir.exists():
        shutil.rmtree(batches_dir)
    batches_dir.mkdir(parents=True, exist_ok=True)

    batch_files: List[str] = []
    batch_idx = 0
    doc_count = 0
    failures: List[Dict[str, Any]] = []
    chunk_size = settings.inference.batch_size

    # Path prefix for dataset-relative paths (e.g., "outputs/extract")
    path_prefix = settings.hub.path_in_repo.strip("/") if settings.hub.path_in_repo else ""

    LOGGER.info("Extract | dataset=%s/%s/%s | max_samples=%s | batch=%s | path_prefix=%s",
                settings.dataset_name, settings.dataset_config, settings.dataset_split,
                settings.max_samples, chunk_size, path_prefix or "(root)")

    contexts: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []

    def make_dataset_path(sample_id: str, filename: str) -> str:
        """Create path relative to dataset root."""
        if path_prefix:
            return f"{path_prefix}/{sample_id}/{filename}"
        return f"{sample_id}/{filename}"

    def flush():
        nonlocal contexts, requests, doc_count, batch_idx
        if not contexts:
            return

        try:
            responses = settings.client.infer(requests)
        except Exception as exc:
            LOGGER.exception("Batch inference failed for %d samples", len(contexts))
            for ctx in contexts:
                failures.append({"sample_id": ctx["sample_id"], "error": str(exc)})
                if hasattr(ctx.get("image"), "close"):
                    ctx["image"].close()
            contexts, requests = [], []
            return

        docs: List[Dict[str, Any]] = []
        for i, ctx in enumerate(contexts):
            img = ctx.get("image")
            try:
                text = responses[i].strip() if i < len(responses) else ""
                if not text:
                    raise RuntimeError("Empty response")

                sample_dir = ctx["sample_dir"]
                sample_id = ctx["sample_id"]
                write_text(sample_dir / "raw_response.md", text)

                markdown, figures, img_draw = build_document_markdown(
                    image=img, response_text=text, sample_dir=sample_dir, 
                    sample_id=sample_id, path_prefix=path_prefix
                )
                write_text(sample_dir / "document.md", markdown)
                img_draw.save(sample_dir / "document_with_boxes.png")

                docs.append({
                    "sample_id": sample_id,
                    "dataset_index": ctx["dataset_index"],
                    "source_image_path": str(sample_dir / "source.png"),
                    "document_with_boxes_image_path": str(sample_dir / "document_with_boxes.png"),
                    "document_markdown_text": markdown,
                    "extracted_figures": [str(f.image_path) for f in figures],
                    "extracted_figures_metadata": [json.dumps(asdict(f)) for f in figures],
                    "document_markdown_path": make_dataset_path(sample_id, "document.md"),
                    "document_final_markdown_path": "",
                    "document_final_markdown_text": "",
                    "raw_response_path": make_dataset_path(sample_id, "raw_response.md"),
                })
            except Exception as exc:
                LOGGER.exception("Failed sample %s", ctx["sample_id"])
                failures.append({"sample_id": ctx["sample_id"], "error": str(exc)})
            finally:
                if hasattr(img, "close"):
                    img.close()

        if docs:
            batch_file = batches_dir / f"batch_{batch_idx:05d}.json"
            write_json(batch_file, docs)
            batch_files.append(str(batch_file))
            batch_idx += 1
            doc_count += len(docs)

        contexts, requests = [], []

    for idx, sample in enumerate(sample_iter):
        if settings.max_samples and idx >= settings.max_samples:
            break

        sample_id = f"sample_{idx:05d}"
        sample_dir = settings.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        img = sample["images"][0].copy()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(sample_dir / "source.png")

        contexts.append({"sample_id": sample_id, "dataset_index": idx, "sample_dir": sample_dir, "image": img.copy()})
        requests.append({
            "image": contexts[-1]["image"],
            "prompt": settings.prompt,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "request_timeout": settings.inference.request_timeout,
        })
        img.close()

        if len(requests) >= chunk_size:
            flush()

    flush()

    # Save manifest
    write_json(settings.output_dir / "manifest.json", {
        "generated_at": _now_iso(),
        "stage": "extract",
        "documents_count": doc_count,
        "failures": failures,
    })

    # Load as HF dataset and push
    ds = load_dataset("json", data_files=batch_files, features=_dataset_features())
    shutil.rmtree(batches_dir)

    token = env("HF_TOKEN")
    commit_msg = settings.hub.commit_message or f"Extract stage {_now_iso()}"

    if settings.hub.repo_id:
        maybe_upload_dataset(
            output_dir=settings.output_dir,
            repo_id=settings.hub.repo_id,
            path_in_repo=settings.hub.path_in_repo,
            commit_message=commit_msg,
            revision=settings.hub.branch,
        )
        ds.push_to_hub(settings.hub.repo_id, token=token, revision=settings.hub.branch, commit_message=commit_msg)

    LOGGER.info("Extract complete | docs=%d | failures=%d", doc_count, len(failures))


def run_stage_describe(settings: DescribeSettings) -> None:
    """Describe figures in the dataset that lack descriptions."""
    repo_id = settings.source_repo_id or settings.hub.repo_id
    if not repo_id:
        raise ValueError("No source repo_id for describe stage")

    token = env("HF_TOKEN")
    LOGGER.info("Loading dataset from %s", repo_id)
    dataset = load_dataset(repo_id, split="train", token=token)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    desc_dir = settings.output_dir / "descriptions"
    if desc_dir.exists():
        shutil.rmtree(desc_dir)
    desc_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = settings.inference.batch_size
    failures: List[Dict[str, Any]] = []
    contexts: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    batch_idx = 0
    described = 0

    def flush():
        nonlocal contexts, requests, batch_idx, described
        if not contexts:
            return

        results = []
        try:
            responses = settings.client.infer(requests)
            for i, ctx in enumerate(contexts):
                desc = responses[i].strip() if i < len(responses) else ""
                if desc:
                    results.append({"figure_id": ctx["figure_id"], "description": desc})
                    described += 1
        except Exception as exc:
            LOGGER.exception("Describe batch failed")
            for ctx in contexts:
                failures.append({"figure_id": ctx.get("figure_id"), "error": str(exc)})
        finally:
            for ctx in contexts:
                if hasattr(ctx.get("image"), "close"):
                    ctx["image"].close()
            contexts, requests = [], []

        if results:
            with (desc_dir / f"batch_{batch_idx:05d}.jsonl").open("w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            batch_idx += 1

    # Queue figures needing descriptions
    pending = 0
    for row in dataset:
        sample_id = row["sample_id"]
        metas = row.get("extracted_figures_metadata") or []
        images = row.get("extracted_figures") or []

        for i, meta_json in enumerate(metas):
            meta = json.loads(meta_json) if isinstance(meta_json, str) else meta_json
            if meta.get("description"):
                continue

            pending += 1
            fig_id = meta.get("figure_id", "")

            if i >= len(images) or images[i] is None:
                failures.append({"sample_id": sample_id, "figure_id": fig_id, "reason": "missing_image"})
                continue

            fig_img = images[i]
            if not isinstance(fig_img, Image.Image):
                try:
                    fig_img = Image.open(fig_img["path"]) if isinstance(fig_img, dict) else fig_img
                except Exception:
                    failures.append({"sample_id": sample_id, "figure_id": fig_id, "reason": "open_failed"})
                    continue

            contexts.append({"sample_id": sample_id, "figure_id": fig_id, "image": fig_img})
            requests.append({
                "image": fig_img,
                "prompt": settings.prompt,
                "max_tokens": settings.max_tokens,
                "temperature": settings.temperature,
                "request_timeout": settings.inference.request_timeout,
            })

            if len(requests) >= chunk_size:
                flush()

    flush()

    if pending == 0:
        LOGGER.info("No figures need descriptions")
        return

    # Load descriptions and apply to dataset
    lookup = {}
    for f in sorted(desc_dir.glob("batch_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                lookup[r["figure_id"]] = r["description"]

    if not lookup:
        LOGGER.info("No descriptions generated")
        return

    def apply(row):
        metas = row.get("extracted_figures_metadata") or []
        new_metas = []
        for m in metas:
            meta = json.loads(m) if isinstance(m, str) else m
            if meta.get("figure_id") in lookup:
                meta["description"] = lookup[meta["figure_id"]]
            new_metas.append(json.dumps(meta))
        row["extracted_figures_metadata"] = new_metas
        return row

    updated = dataset.map(apply, features=dataset.features)
    target = settings.hub.repo_id or repo_id
    commit_msg = settings.hub.commit_message or f"Describe stage {_now_iso()}"

    updated.push_to_hub(target, token=token, revision=settings.hub.branch, commit_message=commit_msg)
    shutil.rmtree(desc_dir)

    LOGGER.info("Describe complete | described=%d | failures=%d", described, len(failures))


def run_stage_assemble(settings: AssembleSettings) -> None:
    """Enrich markdown with figure descriptions."""
    repo_id = settings.source_repo_id or settings.hub.repo_id
    if not repo_id:
        raise ValueError("No source repo_id for assemble stage")

    token = env("HF_TOKEN")
    LOGGER.info("Loading dataset from %s", repo_id)
    dataset = load_dataset(repo_id, split="train", token=token)

    def assemble(row):
        markdown = row.get("document_markdown_text") or ""
        if not markdown:
            return row

        desc_map = {}
        for m in row.get("extracted_figures_metadata") or []:
            meta = json.loads(m) if isinstance(m, str) else m
            if meta.get("figure_id"):
                desc_map[meta["figure_id"]] = meta

        row["document_final_markdown_text"] = enrich_markdown_with_captions(markdown, desc_map)
        
        # Derive path prefix from existing document_markdown_path
        # e.g., "outputs/extract/sample_00000/document.md" -> "outputs/extract/sample_00000"
        existing_path = row.get("document_markdown_path") or ""
        if "/" in existing_path:
            path_base = existing_path.rsplit("/", 1)[0]  # Remove filename
            row["document_final_markdown_path"] = f"{path_base}/document_final.md"
        else:
            row["document_final_markdown_path"] = f"{row['sample_id']}/document_final.md"
        return row

    dataset = dataset.map(assemble)
    target = settings.hub.repo_id or repo_id
    commit_msg = settings.hub.commit_message or f"Assemble stage {_now_iso()}"

    dataset.push_to_hub(target, token=token, revision=settings.hub.branch, commit_message=commit_msg)
    LOGGER.info("Assemble complete")


__all__ = ["run_stage_extract", "run_stage_describe", "run_stage_assemble"]
