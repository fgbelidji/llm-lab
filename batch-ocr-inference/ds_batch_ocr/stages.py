from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

import shutil
from datasets import Dataset, Features, Sequence, Value, load_dataset, Image as HfImage
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from .config import (
    AssembleSettings,
    DescribeSettings,
    DocumentMetadata,
    ExtractSettings,
    FigureMetadata,
)
from .document import (
    build_document_markdown,
    enrich_markdown_with_captions,
    write_json,
    write_jsonl,
    write_text,
)
from .hf_io import maybe_upload_dataset, resolve_stage_dir

LOGGER = logging.getLogger(__name__)

DATASET_FILENAME = "dataset.jsonl"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []

    def _generator() -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    return _generator()


def write_jsonl_iter(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def _dataset_features() -> Features:
    return Features(
        {
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
        }
    )


def _dataset_path(base_dir: Path) -> Path:
    return base_dir / DATASET_FILENAME



def _build_dataset_records_iter(documents: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for doc in documents:
        document_with_boxes_path = doc.get("document_with_boxes_path") 
        document_with_boxes_relpath = str(document_with_boxes_path) 

        yield {
            "sample_id": str(doc.get("sample_id")),
            "dataset_index": int(doc.get("dataset_index") or 0),
            "source_image_path": str(doc.get("source_image_path") or ""),
            "document_with_boxes_image_path": document_with_boxes_relpath,
            "document_markdown_text": doc.get("document_markdown_text") or "",
            "extracted_figures": doc.get("extracted_figures") or [],
            "extracted_figures_metadata": json.dumps(doc.get("extracted_figures_metadata") or []),
            "document_markdown_path": str(doc.get("document_path") or ""),
            "document_final_markdown_path": str(doc.get("document_final_markdown_path") or ""),
            "document_final_markdown_text": doc.get("document_final_markdown_text") or "",
            "raw_response_path": str(doc.get("raw_response_path") or ""),
        }


def _build_dataset_records(documents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(_build_dataset_records_iter(documents))


def _push_dataset_records(
    records_files: List[str],
    output_dir: Path,
    repo_id: Optional[str],
    commit_message: Optional[str],
    revision: Optional[str],
) -> None:
    if not repo_id:
        return

    dataset = load_dataset("json", data_files=records_files)

    token = os.environ.get("HF_TOKEN", None)
    dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        revision=revision,
        commit_message=commit_message or "Update dataset records",
    )


def _load_dataset_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["figures"] = _figures_from_columnar(record.get("figures"))
            records.append(record)
    return records


def _collate_single_item(batch: List[Any]) -> Any:
    return batch[0]


def run_stage_extract(settings: ExtractSettings) -> None:
    dataset = load_dataset(
        settings.dataset_name,
        settings.dataset_config,
        split=settings.dataset_split,
        streaming=settings.stream_dataset,
    )

    if settings.stream_dataset:
        try:
            num_workers = max(0, int(os.environ.get("EXTRACT_DATALOADER_WORKERS", "2")))
        except ValueError:
            LOGGER.warning("Invalid EXTRACT_DATALOADER_WORKERS value; defaulting to 2")
            num_workers = 2
        try:
            prefetch_factor = max(1, int(os.environ.get("EXTRACT_DATALOADER_PREFETCH", "2")))
        except ValueError:
            LOGGER.warning("Invalid EXTRACT_DATALOADER_PREFETCH value; defaulting to 2")
            prefetch_factor = 2
        dataloader_kwargs: Dict[str, Any] = {
            "batch_size": 1,
            "num_workers": num_workers,
            "collate_fn": _collate_single_item,
        }
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
        sample_iterator = iter(DataLoader(dataset, **dataloader_kwargs))
    else:
        sample_iterator = iter(dataset)

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    documents_batches_dir = settings.output_dir / "document_batches"
    if documents_batches_dir.exists():
        shutil.rmtree(documents_batches_dir)
    documents_batches_dir.mkdir(parents=True, exist_ok=True)

    document_batch_files: List[Path] = []
    batch_index = 0

    document_count = 0
    failures: List[Dict[str, Any]] = []

    chunk_size = max(settings.inference.max_batch_size, 1)

    LOGGER.info(
        "Extract stage | dataset=%s/%s/%s | max_samples=%s | chunk=%s",
        settings.dataset_name,
        settings.dataset_config,
        settings.dataset_split,
        settings.max_samples,
        chunk_size,
    )

    batch_contexts: List[Dict[str, Any]] = []
    batch_requests: List[Dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal batch_contexts, batch_requests, document_count, batch_index
        if not batch_contexts:
            return

        try:
            responses = settings.client.infer(batch_requests)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Batch inference failed for %s samples", len(batch_contexts))
            for ctx in batch_contexts:
                failures.append(
                    {
                        "sample_id": ctx["sample_id"],
                        "dataset_index": ctx["dataset_index"],
                        "error": str(exc),
                        "exception_type": exc.__class__.__name__,
                    }
                )
                image_obj = ctx.get("image")
                if hasattr(image_obj, "close"):
                    image_obj.close()
            batch_contexts = []
            batch_requests = []
            return

        if len(responses) != len(batch_contexts):
            LOGGER.warning(
                "Mismatch between responses (%s) and requests (%s) in extract batch",
                len(responses),
                len(batch_contexts),
            )

        batch_document_dicts: List[Dict[str, Any]] = []

        for idx, ctx in enumerate(batch_contexts):
            image_obj = ctx.get("image")
            try:
                response_text = responses[idx].strip() if idx < len(responses) else ""
                if not response_text:
                    raise RuntimeError("Empty response from DeepSeek inference")
                
                #write raw response markdown to file
                raw_response_path = ctx["sample_dir"] / "raw_response.md"
                write_text(raw_response_path, response_text)

                #build document markdown and extract figures
                markdown, figures, img_draw = build_document_markdown(
                    image=image_obj,
                    response_text=response_text,
                    sample_dir=ctx["sample_dir"],
                    sample_id=ctx["sample_id"],
                )

                #write document markdown to file
                document_path = ctx["sample_dir"] / "document.md"
                write_text(document_path, markdown)

                #write document with boxes image to file
                img_draw.save(ctx["sample_dir"] / "document_with_boxes.png")
                
        
                
                #build document metadata
                batch_document_dicts.append(
                    {
                        "sample_id": str(ctx["sample_id"]),
                        "dataset_index": int(ctx["dataset_index"]),
                        "document_path": str(document_path),
                        "raw_response_path": str(raw_response_path),
                        "source_image_path": str(ctx["sample_dir"] / "source.png"),
                        "document_with_boxes_path": str(ctx["sample_dir"] / "document_with_boxes.png"),
                        "document_markdown_text": str(markdown),
                        "extracted_figures": [str(figure.image_path) for figure in figures],
                        "extracted_figures_metadata": [asdict(figure) for figure in figures],
                    }
                )
                extracted_figures_metadata = [asdict(figure) for figure in figures]
                LOGGER.info(extracted_figures_metadata)

                LOGGER.debug(
                    "Processed sample %s | figures=%s | markdown_chars=%s",
                    ctx["sample_id"],
                    len(figures),
                    len(markdown),
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Failed to finalize sample %s", ctx["sample_id"])
                failures.append(
                    {
                        "sample_id": ctx["sample_id"],
                        "dataset_index": ctx["dataset_index"],
                        "error": str(exc),
                        "exception_type": exc.__class__.__name__,
                    }
                )
            finally:
                if hasattr(image_obj, "close"):
                    image_obj.close()

        if batch_document_dicts:
            batch_file = documents_batches_dir / f"batch_{batch_index:05d}.json"
            write_json(batch_file, batch_document_dicts)
            document_batch_files.append(str(batch_file))
            batch_index += 1
            document_count += len(batch_document_dicts)

        #reset batch contexts and requests
        batch_contexts = []
        batch_requests = []

    for idx, sample in enumerate(sample_iterator):
        if settings.max_samples is not None and idx >= settings.max_samples:
            break

        sample_id = f"sample_{idx:05d}"
        sample_dir = settings.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        raw_image = sample["images"][0]
        image = raw_image.copy()
        if image.mode != "RGB":
            image = image.convert("RGB")

        #write source image to file
        source_image_path = sample_dir / "source.png"
        image.save(source_image_path)

        #copy image for processing
        processing_image = image.copy()
        image.close()

        batch_contexts.append(
            {
                "sample_id": sample_id,
                "dataset_index": idx,
                "sample_dir": sample_dir,
                "image": processing_image,
            }
        )
        batch_requests.append(
            {
                "image": processing_image,
                "prompt": settings.prompt,
                "max_tokens": settings.max_tokens,
                "temperature": settings.temperature,
                "request_timeout": settings.inference.request_timeout,
            }
        )

        if len(batch_requests) >= chunk_size:
            flush_batch()

    #process batch if not empty
    flush_batch()

    manifest = {
        "generated_at": __now_iso(),
        "stage": "extract",
        "dataset": {
            "name": settings.dataset_name,
            "config": settings.dataset_config,
            "split": settings.dataset_split,
        },
        "model": {
            "served_model_name": settings.served_model_name,
            "prompt": settings.prompt,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
        },
        "inference": {
            "max_batch_size": settings.inference.max_batch_size,
            "max_concurrency": settings.inference.max_concurrency,
            "request_timeout": settings.inference.request_timeout,
            "max_retries": settings.inference.max_retries,
            "retry_backoff_seconds": settings.inference.retry_backoff_seconds,
            "max_retry_wait_seconds": settings.inference.max_retry_wait_seconds,
        },
        "documents": [],
        "documents_path": documents_batches_dir.name,
        "documents_batches": document_batch_files,
        "documents_count": document_count,
        "failures": failures,
    }

    write_json(settings.output_dir / "manifest.json", manifest)
    extract_commit = settings.upload_commit_message
    if settings.upload_repo_id and not extract_commit:
        extract_commit = f"Upload extract stage outputs {__now_iso()}"
    # def iter_documents_from_batches(files: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    #     for file_path in files:
    #         try:
    #             batch_data = json.loads(file_path.read_text(encoding="utf-8"))
    #         except Exception as exc:  # pragma: no cover - defensive logging
    #             LOGGER.warning("Failed to read documents batch %s: %s", file_path, exc)
    #             continue
    #         yield from batch_data

    #documents_iter_for_push = iter_documents_from_batches(document_batch_files)
    #dataset_records_iter = _build_dataset_records_iter(documents_iter_for_push)
    LOGGER.info(document_batch_files)
    _push_dataset_records(
        records_files=document_batch_files,
        output_dir=settings.output_dir,
        repo_id=settings.upload_repo_id,
        commit_message=extract_commit,
        revision=settings.upload_revision,
    )
    maybe_upload_dataset(
        output_dir=settings.output_dir,
        repo_id=settings.upload_repo_id,
        path_in_repo=settings.upload_path_in_repo,
        commit_message=extract_commit,
        revision=settings.upload_revision,
    )
    LOGGER.info(
        "Extract stage complete | documents=%s | failures=%s",
        document_count,
        len(failures),
    )


def run_stage_describe(settings: DescribeSettings) -> None:
    stage1_dir = resolve_stage_dir(settings.stage1_dir, settings.source_locator)

    manifest_name = settings.source_locator.manifest_name or "manifest.json"
    manifest_path = stage1_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 1 manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    documents: List[Dict[str, Any]] = []
    batch_rel_paths = manifest.get("documents_batches") or []
    if batch_rel_paths:
        for rel in batch_rel_paths:
            batch_path = stage1_dir / rel
            try:
                batch_data = json.loads(batch_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to load document batch %s: %s", batch_path, exc)
                continue

            if isinstance(batch_data, list):
                documents.extend(batch_data)
            else:
                LOGGER.warning("Unexpected document batch format at %s", batch_path)
    else:
        documents_path_str = manifest.get("documents_path")
        if documents_path_str:
            documents_path = stage1_dir / documents_path_str
            documents = read_jsonl(documents_path)
        else:
            documents = manifest.get("documents", []) or []
    doc_by_sample: Dict[str, Dict[str, Any]] = {doc.get("sample_id", ""): doc for doc in documents}

    dataset_path = _dataset_path(stage1_dir)
    dataset_records = _load_dataset_records(dataset_path)
    rebuilt_records = False
    if not dataset_records:
        LOGGER.info("Dataset records missing at %s; rebuilding from manifest", dataset_path)
        dataset_records = _build_dataset_records(documents)
        rebuilt_records = True

    records_by_sample: Dict[str, Dict[str, Any]] = {
        record.get("sample_id", ""): record for record in dataset_records
    }

    chunk_size = max(settings.inference.max_batch_size, 1)

    pending_total = sum(
        1
        for record in dataset_records
        for fig in record.get("figures", [])
        if not (fig.get("description") or "".strip())
    )
    if pending_total == 0:
        LOGGER.info("No pending figure descriptions; dataset is already up to date.")
        if rebuilt_records:
            describe_commit = settings.upload_commit_message or (
                f"Upload describe stage outputs {__now_iso()}"
            )
            _push_dataset_records(
                records=dataset_records,
                output_dir=stage1_dir,
                repo_id=settings.upload_repo_id,
                commit_message=describe_commit,
                revision=settings.upload_revision,
            )
            maybe_upload_dataset(
                output_dir=stage1_dir,
                repo_id=settings.upload_repo_id,
                path_in_repo=settings.upload_path_in_repo,
                commit_message=describe_commit,
                revision=settings.upload_revision,
            )
        return

    LOGGER.info("Describe stage | pending figures=%s | chunk=%s", pending_total, chunk_size)

    failures: List[Dict[str, Any]] = []
    batch_contexts: List[Dict[str, Any]] = []
    batch_requests: List[Dict[str, Any]] = []

    def enqueue(sample_id: str, figure_index: int, figure: Dict[str, Any]) -> None:
        image_rel_path = figure.get("image_path")
        if not image_rel_path:
            failures.append(
                {
                    "sample_id": sample_id,
                    "figure_id": figure.get("figure_id", ""),
                    "reason": "missing_image_path",
                }
            )
            return

        image_path = stage1_dir / image_rel_path
        if not image_path.exists():
            failures.append(
                {
                    "sample_id": sample_id,
                    "figure_id": figure.get("figure_id", ""),
                    "reason": "missing_image_file",
                    "path": image_rel_path,
                }
            )
            return

        try:
            image = Image.open(image_path)
        except Exception as exc:  # pragma: no cover - defensive
            failures.append(
                {
                    "sample_id": sample_id,
                    "figure_id": figure.get("figure_id", ""),
                    "reason": "image_open_failed",
                    "path": image_rel_path,
                    "error": str(exc),
                }
            )
            return

        batch_contexts.append(
            {
                "sample_id": sample_id,
                "figure_index": figure_index,
                "figure_id": figure.get("figure_id", ""),
                "image": image,
            }
        )
        batch_requests.append(
            {
                "image": image,
                "prompt": settings.prompt,
                "max_tokens": settings.max_tokens,
                "temperature": settings.temperature,
                "request_timeout": settings.inference.request_timeout,
            }
        )

    def flush_batch() -> None:
        nonlocal batch_contexts, batch_requests
        if not batch_contexts:
            return

        try:
            responses = settings.client.infer(batch_requests)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception(
                "Describe batch inference failed for %s figures", len(batch_contexts)
            )
            for ctx in batch_contexts:
                failures.append(
                    {
                        "sample_id": ctx["sample_id"],
                        "figure_id": ctx.get("figure_id", ""),
                        "reason": "inference_error",
                        "error": str(exc),
                    }
                )
        else:
            if len(responses) != len(batch_contexts):
                LOGGER.warning(
                    "Mismatch between responses (%s) and requests (%s) in describe batch",
                    len(responses),
                    len(batch_contexts),
                )

            for idx, ctx in enumerate(batch_contexts):
                try:
                    description = responses[idx].strip() if idx < len(responses) else ""
                    if not description:
                        raise RuntimeError("Empty description generated for figure")

                    record = records_by_sample.get(ctx["sample_id"])
                    if record and ctx["figure_index"] < len(record.get("figures", [])):
                        record["figures"][ctx["figure_index"]]["description"] = description

                    doc_entry = doc_by_sample.get(ctx["sample_id"])
                    if doc_entry and ctx["figure_index"] < len(doc_entry.get("figures", [])):
                        doc_entry["figures"][ctx["figure_index"]]["description"] = description
                except Exception as exc:  # pragma: no cover - defensive logging
                    failures.append(
                        {
                            "sample_id": ctx["sample_id"],
                            "figure_id": ctx.get("figure_id", ""),
                            "reason": "postprocess_error",
                            "error": str(exc),
                        }
                    )
        finally:
            for ctx in batch_contexts:
                image = ctx.get("image")
                if hasattr(image, "close"):
                    image.close()
            batch_contexts = []
            batch_requests = []

    for record in dataset_records:
        sample_id = record.get("sample_id", "")
        for figure_index, figure in enumerate(record.get("figures", [])):
            if figure.get("description"):
                continue
            enqueue(sample_id, figure_index, figure)
            if len(batch_requests) >= chunk_size:
                flush_batch()

    flush_batch()

    describe_commit = settings.upload_commit_message or (
        f"Upload describe stage outputs {__now_iso()}"
    )

    write_json(manifest_path, manifest)
    _push_dataset_records(
        records=dataset_records,
        output_dir=stage1_dir,
        repo_id=settings.upload_repo_id,
        commit_message=describe_commit,
        revision=settings.upload_revision,
    )
    maybe_upload_dataset(
        output_dir=stage1_dir,
        repo_id=settings.upload_repo_id,
        path_in_repo=settings.upload_path_in_repo,
        commit_message=describe_commit,
        revision=settings.upload_revision,
    )

    failure_path = stage1_dir / "describe_failures.jsonl"
    if failures:
        write_jsonl(failure_path, failures)
    elif failure_path.exists():
        failure_path.unlink()

    LOGGER.info(
        "Describe stage complete | figures=%s | failures=%s",
        sum(len(rec.get("figures", [])) for rec in dataset_records),
        len(failures),
    )


def run_stage_assemble(settings: AssembleSettings) -> None:
    stage1_dir = resolve_stage_dir(settings.stage1_dir, settings.stage1_locator)

    dataset_path = _dataset_path(stage1_dir)
    dataset_records = _load_dataset_records(dataset_path)
    if not dataset_records:
        raise FileNotFoundError(
            f"Dataset records not found at {dataset_path}. Run extract stage first."
        )

    failures: List[Dict[str, Any]] = []
    final_documents: List[Dict[str, Any]] = []

    for record in dataset_records:
        sample_id = record.get("sample_id", "")
        sample_dir = stage1_dir / sample_id
        doc_rel_path = record.get("document_markdown_path", "")
        stage1_doc_path = stage1_dir / doc_rel_path

        if not stage1_doc_path.exists():
            LOGGER.warning("Document markdown missing: %s", stage1_doc_path)
            failures.append(
                {
                    "sample_id": sample_id,
                    "dataset_index": record.get("dataset_index"),
                    "missing_path": stage1_doc_path.as_posix(),
                    "reason": "document_missing",
                }
            )
            continue

        markdown = stage1_doc_path.read_text(encoding="utf-8")
        description_map = {
            fig.get("figure_id", ""): fig for fig in record.get("figures", [])
        }
        enriched_markdown = enrich_markdown_with_captions(markdown, description_map)

        final_doc_path = sample_dir / "document_final.md"
        write_text(final_doc_path, enriched_markdown)

        record["document_final_markdown_path"] = (
            Path(sample_id) / "document_final.md"
        ).as_posix()
        record["document_final_markdown_text"] = enriched_markdown

        copied_figures = [
            {
                "figure_id": fig.get("figure_id", ""),
                "image_path": fig.get("image_path", ""),
                "description": fig.get("description", ""),
            }
            for fig in record.get("figures", [])
        ]

        final_documents.append(
            {
                "sample_id": sample_id,
                "dataset_index": record.get("dataset_index"),
                "final_document_path": record["document_final_markdown_path"],
                "figures": copied_figures,
            }
        )

    aggregate = {
        "generated_at": __now_iso(),
        "stage": "assemble",
        "documents": final_documents,
        "failures": failures,
    }
    write_json(stage1_dir / "assemble_summary.json", aggregate)

    assemble_commit = settings.dataset_commit_message or (
        f"Upload assemble stage outputs {__now_iso()}"
    )

    _push_dataset_records(
        records=dataset_records,
        output_dir=stage1_dir,
        repo_id=settings.dataset_repo_id,
        commit_message=assemble_commit,
        revision=settings.dataset_branch,
    )
    maybe_upload_dataset(
        output_dir=stage1_dir,
        repo_id=settings.dataset_repo_id,
        path_in_repo=settings.dataset_path_in_repo,
        commit_message=assemble_commit,
        revision=settings.dataset_branch,
    )
    LOGGER.info(
        "Assemble stage complete | documents=%s | failures=%s",
        len(final_documents),
        len(failures),
    )


def _load_figure_descriptions(stage2_dir: Path) -> Dict[str, Dict[str, Any]]:
    aggregate_path = stage2_dir / "figure_descriptions.json"
    descriptions: Dict[str, Dict[str, Any]] = {}
    if aggregate_path.exists():
        data = json.loads(aggregate_path.read_text(encoding="utf-8"))
        for entry in data.get("figures", []):
            descriptions[entry["figure_id"]] = entry
        return descriptions

    for json_file in stage2_dir.glob("*.json"):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        for entry in data.get("figures", []):
            descriptions[entry["figure_id"]] = entry
    return descriptions


def dataclass_to_dict(document: DocumentMetadata) -> Dict[str, Any]:
    result = {
        "sample_id": document.sample_id,
        "dataset_index": document.dataset_index,
        "document_path": document.document_path,
        "raw_response_path": document.raw_response_path,
        "source_image_path": document.source_image_path,
        "document_with_boxes_path": document.document_with_boxes_path,
        "document_markdown_text": document.document_markdown_text,
        "document_final_markdown_path": document.document_final_markdown_path or "",
        "document_final_markdown_text": document.document_final_markdown_text or "",
        "extracted_figures": [
            {
                "figure_id": figure.figure_id,
                "label": figure.label,
                "image_path": figure.image_path,
                "document_relative_path": figure.document_relative_path,
                "bounding_box_pixels": figure.bounding_box_pixels,
                "description": figure.description or "",
            }
            for figure in document.extracted_figures
        ],
    }
    return result


def __now_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat() + "Z"

__all__ = [
    "run_stage_extract",
    "run_stage_describe",
    "run_stage_assemble",
]

