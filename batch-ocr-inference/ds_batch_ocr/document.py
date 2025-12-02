from __future__ import annotations

import ast
import base64
import json
import re
import logging
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import FigureMetadata

GROUNDING_PATTERN = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>",
    re.DOTALL,
)

FIGURE_MARKDOWN_PATTERN = re.compile(
    r"!\[Figure (?P<figure_id>[^\]]+)\]\((?P<path>[^)]+)\)"
)


def encode_image(image: "Image.Image") -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_grounding_blocks(text: str) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for match in GROUNDING_PATTERN.finditer(text):
        label = match.group(1).strip()
        coords_text = match.group(2).strip()
        coordinates = None
        if coords_text:
            try:
                coordinates = ast.literal_eval(coords_text)
            except Exception:
                coordinates = None
        matches.append(
            {
                "label": label,
                "coordinates": coordinates,
                "raw": match.group(0),
                "span": match.span(),
            }
        )
    return matches


def postprocess_markdown(text: str) -> str:
    cleaned = (
        text.replace("\\coloneqq", ":=")
        .replace("\\eqqcolon", "=:")
        .replace("<|image_pad|>", "")
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def apply_replacements(text: str, replacements: List[Tuple[int, int, str]]) -> str:
    if not replacements:
        return postprocess_markdown(text)
    sorted_replacements = sorted(replacements, key=lambda item: item[0])
    segments: List[str] = []
    cursor = 0
    for start, end, replacement in sorted_replacements:
        segments.append(text[cursor:start])
        segments.append(replacement)
        cursor = end
    segments.append(text[cursor:])
    return postprocess_markdown("".join(segments))


def save_figure(
    image: "Image.Image",
    sample_dir: Path,
    sample_id: str,
    figure_index: int,
    pixel_box: List[int],
    label: str,
) -> Optional[FigureMetadata]:

    x1, y1, x2, y2 = pixel_box
    crop = image.crop((x1, y1, x2, y2)).copy()

    figures_dir = sample_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_id = f"{sample_id}_fig{figure_index:02d}"
    figure_filename = f"{figure_id}.png"
    figure_relative_doc_path = Path("figures") / figure_filename
    full_path = figures_dir / figure_filename
    crop.save(full_path)

    bounding_box_pixels = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    return FigureMetadata(
        figure_id=figure_id,
        label=label,
        image_path=str(Path(sample_id) / figure_relative_doc_path),
        document_relative_path=str(figure_relative_doc_path),
        bounding_box_pixels=bounding_box_pixels,
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def build_document_markdown(
    image: "Image.Image",
    response_text: str,
    sample_dir: Path,
    sample_id: str,
) -> Tuple[str, List[FigureMetadata], "Image.Image"]:
    blocks = extract_grounding_blocks(response_text)
    replacements: List[Tuple[int, int, str]] = []
    figures: List[FigureMetadata] = []
    figure_index = 1
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()

    for block in blocks:
        label = block["label"].lower()
        start, end = block["span"]
        
        color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
        color_a = color + (20, )
        
        width, height = image.size
        
        pixel_box = block["coordinates"][0]
        
        x1, y1, x2, y2 = pixel_box
        x1 = int(x1 / 999 * width)
        y1 = int(y1 / 999 * height)
        
        x2 = int(x2 / 999 * width)
        y2 = int(y2 / 999 * height)
        pixel_box = (x1, y1, x2, y2)
        
        if label == "image":
            logging.info(f"Image: {pixel_box}")
   
            figure_metadata = save_figure(
                image=image,
                sample_dir=sample_dir,
                sample_id=sample_id,
                figure_index=figure_index,
                pixel_box=pixel_box,
                label=block["label"],
            )
            if figure_metadata:
                figures.append(figure_metadata)
                replacements.append(
                    (
                        start,
                        end,
                        f"![Figure {figure_metadata.figure_id}]({figure_metadata.document_relative_path})",
                    )
                )
                figure_index += 1
            else:
                replacements.append((start, end, ""))
        else:
            replacements.append((start, end, ""))
        
        if label == "title":
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
        
        text_x = x1
        text_y = max(0, y1 - 15)
        
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                        fill=(255, 255, 255, 30))
        draw.text((text_x, text_y), label, font=font, fill=color)
        
    img_draw.paste(overlay, (0, 0), overlay)

    markdown = apply_replacements(response_text, replacements)
    return markdown, figures, img_draw


def enrich_markdown_with_captions(
    markdown: str,
    description_map: Dict[str, Dict[str, Any]],
) -> str:
    used: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        figure_id = match.group("figure_id").strip()
        path = match.group("path").strip()
        entry = description_map.get(figure_id)
        if not entry:
            return match.group(0)

        description = entry.get("description", "").strip()
        if not description:
            return match.group(0)

        alt_text = f"Figure {figure_id}: {description}"
        rendered = f"![{alt_text}]({path})"
        if figure_id not in used:
            rendered += f"\n\n*Figure {figure_id}: {description}*\n"
            used.add(figure_id)
        return rendered

    return FIGURE_MARKDOWN_PATTERN.sub(replace, markdown)


__all__ = [
    "encode_image",
    "extract_grounding_blocks",
    "flatten_boxes",
    "merge_boxes",
    "normalized_to_pixels",
    "postprocess_markdown",
    "apply_replacements",
    "save_figure",
    "write_text",
    "write_json",
    "write_jsonl",
    "build_document_markdown",
    "enrich_markdown_with_captions",
    "FigureMetadata",
    "GROUNDING_PATTERN",
    "FIGURE_MARKDOWN_PATTERN",
]


