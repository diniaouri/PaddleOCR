import re
from pathlib import Path

import fitz  # pymupdf
from paddleocr import PaddleOCRVL

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIRS = {
    "pdfs": "/home/dimitra/Documents/Github/PaddleOCR/my_project/pdf_orig",
    "pngs": "/home/dimitra/Documents/Github/PaddleOCR/my_project/png_orig",
}
OUTPUT_BASE = "/home/dimitra/Documents/Github/PaddleOCR/my_project/output_vl"

SAVE_MARKDOWN        = True
SAVE_PLAINTEXT       = True
SAVE_SEARCHABLE_PDF  = True
SAVE_JSON            = False

PIPELINE_OPTIONS = dict(
    pipeline_version             = "v1.5",
    use_doc_orientation_classify = True,
    use_doc_unwarping            = True,
    use_layout_detection         = True,
    use_seal_recognition         = False,
    use_chart_recognition        = False,
    merge_layout_blocks          = True,
    format_block_content         = True,
    engine                       = "transformers",
)

PREDICT_OPTIONS = dict(
    max_new_tokens     = 4096,
    repetition_penalty = 1.05,
)
# ───────────────────────────────────────────────────────────────────────────────


def extract_markdown_text(res) -> str:
    """Safely extract markdown string from a result object."""
    md = getattr(res, "markdown", None) or res.get("markdown", None)
    if md is None:
        return ""
    if isinstance(md, str):
        return md
    if isinstance(md, dict):
        for key in ("markdown_texts", "markdown", "text", "content"):
            if key in md and isinstance(md[key], str):
                return md[key]
        return "\n\n".join(v for v in md.values() if isinstance(v, str))
    return str(md)


def markdown_to_plaintext(md_text: str) -> str:
    """Strip markdown syntax → clean plain text."""
    text = re.sub(r"<[^>]+>", "", md_text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", text)
    text = re.sub(r"^\|[-| :]+\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|", "  ", text)
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_text_blocks(res) -> list[dict]:
    """
    Parse parsing_res_list entries which look like:
        #################
        label:   header
        bbox:    [530, 128, 726, 150]
        content: THE CRISIS.
        #################
    Returns list of {bbox: [x1,y1,x2,y2], text: str}
    """
    blocks = []
    parsing_res_list = res.get("parsing_res_list", [])

    for entry in parsing_res_list:
        if not isinstance(entry, str):
            continue

        bbox_match    = re.search(r"bbox:\s*\[([^\]]+)\]", entry)
        content_match = re.search(r"content:\t(.+?)(?=\n#{5,}|\Z)", entry, re.DOTALL)

        if not bbox_match or not content_match:
            continue

        try:
            bbox = [float(x.strip()) for x in bbox_match.group(1).split(",")]
        except ValueError:
            continue

        text = content_match.group(1).strip()
        if not text:
            continue

        blocks.append({"bbox": bbox, "text": text})

    return blocks


def _overlay_text_on_page(out_page, blocks, page_w, page_h, img_w, img_h):
    """
    Place invisible selectable text onto a PDF page.
    Uses insert_text (not insert_textbox) so text is never silently dropped.
    For multi-line blocks, splits lines and places each one individually
    spaced evenly within the bounding box.
    """
    for block in blocks:
        bbox = block["bbox"]
        text = block["text"]

        # Scale OCR image coords → PDF point coords
        x1 = bbox[0] * page_w / img_w
        y1 = bbox[1] * page_h / img_h
        x2 = bbox[2] * page_w / img_w
        y2 = bbox[3] * page_h / img_h

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue

        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            continue

        n_lines  = len(lines)
        fontsize = max(4.0, min(12.0, box_h / n_lines * 0.85))
        line_h   = box_h / n_lines

        for j, line in enumerate(lines):
            # Baseline = top of box + (j+1) * line_height - small descender offset
            baseline_y = y1 + (j + 1) * line_h - line_h * 0.15
            out_page.insert_text(
                fitz.Point(x1, baseline_y),
                line,
                fontsize=fontsize,
                fontname="helv",
                color=(0, 0, 0),
                render_mode=3,   # 3 = invisible (searchable/copyable only)
                overlay=True,
            )


def make_searchable_overlay_pdf(
    original_pdf_path: Path,
    ocr_results: list,
    output_pdf_path: Path,
):
    """
    Copy original PDF pages pixel-perfect, overlay invisible selectable text
    at the exact bounding box coordinates from OCR.
    """
    src_doc = fitz.open(str(original_pdf_path))
    out_doc = fitz.open()

    for page_idx, page in enumerate(src_doc):
        out_doc.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
        out_page = out_doc[page_idx]

        page_w = page.rect.width
        page_h = page.rect.height

        if page_idx >= len(ocr_results):
            continue

        res    = ocr_results[page_idx]
        img_w  = res.get("width",  page_w)
        img_h  = res.get("height", page_h)
        blocks = _extract_text_blocks(res)

        print(f"    page {page_idx+1}: {len(blocks)} text blocks overlaid")
        _overlay_text_on_page(out_page, blocks, page_w, page_h, img_w, img_h)

    out_doc.save(str(output_pdf_path), garbage=4, deflate=True)
    src_doc.close()
    out_doc.close()


def make_searchable_pdf_from_image(
    image_path: Path,
    ocr_result,
    output_pdf_path: Path,
):
    """For PNG/JPG: embed original image + overlay invisible text."""
    out_doc  = fitz.open()
    img_doc  = fitz.open(str(image_path))
    img_rect = img_doc[0].rect
    img_doc.close()

    out_page = out_doc.new_page(width=img_rect.width, height=img_rect.height)
    out_page.insert_image(img_rect, filename=str(image_path))

    img_w  = ocr_result.get("width",  img_rect.width)
    img_h  = ocr_result.get("height", img_rect.height)
    blocks = _extract_text_blocks(ocr_result)

    print(f"    {len(blocks)} text blocks overlaid")
    _overlay_text_on_page(out_page, blocks, img_rect.width, img_rect.height, img_w, img_h)

    out_doc.save(str(output_pdf_path), garbage=4, deflate=True)
    out_doc.close()


# ── Main ───────────────────────────────────────────────────────────────────────

pipeline = PaddleOCRVL(**PIPELINE_OPTIONS)

for category, input_dir in INPUT_DIRS.items():
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[SKIP] Folder not found: {input_dir}")
        continue

    files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg")
    )

    print(f"\n{'='*55}")
    print(f" Processing {len(files)} file(s) from: {input_dir}")
    print(f"{'='*55}")

    for file in files:
        print(f"\n→ {file.name}")
        out_dir = Path(OUTPUT_BASE) / category / file.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            results_raw = pipeline.predict(str(file), **PREDICT_OPTIONS)

            results = pipeline.restructure_pages(
                results_raw,
                merge_tables=True,
                relevel_titles=True,
                concatenate_pages=False,
            )

            all_md_pages = []

            for i, res in enumerate(results):
                page_label = f"_page{i+1}" if len(results) > 1 else ""

                if SAVE_MARKDOWN:
                    res.save_to_markdown(str(out_dir))
                    print(f"  ✓ Markdown   → {out_dir}/{file.stem}{page_label}.md")

                if SAVE_JSON:
                    res.save_to_json(str(out_dir / f"{file.stem}{page_label}.json"))
                    print(f"  ✓ JSON       → {out_dir}/{file.stem}{page_label}.json")

                md_text = extract_markdown_text(res)
                if md_text:
                    all_md_pages.append(md_text)

            combined_md = "\n\n".join(all_md_pages)

            if SAVE_PLAINTEXT and combined_md:
                plain = markdown_to_plaintext(combined_md)
                txt_path = out_dir / f"{file.stem}.txt"
                txt_path.write_text(plain, encoding="utf-8")
                print(f"  ✓ Plaintext  → {txt_path}")

            if SAVE_SEARCHABLE_PDF:
                overlay_path = out_dir / f"{file.stem}_searchable.pdf"
                if file.suffix.lower() == ".pdf":
                    # Pass results_raw (not restructured) — has parsing_res_list intact
                    make_searchable_overlay_pdf(file, results_raw, overlay_path)
                else:
                    make_searchable_pdf_from_image(file, results_raw[0], overlay_path)
                print(f"  ✓ Search PDF → {overlay_path}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback; traceback.print_exc()

pipeline.close()
print(f"\n\nDone! Output saved to: {OUTPUT_BASE}")
