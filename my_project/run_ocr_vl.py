import re
import shutil
import tempfile
from pathlib import Path

import fitz  # pymupdf
from paddleocr import PaddleOCR, PaddleOCRVL

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

# VLM pipeline — high quality markdown/text output
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

# DPI to render PDF pages for classic OCR (higher = more accurate bbox coords)
PDF_RENDER_DPI = 150
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


def _get_classic_ocr_blocks(classic_ocr, image_path: str) -> list[dict]:
    """
    Run classic PP-OCRv4 on an image file.
    Returns list of {"bbox": [x1,y1,x2,y2], "text": str, "img_w": int, "img_h": int}
    One entry per detected text line with precise bboxes.
    """
    import numpy as np
    from PIL import Image

    img    = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    img_np = np.array(img)

    result = classic_ocr.predict(img_np)

    blocks = []
    for page in result:
        if page is None:
            continue
        texts = page.get("rec_texts", [])
        boxes = page.get("rec_boxes", None)    # [[x1,y1,x2,y2], ...]
        polys = page.get("det_polys", None)    # [[[x,y],...], ...]

        if boxes is None and polys is not None:
            boxes = []
            for poly in polys:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])

        if boxes is None:
            continue

        for text, box in zip(texts, boxes):
            text = text.strip()
            if not text:
                continue
            blocks.append({
                "bbox":  [float(v) for v in box],
                "text":  text,
                "img_w": img_w,
                "img_h": img_h,
            })

    return blocks


def _overlay_blocks_on_page(out_page, blocks, page_w, page_h):
    """
    Place invisible selectable text onto a PDF page.
    blocks: list of {"bbox": [x1,y1,x2,y2], "text": str, "img_w": int, "img_h": int}
    Each block is one OCR line — placed at precise baseline position.
    """
    for block in blocks:
        bbox  = block["bbox"]
        text  = block["text"]
        img_w = block["img_w"]
        img_h = block["img_h"]

        # Scale OCR image pixel coords → PDF point coords
        x1 = bbox[0] * page_w / img_w
        y1 = bbox[1] * page_h / img_h
        x2 = bbox[2] * page_w / img_w
        y2 = bbox[3] * page_h / img_h

        box_h = y2 - y1
        box_w = x2 - x1
        if box_w <= 0 or box_h <= 0:
            continue

        fontsize = max(4.0, min(12.0, box_h * 0.85))

        out_page.insert_text(
            fitz.Point(x1, y2 - box_h * 0.1),  # baseline near bottom of bbox
            text,
            fontsize=fontsize,
            fontname="helv",
            color=(0, 0, 0),
            render_mode=3,   # invisible but searchable/copyable
            overlay=True,
        )


def make_searchable_overlay_pdf(
    original_pdf_path: Path,
    output_pdf_path: Path,
    classic_ocr,
    tmp_dir: Path,
):
    """
    Render each PDF page to an image, run classic PP-OCRv4 for precise
    line bboxes, then write a PDF with the original page + invisible text overlay.
    """
    src_doc = fitz.open(str(original_pdf_path))
    out_doc = fitz.open()
    mat     = fitz.Matrix(PDF_RENDER_DPI / 72, PDF_RENDER_DPI / 72)

    for page_idx, page in enumerate(src_doc):
        # Copy original page pixel-perfect
        out_doc.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
        out_page = out_doc[page_idx]

        page_w = page.rect.width
        page_h = page.rect.height

        # Render page to PNG for classic OCR
        pix      = page.get_pixmap(matrix=mat, alpha=False)
        img_path = str(tmp_dir / f"page_{page_idx}.png")
        pix.save(img_path)

        blocks = _get_classic_ocr_blocks(classic_ocr, img_path)
        print(f"    page {page_idx+1}: {len(blocks)} text lines overlaid")
        _overlay_blocks_on_page(out_page, blocks, page_w, page_h)

    out_doc.save(str(output_pdf_path), garbage=4, deflate=True)
    src_doc.close()
    out_doc.close()


def make_searchable_pdf_from_image(
    image_path: Path,
    output_pdf_path: Path,
    classic_ocr,
):
    """For PNG/JPG: embed original image as background + overlay invisible text."""
    out_doc  = fitz.open()
    img_doc  = fitz.open(str(image_path))
    img_rect = img_doc[0].rect
    img_doc.close()

    out_page = out_doc.new_page(width=img_rect.width, height=img_rect.height)
    out_page.insert_image(img_rect, filename=str(image_path))

    blocks = _get_classic_ocr_blocks(classic_ocr, str(image_path))
    print(f"    {len(blocks)} text lines overlaid")
    _overlay_blocks_on_page(out_page, blocks, img_rect.width, img_rect.height)

    out_doc.save(str(output_pdf_path), garbage=4, deflate=True)
    out_doc.close()


# ── Main ───────────────────────────────────────────────────────────────────────

# Pass 1: VLM — high quality structured markdown + plain text
vlm_pipeline = PaddleOCRVL(**PIPELINE_OPTIONS)

# Pass 2: Classic OCR — precise line-level bboxes for searchable PDF overlay
# Note: show_log is not a valid arg in this version of PaddleOCR
classic_ocr = None
if SAVE_SEARCHABLE_PDF:
    classic_ocr = PaddleOCR(ocr_version="PP-OCRv4", lang="en")

tmp_dir = Path(tempfile.mkdtemp())

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
            # ── Pass 1: VLM → markdown + plain text ──────────────────────
            results_raw = vlm_pipeline.predict(str(file), **PREDICT_OPTIONS)

            results = vlm_pipeline.restructure_pages(
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

            # ── Pass 2: Classic OCR → precise bboxes → searchable PDF ────
            if SAVE_SEARCHABLE_PDF:
                overlay_path = out_dir / f"{file.stem}_searchable.pdf"
                if file.suffix.lower() == ".pdf":
                    make_searchable_overlay_pdf(file, overlay_path, classic_ocr, tmp_dir)
                else:
                    make_searchable_pdf_from_image(file, overlay_path, classic_ocr)
                print(f"  ✓ Search PDF → {overlay_path}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback; traceback.print_exc()

vlm_pipeline.close()
shutil.rmtree(tmp_dir, ignore_errors=True)
print(f"\n\nDone! Output saved to: {OUTPUT_BASE}")
