import re
from pathlib import Path

from paddleocr import PaddleOCRVL

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIRS = {
    "pdfs": "/home/dimitra/Documents/Github/PaddleOCR/my_project/pdf_orig",
    "pngs": "/home/dimitra/Documents/Github/PaddleOCR/my_project/png_orig",
}
OUTPUT_BASE = "/home/dimitra/Documents/Github/PaddleOCR/my_project/output_vl"

SAVE_MARKDOWN  = True
SAVE_PLAINTEXT = True
SAVE_JSON      = False

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

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback; traceback.print_exc()

pipeline.close()
print(f"\n\nDone! Output saved to: {OUTPUT_BASE}")
