# How to Run the OCR Pipeline

This guide explains how to set up and run the document OCR pipeline (`my_project/run_ocr_vl.py`).  
It processes PDFs and images and outputs structured **Markdown** and **plain text** files.

---

## Requirements

- Python 3.10 or higher
- A Linux machine with a GPU is recommended (it will run on CPU but very slowly)
- Git

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/diniaouri/PaddleOCR.git
cd PaddleOCR
```

---

## Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

> You need to activate the environment every time you open a new terminal:
> ```bash
> source venv/bin/activate
> ```

---

## Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install paddlepaddle paddleocr paddlex pymupdf pillow numpy transformers
```

> ⚠️ If you have a GPU, install the GPU version of PaddlePaddle instead.  
> Check https://www.paddlepaddle.org.cn/install/quick for the right command for your CUDA version.

---

## Step 4 — Add your input files

Place your files in the correct input folders:

| File type | Folder |
|-----------|--------|
| PDF files | `my_project/pdf_orig/` |
| PNG / JPG images | `my_project/png_orig/` |

Create the folders if they don't exist:

```bash
mkdir -p my_project/pdf_orig
mkdir -p my_project/png_orig
```

Then copy your PDFs or images into those folders.

---

## Step 5 — Run the script

```bash
python my_project/run_ocr_vl.py
```

The first run will automatically download the required AI models (~several GB).  
This only happens once — subsequent runs will be much faster.

---

## Step 6 — Find your output

Results are saved in `my_project/output_vl/`, organised by input type and file name:

```
my_project/output_vl/
├── pdfs/
│   └── your_document/
│       ├── your_document.md       ← structured markdown
│       └── your_document.txt      ← clean plain text
└── pngs/
    └── your_image/
        ├── your_image.md
        └── your_image.txt
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: paddleocr` | Make sure your venv is activated: `source venv/bin/activate` |
| `ModuleNotFoundError: fitz` | Run `pip install pymupdf` |
| Script is very slow | You are running on CPU — this is normal, expect several minutes per page |
| Empty output files | Check the terminal for `✗ ERROR:` lines and share them for help |
| Out of memory | Reduce `max_new_tokens` in the script from `4096` to `2048` |
