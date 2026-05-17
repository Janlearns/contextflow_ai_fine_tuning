"""
Company Data Loader — Mendukung JSON, CSV, PDF, DOCX, dan TXT.

File data perusahaan ditaruh di: data/raw/company/
Supported formats:
  - JSON  (.json)  → [{"instruction": "...", "context": "...", "response": "..."}]
  - CSV   (.csv)   → kolom: instruction, context, response
  - PDF   (.pdf)   → teks diekstrak per halaman sebagai context
  - DOCX  (.docx)  → teks diekstrak per paragraf sebagai context
  - TXT   (.txt)   → seluruh isi file sebagai context

Untuk PDF/DOCX/TXT:
  Karena file dokumen tidak memiliki pasangan Q&A, teks yang diekstrak
  dijadikan 'context' dan sistem akan generate pertanyaan generik.
  Untuk hasil terbaik, gunakan format JSON/CSV dengan pasangan Q&A eksplisit.
"""

import os
import json
import csv
import glob
# pyrefly: ignore [missing-import]
from datasets import Dataset
from app.utils.logger import logger
from app.utils.config import config
from app.preprocessing.pdf_loader_robust import load_pdf_robust


COMPANY_DATA_DIR = os.path.join(config.RAW_DATA_DIR, "company")


def _load_json_file(filepath: str) -> list:
    """Load JSON file berisi list of {instruction, context, response}."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    rows = []
    for item in data:
        rows.append({
            "instruction": item.get("instruction", ""),
            "context": item.get("context", ""),
            "response": item.get("response", ""),
        })
    logger.info(f"  Loaded {len(rows)} QA pairs from {os.path.basename(filepath)}")
    return rows


def _load_csv_file(filepath: str) -> list:
    """Load CSV file dengan kolom: instruction, context, response."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "instruction": row.get("instruction", ""),
                "context": row.get("context", ""),
                "response": row.get("response", ""),
            })
    logger.info(f"  Loaded {len(rows)} QA pairs from {os.path.basename(filepath)}")
    return rows


def _load_pdf_file(filepath: str) -> list:
    
    return load_pdf_robust(filepath)


def _load_docx_file(filepath: str) -> list:
    """Ekstrak teks dari DOCX per section sebagai context."""
    try:
        # pyrefly: ignore [missing-import]
        from docx import Document
    except ImportError:
        logger.warning("python-docx belum terinstall. Jalankan: pip install python-docx")
        return []

    doc = Document(filepath)
    filename = os.path.basename(filepath)

    # Gabungkan paragraf yang berurutan menjadi section
    sections = []
    current_section = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            if current_section:
                sections.append("\n".join(current_section))
                current_section = []
        else:
            current_section.append(text)

    if current_section:
        sections.append("\n".join(current_section))

    rows = []
    for i, section in enumerate(sections):
        if len(section) > 20:  # Skip section terlalu pendek
            rows.append({
                "instruction": f"Apa isi dari bagian {i+1} dokumen {filename}?",
                "context": section[:2000],
                "response": section[:1000],
            })

    logger.info(f"  Extracted {len(rows)} sections from {filename}")
    return rows


def _load_txt_file(filepath: str) -> list:
    """Load TXT file, split per paragraf sebagai context."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    rows = []
    for i, para in enumerate(paragraphs):
        if len(para) > 20:
            rows.append({
                "instruction": f"Jelaskan isi dari bagian {i+1} dokumen {filename}",
                "context": para[:2000],
                "response": para[:1000],
            })

    logger.info(f"  Extracted {len(rows)} paragraphs from {filename}")
    return rows


# Extension → loader mapping
FILE_LOADERS = {
    ".json": _load_json_file,
    ".csv": _load_csv_file,
    ".pdf": _load_pdf_file,
    ".docx": _load_docx_file,
    ".txt": _load_txt_file,
}


def load_company_data() -> Dataset:
    """
    Load semua file dari data/raw/company/ (JSON, CSV, PDF, DOCX, TXT).
    Return sebagai Hugging Face Dataset.
    """
    if not os.path.exists(COMPANY_DATA_DIR):
        logger.info(f"Company data directory tidak ditemukan: {COMPANY_DATA_DIR}")
        return None

    all_rows = []

    # Scan semua file di directory
    for ext, loader in FILE_LOADERS.items():
        pattern = os.path.join(COMPANY_DATA_DIR, f"*{ext}")
        files = glob.glob(pattern)
        # Juga cari di subdirectory
        pattern_sub = os.path.join(COMPANY_DATA_DIR, "**", f"*{ext}")
        files += glob.glob(pattern_sub, recursive=True)
        files = list(set(files))  # Deduplicate

        for filepath in sorted(files):
            try:
                rows = loader(filepath)
                all_rows.extend(rows)
            except Exception as e:
                logger.warning(f"  Gagal load {filepath}: {e}")

    if not all_rows:
        logger.warning("Tidak ada data perusahaan yang ditemukan di data/raw/company/")
        return None

    # Filter rows yang valid (minimal ada instruction dan response)
    valid_rows = [
        r for r in all_rows
        if r.get("instruction", "").strip() and r.get("response", "").strip()
    ]

    logger.info(f"Total company data loaded: {len(valid_rows)} QA pairs "
                f"(dari {len(all_rows)} total, {len(all_rows) - len(valid_rows)} invalid)")

    return Dataset.from_list(valid_rows)
