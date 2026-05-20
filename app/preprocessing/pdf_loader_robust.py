# app/preprocessing/pdf_loader_robust.py

import os
import logging

logger = logging.getLogger(__name__)

# Threshold minimum karakter per halaman agar dianggap "text valid"
# Kalau di bawah ini, dianggap scanned/kosong → fallback ke OCR
MIN_CHARS_PER_PAGE = 50


def _extract_text_pdfplumber(filepath: str) -> list[dict]:
    """
    Coba extract text pakai pdfplumber (lebih akurat dari PyPDF2 untuk layout kompleks).
    Return list of {"page": int, "text": str}.
    """
    try:
        import pdfplumber  # pip install pdfplumber
    except ImportError:
        logger.warning("pdfplumber belum terinstall. Jalankan: pip install pdfplumber")
        return []

    results = []
    try:
        with pdfplumber.open(filepath) as pdf:  # buka file PDF
            for i, page in enumerate(pdf.pages):  # iterasi tiap halaman
                text = page.extract_text()  # ekstrak teks dari halaman
                results.append({
                    "page": i + 1,
                    # kalau text None (halaman kosong/gambar), fallback ke string kosong
                    "text": text.strip() if text else ""
                })
    except Exception as e:
        # Kalau file corrupt, password-protected, atau format aneh
        logger.warning(f"pdfplumber gagal buka {os.path.basename(filepath)}: {e}")
        return []

    return results


def _extract_text_ocr(filepath: str) -> list[dict]:
    """
    OCR fallback: konversi halaman PDF ke gambar dulu, lalu jalankan Tesseract.
    Dipakai kalau halaman PDF adalah hasil scan (isinya gambar, bukan text layer).
    
    Requires:
        pip install pdf2image pytesseract
        sudo apt install tesseract-ocr poppler-utils  (Linux)
        brew install tesseract poppler              (Mac)
    """
    try:
        # pyrefly: ignore [missing-import]
        from pdf2image import convert_from_path  # konversi PDF ke list PIL images
        # pyrefly: ignore [missing-import]
        import pytesseract  # wrapper Tesseract OCR
    except ImportError:
        logger.warning(
            "OCR dependencies belum terinstall. "
            "Jalankan: pip install pdf2image pytesseract"
        )
        return []

    results = []
    try:
        # Konversi semua halaman PDF ke gambar (DPI 200 cukup untuk OCR, 300 lebih akurat)
        images = convert_from_path(filepath, dpi=200)

        for i, img in enumerate(images):  # iterasi tiap gambar halaman
            # Jalankan Tesseract OCR; lang="ind+eng" untuk Bahasa Indonesia + Inggris
            # Ganti lang sesuai kebutuhan, atau "eng" aja kalau data lo Inggris semua
            text = pytesseract.image_to_string(img, lang="ind+eng")
            results.append({
                "page": i + 1,
                "text": text.strip() if text else ""
            })

    except Exception as e:
        logger.warning(f"OCR gagal untuk {os.path.basename(filepath)}: {e}")
        return []

    return results


def _is_text_valid(text: str) -> bool:
    """
    Cek apakah teks dari suatu halaman dianggap "cukup bermakna".
    Halaman scanned atau halaman penuh gambar biasanya return text kosong
    atau cuma noise (whitespace, karakter random, dll).
    """
    if not text:
        return False

    # Hapus whitespace, cek panjang karakter bersih
    cleaned = text.replace("\n", " ").replace("\t", " ").strip()
    
    # Terlalu pendek → kemungkinan besar halaman gambar atau header doang
    if len(cleaned) < MIN_CHARS_PER_PAGE:
        return False

    # Rasio karakter alfanumerik vs total karakter
    # Kalau banyak karakter aneh (OCR noise dari gambar), rasio ini rendah
    alnum_count = sum(1 for c in cleaned if c.isalnum())
    alnum_ratio = alnum_count / len(cleaned)

    # Kalau kurang dari 40% alfanumerik, teks dianggap noise
    return alnum_ratio >= 0.4


def load_pdf_robust(filepath: str) -> list:
    """
    Load PDF dengan strategi berlapis:
    1. Coba extract text biasa via pdfplumber
    2. Kalau halaman tertentu kosong/noise → fallback OCR untuk halaman itu
    3. Kalau OCR juga gagal → skip halaman, log warning
    
    Return: list of dicts siap masuk Dataset
    {instruction, context, response, source}
    """
    filename = os.path.basename(filepath)
    logger.info(f"Loading PDF: {filename}")

    # === STEP 1: Ekstrak text via pdfplumber ===
    pages_text = _extract_text_pdfplumber(filepath)

    if not pages_text:
        # pdfplumber total fail (corrupt, dll) → langsung coba full OCR
        logger.warning(f"{filename}: pdfplumber gagal total, mencoba full OCR...")
        pages_text = _extract_text_ocr(filepath)

    if not pages_text:
        # Kedua metode gagal → skip file ini
        logger.error(f"{filename}: Tidak bisa diekstrak sama sekali. Skip.")
        return []

    # === STEP 2: Identifikasi halaman mana yang perlu OCR ===
    # Pisahkan halaman yang text-nya valid vs yang perlu OCR
    valid_pages = []    # halaman dengan text OK
    ocr_needed = []     # nomor halaman yang perlu di-OCR

    for page_data in pages_text:  # iterasi hasil ekstraksi
        if _is_text_valid(page_data["text"]):
            valid_pages.append(page_data)
        else:
            # Halaman ini kosong atau noise → tandai untuk OCR
            ocr_needed.append(page_data["page"])

    if ocr_needed:
        logger.info(
            f"{filename}: {len(ocr_needed)} halaman perlu OCR "
            f"(halaman {ocr_needed})"
        )

        # Jalankan OCR hanya untuk halaman yang butuh
        # (convert_from_path bisa terima first_page & last_page untuk efisiensi)
        # Tapi karena halaman yang perlu OCR bisa tidak berurutan,
        # kita OCR semua lalu filter
        ocr_pages = _extract_text_ocr(filepath)

        # Buat mapping page_number → text dari hasil OCR
        ocr_map = {p["page"]: p["text"] for p in ocr_pages}  # dict lookup cepat

        for page_num in ocr_needed:  # cek tiap halaman yang butuh OCR
            ocr_text = ocr_map.get(page_num, "")

            if _is_text_valid(ocr_text):
                # OCR berhasil → tambah ke valid pages
                valid_pages.append({"page": page_num, "text": ocr_text})
                logger.debug(f"{filename}: Halaman {page_num} OK via OCR")
            else:
                # OCR juga gagal → skip halaman ini
                logger.warning(
                    f"{filename}: Halaman {page_num} skip "
                    f"(kosong bahkan setelah OCR — kemungkinan full-image atau blank)"
                )

    # === STEP 3: Format ke struktur Dataset ===
    # Urutkan berdasarkan nomor halaman (bisa acak setelah merge)
    valid_pages.sort(key=lambda x: x["page"])

    rows = []
    for page_data in valid_pages:  # konversi tiap halaman ke row dataset
        page_num = page_data["page"]
        text = page_data["text"]

        rows.append({
            "instruction": f"Jelaskan isi dari dokumen '{filename}' halaman {page_num}.",
            "context": text[:2000],   # potong kalau terlalu panjang
            "response": text[:1000],  # response lebih pendek dari context
        })

    logger.info(
        f"{filename}: {len(rows)} halaman valid dari {len(pages_text)} total halaman"
    )
    return rows