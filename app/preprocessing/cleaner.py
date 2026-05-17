import re
import pandas as pd
from datasets import Dataset
from app.utils.logger import logger


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    Menerima string, return string yang sudah dibersihkan.
    """
    if not isinstance(text, str):  # kalau bukan string (None, int, dll) → return kosong
        return ""
    text = text.strip()                        # hapus whitespace di awal/akhir
    text = re.sub(r'\s+', ' ', text)          # collapse multiple whitespace jadi satu spasi
    text = re.sub(r'http\S+', '', text)        # hapus URL
    return text.strip()                        # strip lagi setelah substitusi


def remove_duplicates(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """
    Hapus baris duplikat berdasarkan kombinasi kolom di 'subset'.
    Menjaga baris pertama, hapus yang berikutnya.
    """
    before = len(df)                           # jumlah baris sebelum deduplikasi
    df = df.drop_duplicates(subset=subset)     # hapus duplikat berdasarkan kolom subset
    after = len(df)                            # jumlah baris setelah deduplikasi
    logger.info(f"Removed {before - after} duplicates")
    return df


def remove_missing(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    """
    Hapus baris yang kolom wajibnya kosong, None, atau NaN.
    
    Khusus untuk 'output' dari source openassistant:
    openassistant by-design punya output kosong (tree-based dataset),
    jadi kolom 'output' di-skip untuk source tersebut.
    """
    before = len(df)  # jumlah baris sebelum cleaning

    for col in required_cols:  # iterasi tiap kolom wajib
        # Paksa konversi ke string dulu supaya .isin() bisa compare dengan benar
        df[col] = df[col].astype(str).str.strip()

    for col in required_cols:  # iterasi kedua untuk filter setelah konversi
        if col == "output" and "source" in df.columns:
            # Untuk kolom output: skip baris openassistant karena memang kosong by design
            # Filter hanya berlaku untuk source selain openassistant
            non_oa_mask = df["source"] != "openassistant"  # mask: baris bukan openassistant
            empty_mask = df[col].isin(["", "nan", "None", "NaN"])  # mask: output kosong

            # Hapus baris yang: bukan openassistant DAN output-nya kosong
            df = df[~(non_oa_mask & empty_mask)]
        else:
            # Untuk kolom selain output: filter normal, hapus semua yang kosong
            df = df[~df[col].isin(["", "nan", "None", "NaN"])]

    after = len(df)  # jumlah baris setelah cleaning
    logger.info(f"Removed {before - after} missing/empty rows")
    return df


# Konfigurasi threshold panjang output per source dataset.
# Setiap dataset punya karakteristik panjang jawaban yang berbeda:
# - CoQA: jawaban sangat pendek (1-5 kata), contoh: "1475", "research", "five"
# - Dolly/Alpaca: response panjang dan deskriptif
# - OpenAssistant: output memang kosong (by design), jadi max_len sangat longgar
# - Company: fleksibel tergantung dokumen perusahaan
LENGTH_CONFIG = {
    "dolly":         {"min_len": 10, "max_len": 1500},
    "alpaca":        {"min_len": 10, "max_len": 1500},
    "openassistant": {"min_len": 0,  "max_len": 9999},  # output kosong by design, jangan difilter
    "coqa":          {"min_len": 1,  "max_len": 500},   # jawaban pendek 1-5 kata
    "company":       {"min_len": 5,  "max_len": 1500},
}


def filter_by_length(
    df: pd.DataFrame,
    col: str,
    min_len: int = 10,
    max_len: int = 2000
) -> pd.DataFrame:
    """
    Filter baris berdasarkan panjang karakter kolom tertentu.
    Dipakai sebagai fallback kalau source tidak ada di LENGTH_CONFIG.
    """
    before = len(df)  # jumlah baris sebelum filter
    df = df[df[col].str.len().between(min_len, max_len)]  # filter berdasarkan range panjang
    after = len(df)   # jumlah baris setelah filter
    logger.info(f"Filtered {before - after} rows by length in '{col}'")
    return df


def filter_by_length_per_source(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Filter panjang kolom secara source-aware: setiap source punya threshold sendiri
    sesuai LENGTH_CONFIG di atas.
    
    Lebih proper daripada filter_by_length() biasa karena menghindari
    CoQA terhapus semua akibat min_len=10 (jawabannya memang pendek).
    """
    filtered_parts = []  # tampung hasil filter tiap source

    for source, cfg in LENGTH_CONFIG.items():  # iterasi tiap source dan config-nya
        subset = df[df["source"] == source].copy()  # ambil baris source ini saja

        if len(subset) == 0:  # kalau source ini tidak ada di dataset, skip
            continue

        # Apply filter panjang sesuai config source ini
        subset = subset[subset[col].str.len().between(cfg["min_len"], cfg["max_len"])]
        filtered_parts.append(subset)  # simpan hasil filter

        logger.info(
            f"[{source}] filter '{col}': "
            f"min={cfg['min_len']}, max={cfg['max_len']} → {len(subset)} rows"
        )

    # Tangani source yang tidak ada di LENGTH_CONFIG (misal source baru)
    known_sources = set(LENGTH_CONFIG.keys())  # source yang sudah terdefinisi
    unknown_df = df[~df["source"].isin(known_sources)].copy()  # source yang belum dikenal

    if len(unknown_df) > 0:
        # Pakai threshold default untuk source yang tidak dikenal
        unknown_df = unknown_df[unknown_df[col].str.len().between(10, 1500)]
        filtered_parts.append(unknown_df)
        logger.warning(
            f"Unknown sources filtered with default threshold: "
            f"{unknown_df['source'].unique().tolist()}"
        )

    if not filtered_parts:  # kalau semua kosong (edge case)
        return df.iloc[0:0]  # return DataFrame kosong dengan struktur yang sama

    # Gabungkan semua subset yang sudah difilter
    result = pd.concat(filtered_parts, ignore_index=True)
    logger.info(f"Total after filter_by_length_per_source: {len(result)} rows")
    return result