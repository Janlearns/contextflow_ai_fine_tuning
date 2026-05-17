import os
import pandas as pd
from datasets import Dataset, concatenate_datasets
from app.preprocessing.data_loader import load_all_datasets, save_raw_datasets
from app.preprocessing.formatter import apply_formatter
from app.preprocessing.cleaner import remove_duplicates, remove_missing, filter_by_length
from app.utils.config import config
from app.utils.logger import logger
from typing import Tuple


def run_preprocessing_pipeline(sample_size: int = None) -> Tuple[Dataset, Dataset]:
    """
    Pipeline preprocessing lengkap:
    1. Load semua dataset
    2. Format ke instruction style
    3. Clean & deduplicate
    4. Split train/val
    5. Simpan ke disk
    """
    logger.info("=== Starting Preprocessing Pipeline ===")

    # Step 1: Load
    raw_datasets = load_all_datasets()
    save_raw_datasets(raw_datasets)

    # Step 2: Format
    formatted = []
    for source, ds in raw_datasets.items():
        if sample_size:
            ds = ds.select(range(min(sample_size, len(ds))))
        try:
            formatted_ds = apply_formatter(ds, source)
            formatted.append(formatted_ds)
        except Exception as e:
            logger.warning(f"Skipping {source} due to error: {e}")

    if not formatted:
        raise ValueError("Tidak ada dataset yang berhasil diformat!")

    combined = concatenate_datasets(formatted)
    logger.info(f"Total combined samples: {len(combined)}")

    # Step 3: Convert ke pandas untuk cleaning
    df = combined.to_pandas()

    # Clean
    df = remove_missing(df, required_cols=["instruction", "output"])
    df = remove_duplicates(df, subset=["instruction", "output"])
    df = filter_by_length(df, col="output", min_len=1, max_len=1500)
    df = df[df["instruction"].str.len() >= 5]
    df = df.reset_index(drop=True)

    logger.info(f"Final clean samples: {len(df)}")

    # Step 4: Split
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    # Step 5: Simpan
    os.makedirs(config.FORMATTED_DATA_DIR, exist_ok=True)

    train_ds.save_to_disk(os.path.join(config.FORMATTED_DATA_DIR, "train"))
    val_ds.save_to_disk(os.path.join(config.FORMATTED_DATA_DIR, "val"))

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    logger.info("=== Preprocessing Pipeline Done ===")

    return train_ds, val_ds