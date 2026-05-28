import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env dari root project
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


class Config:
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    SUPABASE_DB_URL: str = os.getenv("SUPABASE_DB_URL", "")

    # Model — Qwen2.5-3B-Instruct (4-bit) via Unsloth — jauh lebih bagus dari TinyLlama
    BASE_MODEL: str = os.getenv("BASE_MODEL", "unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    OUTPUT_MODEL_DIR: str = str(
        _project_root / os.getenv("OUTPUT_MODEL_DIR", "models/contextflow-finetuned")
    ) if not os.path.isabs(os.getenv("OUTPUT_MODEL_DIR", "")) else os.getenv("OUTPUT_MODEL_DIR")

    # Training
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "5e-5"))  # Diturunkan dari 2e-4 → 5e-5 untuk mencegah overfit
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1"))
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "3"))
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", "512"))  # Dinaikkan dari 256 → 512, Qwen2.5 bisa handle lebih panjang

    # Paths (relative to project root)
    PROJECT_ROOT: str = str(_project_root)
    RAW_DATA_DIR: str = str(_project_root / "data" / "raw")
    PROCESSED_DATA_DIR: str = str(_project_root / "data" / "processed")
    FORMATTED_DATA_DIR: str = str(_project_root / "data" / "formatted")
    MODELS_DIR: str = str(_project_root / "models")
    LOGS_DIR: str = str(_project_root / "logs")


config = Config()

# Buat direktori yang diperlukan
os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(config.FORMATTED_DATA_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.LOGS_DIR, exist_ok=True)