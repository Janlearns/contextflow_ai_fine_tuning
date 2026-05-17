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

    # Model
    BASE_MODEL: str = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    OUTPUT_MODEL_DIR: str = str(
        _project_root / os.getenv("OUTPUT_MODEL_DIR", "models/contextflow-finetuned")
    ) if not os.path.isabs(os.getenv("OUTPUT_MODEL_DIR", "")) else os.getenv("OUTPUT_MODEL_DIR")

    # Training
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1"))
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "3"))
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", "256"))

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