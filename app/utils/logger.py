import sys
import os
# pyrefly: ignore [missing-import]
from loguru import logger

# Pastikan folder logs ada
_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(_log_dir, exist_ok=True)

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <white>{message}</white>",
    level="INFO"
)
logger.add(
    os.path.join(_log_dir, "contextflow.log"),
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)