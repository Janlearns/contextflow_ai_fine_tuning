# pyrefly: ignore [missing-import]
from supabase import create_client, Client
from app.utils.config import config
from app.utils.logger import logger

_client: Client = None


def get_supabase() -> Client:
    """Inisialisasi Supabase client (singleton)."""
    global _client
    if _client is None:
        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            logger.warning("SUPABASE_URL atau SUPABASE_KEY belum diset. Database tidak aktif.")
            return None
        try:
            _client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.error(f"Gagal inisialisasi Supabase: {e}")
            return None
    return _client