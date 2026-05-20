# pyrefly: ignore [missing-import]
from supabase import create_client, Client
from app.utils.config import config
from app.utils.logger import logger

_client: Client = None


def get_supabase() -> Client:
    """Inisialisasi Supabase client (singleton) dengan service_role key untuk bypass RLS."""
    global _client
    if _client is None:
        # Gunakan SERVICE_KEY agar bisa insert ke semua tabel (bypass RLS)
        key = config.SUPABASE_SERVICE_KEY or config.SUPABASE_KEY
        if not config.SUPABASE_URL or not key:
            logger.warning("SUPABASE_URL atau SUPABASE_KEY belum diset. Database tidak aktif.")
            return None
        try:
            _client = create_client(config.SUPABASE_URL, key)
            logger.info("Supabase client initialized (service_role)")
        except Exception as e:
            logger.error(f"Gagal inisialisasi Supabase: {e}")
            return None
    return _client