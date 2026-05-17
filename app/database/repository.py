import time
from uuid import uuid4
from app.database.supabase_client import get_supabase
from app.utils.logger import logger


def _get_db():
    """Helper: ambil Supabase client, return None jika tidak tersedia."""
    db = get_supabase()
    if db is None:
        logger.warning("Database tidak tersedia, operasi dilewati.")
    return db


def save_message(session_id: str, role: str, content: str, conversation_id: str = None):
    db = _get_db()
    if not db:
        return None
    try:
        data = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
        }
        db.table("messages").insert(data).execute()
    except Exception as e:
        logger.error(f"Gagal save message: {e}")


def save_prediction(session_id: str, instruction: str, response: str, context: str = "", latency_ms: float = 0):
    db = _get_db()
    if not db:
        return str(uuid4())  # Return dummy ID agar UI tetap berjalan
    try:
        data = {
            "session_id": session_id,
            "instruction": instruction,
            "context": context,
            "response": response,
            "model_version": "contextflow-v1",
            "latency_ms": latency_ms,
        }
        result = db.table("predictions").insert(data).execute()
        return result.data[0]["id"] if result.data else str(uuid4())
    except Exception as e:
        logger.error(f"Gagal save prediction: {e}")
        return str(uuid4())


def save_feedback(prediction_id: str, rating: int, comment: str = ""):
    db = _get_db()
    if not db:
        return
    try:
        db.table("feedback").insert({
            "prediction_id": prediction_id,
            "rating": rating,
            "comment": comment,
        }).execute()
    except Exception as e:
        logger.error(f"Gagal save feedback: {e}")


def get_conversation_history(session_id: str, limit: int = 20) -> list:
    db = _get_db()
    if not db:
        return []
    try:
        result = (
            db.table("predictions")
            .select("session_id, instruction, response, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Gagal get conversation history: {e}")
        return []


def save_model_metadata(model_name: str, base_model: str, metrics: dict):
    """Simpan metadata model ke Supabase. Return model_id."""
    db = _get_db()
    if not db:
        return None
    try:
        result = db.table("model_metadata").insert({
            "model_name": model_name,
            "base_model": base_model,
            "bleu_score": metrics.get("bleu"),
            "rouge1": metrics.get("rouge1"),
            "rougeL": metrics.get("rougeL"),
        }).execute()
        model_id = result.data[0]["id"] if result.data else None
        logger.info(f"Model metadata saved: {model_id}")
        return model_id
    except Exception as e:
        logger.error(f"Gagal save model metadata: {e}")
        return None


def save_training_dataset(model_id: str, dataset, batch_size: int = 500):
    """
    Simpan semua training samples ke Supabase setelah fine-tuning berhasil.
    Dataset bisa berupa HuggingFace Dataset atau pandas DataFrame.
    Data di-upload dalam batch untuk efisiensi.
    """
    db = _get_db()
    if not db:
        logger.warning("Database tidak tersedia, training dataset tidak disimpan.")
        return 0

    try:
        # Convert ke list of dicts
        if hasattr(dataset, 'to_pandas'):
            df = dataset.to_pandas()
        else:
            df = dataset

        records = []
        for _, row in df.iterrows():
            records.append({
                "model_id": model_id,
                "instruction": str(row.get("instruction", ""))[:5000],
                "context": str(row.get("input", ""))[:5000],
                "response": str(row.get("output", ""))[:5000],
                "source": str(row.get("source", "unknown")),
            })

        # Upload in batches
        total_saved = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                db.table("training_datasets").insert(batch).execute()
                total_saved += len(batch)
                logger.info(f"  Saved batch {i//batch_size + 1}: "
                            f"{total_saved}/{len(records)} samples")
            except Exception as e:
                logger.error(f"  Gagal save batch {i//batch_size + 1}: {e}")

        logger.info(f"Training dataset saved to Supabase: {total_saved}/{len(records)} samples")
        return total_saved

    except Exception as e:
        logger.error(f"Gagal save training dataset: {e}")
        return 0