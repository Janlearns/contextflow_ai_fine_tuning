"""
Entry point untuk training pipeline ContextFlow AI.
Jalankan: python train.py
"""
from app.preprocessing.pipeline import run_preprocessing_pipeline
from app.training.trainer import run_training
from app.database.repository import save_model_metadata, save_training_dataset
from app.utils.logger import logger
from app.utils.config import config


def main():
    logger.info("====== ContextFlow AI Training Pipeline ======")

    # Step 1: Preprocessing
    logger.info("Step 1: Preprocessing...")
    train_ds, val_ds = run_preprocessing_pipeline(sample_size=10000)

    # Step 2: Fine-Tuning
    logger.info("Step 2: Fine-Tuning...")
    trainer = run_training(
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_name=config.BASE_MODEL,
        output_dir=config.OUTPUT_MODEL_DIR,
    )

    # Step 3: Simpan ke Supabase
    logger.info("Step 3: Saving to Supabase...")

    # Simpan model metadata
    train_metrics = {}
    try:
        last_log = trainer.state.log_history[-1] if trainer.state.log_history else {}
        train_metrics = {
            "bleu": None,
            "rouge1": None,
            "rougeL": None,
            "train_loss": last_log.get("train_loss"),
            "eval_loss": last_log.get("eval_loss"),
        }
    except Exception:
        pass

    model_id = save_model_metadata(
        model_name="contextflow-finetuned",
        base_model=config.BASE_MODEL,
        metrics=train_metrics,
    )

    # Simpan training dataset ke Supabase
    if model_id:
        logger.info(f"Saving training dataset ({len(train_ds)} samples) to Supabase...")
        saved = save_training_dataset(model_id, train_ds)
        logger.info(f"Training dataset saved: {saved} samples linked to model {model_id}")
    else:
        logger.warning("Model ID tidak tersedia, training dataset tidak disimpan ke Supabase.")

    logger.info("====== Training Done ======")


if __name__ == "__main__":
    main()