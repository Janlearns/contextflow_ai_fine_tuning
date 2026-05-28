"""
Fine-Tuning Trainer untuk ContextFlow AI.

Menggunakan Unsloth FastLanguageModel untuk fine-tuning yang:
- 2x lebih cepat dari PyTorch biasa
- 60% lebih hemat VRAM
- 4-bit quantization (QLoRA) secara native
- Cocok untuk GPU RTX 2050 (8GB VRAM)
"""

import os
import sys
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from unsloth import FastLanguageModel
# pyrefly: ignore [missing-import]
from transformers import TrainingArguments, EarlyStoppingCallback
# pyrefly: ignore [missing-import]
from trl import SFTTrainer
# pyrefly: ignore [missing-import]
from datasets import Dataset, load_from_disk
from app.utils.config import config
from app.utils.logger import logger


def load_model_and_tokenizer(model_name: str):
    """
    Load model dan tokenizer menggunakan Unsloth FastLanguageModel.
    Otomatis 4-bit quantization untuk hemat VRAM di RTX 2050 (8GB).
    """
    logger.info(f"Loading model: {model_name} (Unsloth 4-bit)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,           # QLoRA 4-bit — hemat VRAM
        dtype=None,                  # Auto-detect: bfloat16 jika support, else float16
        trust_remote_code=True,
    )

    # Pastikan pad_token tersedia
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Model loaded. Dtype: {model.dtype}")
    return model, tokenizer


def apply_lora(model):
    """
    Terapkan LoRA adapter menggunakan Unsloth get_peft_model.
    Gradient checkpointing 'unsloth' → 30% lebih hemat VRAM dari standar.
    """
    logger.info("Applying LoRA adapters (Unsloth)...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
            "gate_proj", "up_proj", "down_proj",      # MLP modules
        ],
        lora_dropout=0.1,    # Dinaikkan dari 0.05 → 0.1 untuk regularisasi anti-overfit
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth-optimized — 30% lebih hemat VRAM
        random_state=42,
    )

    # Log trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model


def get_training_args(output_dir: str) -> TrainingArguments:
    """
    Training arguments dioptimalkan untuk Unsloth + RTX 2050 (8GB VRAM).
    """
    # Auto-detect precision — Unsloth handle ini
    use_bf16 = FastLanguageModel.is_bfloat16_supported()
    use_fp16 = not use_bf16

    logger.info(f"Precision: {'bf16' if use_bf16 else 'fp16'}")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=use_fp16,                  # Auto-detect oleh Unsloth
        bf16=use_bf16,                  # Auto-detect oleh Unsloth
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",   # Pilih checkpoint berdasarkan eval_loss terendah
        greater_is_better=False,              # Semakin rendah eval_loss = semakin bagus
        report_to="none",
        optim="adamw_8bit",             # Lebih hemat VRAM dari adamw_torch
        seed=42,
    )


def run_training(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_name: str = None,
    output_dir: str = None,
):
    """
    Pipeline training lengkap menggunakan Unsloth + SFTTrainer.
    Return SFTTrainer agar kompatibel dengan train.py.
    """
    model_name = model_name or config.BASE_MODEL
    output_dir = output_dir or config.OUTPUT_MODEL_DIR

    logger.info(f"=== Starting Fine-Tuning: {model_name} ===")

    # Step 1: Load model & tokenizer (Unsloth)
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Step 2: Apply LoRA (Unsloth)
    model = apply_lora(model)

    # Step 3: Training arguments
    training_args = get_training_args(output_dir)

    # Step 4: SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,              # WAJIB pakai tokenizer= bukan processing_class=
        dataset_text_field="text",         # WAJIB — tanpa ini KeyError: None
        max_seq_length=config.MAX_SEQ_LENGTH,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop jika eval_loss tidak turun 3x eval berturut-turut
    )

    # Step 5: Train
    logger.info("Training started...")
    trainer.train()

    # Save model — WAJIB pakai trainer.model.save_pretrained()
    # BUKAN trainer.save_model() yang menyebabkan meta tensor error
    logger.info(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log saved files
    for f in os.listdir(output_dir):
        if not os.path.isdir(os.path.join(output_dir, f)):
            size = os.path.getsize(os.path.join(output_dir, f))
            logger.info(f"  {f}: {size/1024/1024:.1f} MB")

    logger.info("=== Training Complete ===")
    return trainer