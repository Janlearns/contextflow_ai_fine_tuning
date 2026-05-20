import os
import sys
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
# pyrefly: ignore [missing-import]
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# pyrefly: ignore [missing-import]
from trl import SFTTrainer
# pyrefly: ignore [missing-import]
from datasets import Dataset, load_from_disk
from app.utils.config import config
from app.utils.logger import logger


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str):
    """
    Load base model dengan 4-bit quantization (QLoRA) untuk RTX 2060 (6GB VRAM).
    device_map={"": 0} wajib — mencegah meta tensor error saat save model.
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        try:
            # pyrefly: ignore [missing-import]
            from transformers import BitsAndBytesConfig
            # pyrefly: ignore [missing-import]
            import bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
            logger.info("Loaded with 4bit quantization on GPU")
        except (ImportError, Exception) as e:
            logger.warning(f"bitsandbytes tidak tersedia: {e}. Fallback ke fp32 CPU.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
    else:
        logger.warning("CUDA tidak tersedia. Training di CPU (akan sangat lambat).")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    return model


def apply_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,    # Dinaikkan dari 0.05 → 0.1 untuk regularisasi anti-overfit
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # WAJIB — mencegah grad error saat training
    model.print_trainable_parameters()
    return model


def get_training_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,  # WAJIB — menghemat VRAM
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=False,   # WAJIB False — fp16 + 4bit quantization = crash
        bf16=False,   # WAJIB False — RTX 2060 tidak support bf16
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",   # Pilih checkpoint berdasarkan eval_loss terendah
        greater_is_better=False,              # Semakin rendah eval_loss = semakin bagus
        report_to="none",
        optim="adamw_torch",
        seed=42,
    )


def run_training(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_name: str = None,
    output_dir: str = None,
):
    model_name = model_name or config.BASE_MODEL
    output_dir = output_dir or config.OUTPUT_MODEL_DIR

    logger.info(f"=== Starting Fine-Tuning: {model_name} ===")

    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name)
    model = apply_lora(model)

    training_args = get_training_args(output_dir)

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