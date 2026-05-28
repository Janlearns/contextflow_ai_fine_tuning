# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer
# pyrefly: ignore [missing-import]
from datasets import Dataset
from app.utils.logger import logger
from app.utils.config import config


def load_finetuned_model(model_dir: str, base_model: str):
    """Load fine-tuned model (LoRA adapter) di atas base model."""
    try:
        # pyrefly: ignore [missing-import]
        from peft import PeftModel
    except ImportError:
        logger.error("peft tidak terinstall. Jalankan: pip install peft")
        raise

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": 0},
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return response.strip()


def compute_bleu(predictions: list, references: list) -> float:
    try:
        # pyrefly: ignore [missing-import]
        from evaluate import load as load_metric
        bleu = load_metric("sacrebleu")
        result = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )
        return round(result["score"], 4)
    except Exception as e:
        logger.warning(f"Gagal compute BLEU: {e}")
        return 0.0


def compute_rouge(predictions: list, references: list) -> dict:
    try:
        # pyrefly: ignore [missing-import]
        from evaluate import load as load_metric
        rouge = load_metric("rouge")
        result = rouge.compute(predictions=predictions, references=references)
        return {k: round(v, 4) for k, v in result.items()}
    except Exception as e:
        logger.warning(f"Gagal compute ROUGE: {e}")
        return {}


def evaluate_on_dataset(model, tokenizer, val_dataset: Dataset, num_samples: int = 100) -> dict:
    logger.info(f"Evaluating on {num_samples} samples...")

    samples = val_dataset.select(range(min(num_samples, len(val_dataset))))
    predictions = []
    references = []

    for i, sample in enumerate(samples):
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{sample['instruction']}\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n"
        pred = generate_response(model, tokenizer, prompt)
        predictions.append(pred)
        references.append(sample["output"])

        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{num_samples} samples")

    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)

    results = {
        "bleu": bleu,
        **rouge,
        "num_samples": len(predictions),
    }

    logger.info(f"Evaluation Results: {results}")
    return results, predictions, references