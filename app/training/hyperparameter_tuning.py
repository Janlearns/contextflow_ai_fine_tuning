from dataclasses import dataclass
from typing import List
from app.utils.logger import logger


@dataclass
class HyperparamSet:
    learning_rate: float
    batch_size: int
    num_epochs: int
    lora_r: int
    lora_alpha: int
    max_seq_length: int


HYPERPARAMETER_GRID: List[HyperparamSet] = [
    HyperparamSet(learning_rate=2e-4, batch_size=4, num_epochs=3, lora_r=16, lora_alpha=32, max_seq_length=512),
    HyperparamSet(learning_rate=1e-4, batch_size=4, num_epochs=3, lora_r=8,  lora_alpha=16, max_seq_length=512),
    HyperparamSet(learning_rate=3e-4, batch_size=2, num_epochs=5, lora_r=32, lora_alpha=64, max_seq_length=512),
]


def log_hyperparams(params: HyperparamSet, trial: int):
    logger.info(f"--- Trial {trial} ---")
    logger.info(f"LR: {params.learning_rate} | Batch: {params.batch_size} | Epochs: {params.num_epochs}")
    logger.info(f"LoRA r: {params.lora_r} | Alpha: {params.lora_alpha} | Seq len: {params.max_seq_length}")