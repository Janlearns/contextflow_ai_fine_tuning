import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.utils.config import config
from app.utils.logger import logger


class ContextFlowPredictor:
    def __init__(self, model_dir: str = None, base_model: str = None):
        self.model_dir = model_dir or config.OUTPUT_MODEL_DIR
        self.base_model = base_model or config.BASE_MODEL
        self.model = None
        self.tokenizer = None
        self.is_finetuned = False
        self._load()

    def _load(self):
        """
        Load model: coba fine-tuned dulu, kalau tidak ada fallback ke base model.
        """
        # Cek apakah fine-tuned model ada
        finetuned_exists = os.path.exists(self.model_dir) and any(
            f.endswith(('.bin', '.safetensors', 'adapter_config.json'))
            for f in os.listdir(self.model_dir)
        ) if os.path.exists(self.model_dir) else False

        if finetuned_exists:
            logger.info(f"Loading fine-tuned model dari {self.model_dir}...")
            try:
                from peft import PeftModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                if torch.cuda.is_available():
                    base = base.cuda()

                self.model = PeftModel.from_pretrained(base, self.model_dir)
                self.model.eval()
                self.is_finetuned = True
                logger.info("Fine-tuned model loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Gagal load fine-tuned model: {e}")

        # Fallback: gunakan base model langsung
        logger.info(f"Loading base model: {self.base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map={"": 0} if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()
        self.is_finetuned = False
        logger.info("Base model loaded (belum fine-tuned).")

    def build_prompt(self, instruction: str, context: str = "") -> str:
        """Prompt format ChatML (Qwen2.5) — harus konsisten dengan INSTRUCTION_TEMPLATE di formatter.py."""
        return (
            f"<|im_start|>system\n"
            f"You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{instruction}\n"
            f"{context}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def predict(
        self,
        instruction: str,
        context: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        prompt = self.build_prompt(instruction, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Pindahkan ke device yang sama dengan model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Ambil hanya bagian response (setelah prompt)
        response = decoded[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        return response.strip()

    def get_status(self) -> str:
        if self.is_finetuned:
            return f"✅ Fine-tuned model: {self.model_dir}"
        else:
            return f"⚠️ Base model (belum fine-tuned): {self.base_model}"