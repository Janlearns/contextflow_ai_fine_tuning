import pandas as pd
from datasets import Dataset
from app.utils.logger import logger
from app.preprocessing.cleaner import clean_text


# Format prompt bawaan TinyLlama-Chat — wajib konsisten antara training, inference, dan evaluasi.
# Menggunakan token pembungkus <|system|>, <|user|>, <|assistant|> sesuai arsitektur TinyLlama.
INSTRUCTION_TEMPLATE = """<|system|>
You are a helpful assistant.</s>
<|user|>
{instruction}
{context}</s>
<|assistant|>
{response}</s>"""


def format_dolly(sample: dict) -> dict:
    instruction = clean_text(sample.get("instruction", ""))
    context = clean_text(sample.get("context", ""))
    response = clean_text(sample.get("response", ""))
    return {
        "instruction": instruction,
        "input": context,
        "output": response,
        "text": INSTRUCTION_TEMPLATE.format(
            instruction=instruction,
            context=context,
            response=response
        ),
        "source": "dolly"
    }


def format_alpaca(sample: dict) -> dict:
    instruction = clean_text(sample.get("instruction", ""))
    context = clean_text(sample.get("input", ""))
    response = clean_text(sample.get("output", ""))
    return {
        "instruction": instruction,
        "input": context,
        "output": response,
        "text": INSTRUCTION_TEMPLATE.format(
            instruction=instruction,
            context=context,
            response=response
        ),
        "source": "alpaca"
    }


def format_openassistant(sample: dict) -> dict:
    """
    Format OpenAssistant oasst1 dataset.
    
    Setelah di-pair di loader, kolom yang tersedia:
      - instruction: teks pertanyaan dari prompter
      - response: teks jawaban dari assistant
    
    Tidak ada context/input karena oasst1 adalah percakapan langsung.
    """
    instruction = clean_text(sample.get("instruction", "") or "")  # ambil pertanyaan
    response = clean_text(sample.get("response", "") or "")        # ambil jawaban

    return {
        "instruction": instruction,
        "input": "",   # oasst1 tidak punya context tambahan
        "output": response,
        "text": INSTRUCTION_TEMPLATE.format(
            instruction=instruction,
            context="",
            response=response
        ),
        "source": "openassistant"
    }


def format_coqa(sample: dict) -> dict:
    """
    CoQA struktur: satu sample = satu story + list questions + list answers.
    Formatter ini hanya bisa handle SATU Q&A pair per row,
    jadi loader harus sudah flatten dulu sebelum masuk sini.
    
    Kolom yang diexpect setelah flatten:
    - question: str  (satu pertanyaan)
    - answer: str    (satu jawaban)
    - context: str   (story/paragraph)
    """
    instruction = clean_text(sample.get("question", ""))
    context = clean_text(sample.get("context", "")[:500])  # story sebagai context
    
    # Ambil answer — sudah berupa string setelah di-flatten loader
    answer = sample.get("answer", "") or ""
    
    # Guard: kalau somehow masih list atau dict (flatten belum sempurna)
    if isinstance(answer, list):
        answer = answer[0] if answer else ""     # ambil elemen pertama
    if isinstance(answer, dict):
        answer = answer.get("input_text", "")   # ambil text dari dict

    response = clean_text(str(answer))

    return {
        "instruction": instruction,
        "input": context,
        "output": response,
        "text": INSTRUCTION_TEMPLATE.format(
            instruction=instruction,
            context=context,
            response=response
        ),
        "source": "coqa"
    }


def format_company(sample: dict) -> dict:
    """
    Format company/enterprise dataset.
    Kolom: instruction, context, response — dari JSON/CSV/PDF/DOCX.
    """
    instruction = clean_text(sample.get("instruction", ""))
    context = clean_text(sample.get("context", "")[:500])
    response = clean_text(sample.get("response", ""))

    return {
        "instruction": instruction,
        "input": context,
        "output": response,
        "text": INSTRUCTION_TEMPLATE.format(
            instruction=instruction,
            context=context,
            response=response
        ),
        "source": "company"
    }


FORMATTERS = {
    "dolly": format_dolly,
    "alpaca": format_alpaca,
    "openassistant": format_openassistant,
    "coqa": format_coqa,
    "company": format_company,
}


def apply_formatter(dataset: Dataset, source: str) -> Dataset:
    formatter = FORMATTERS[source]
    
    # Hapus semua kolom asli DULU sebelum map, supaya tidak ada schema conflict
    # antara kolom lama (question, context, answer) dan kolom baru (instruction, input, output)
    formatted = dataset.map(
        formatter,
        remove_columns=dataset.column_names,  # ← hapus semua kolom lama saat map
        batched=False,
    )
        
    logger.info(f"Formatted {len(formatted)} samples from {source}")
    return formatted