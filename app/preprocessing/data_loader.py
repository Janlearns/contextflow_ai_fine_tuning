from datasets import load_dataset, Dataset
from app.utils.logger import logger
from app.utils.config import config
import os


def load_dolly_15k() -> Dataset:
    logger.info("Loading Dolly 15K dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    logger.info(f"Dolly 15K loaded: {len(dataset)} samples")
    return dataset


def load_alpaca() -> Dataset:
    logger.info("Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    logger.info(f"Alpaca loaded: {len(dataset)} samples")
    return dataset


def load_openassistant() -> Dataset:
    """
    Load OpenAssistant oasst1 dan pair prompter→assistant.

    oasst1 adalah tree-based dataset: tiap row adalah satu message.
    Untuk bisa dipakai fine-tuning, kita perlu pasangkan:
      - message dengan role='prompter' sebagai instruction (pertanyaan)
      - reply langsung (child) dengan role='assistant' sebagai response (jawaban)

    Cara pair:
      1. Buat mapping message_id → message untuk lookup cepat
      2. Untuk setiap message assistant, cari parent-nya (prompter)
      3. Simpan sebagai pasangan {instruction, response}
    """
    logger.info("Loading OpenAssistant dataset...")
    raw = load_dataset("OpenAssistant/oasst1", split="train")  # load raw dataset
    logger.info(f"OpenAssistant raw loaded: {len(raw)} messages")

    # Buat dict mapping message_id → full message untuk lookup parent
    id_to_msg = {msg["message_id"]: msg for msg in raw}  # key: message_id, value: dict message

    rows = []  # tampung hasil pair

    for msg in raw:  # iterasi semua message
        # Kita hanya proses message dari assistant (jawaban)
        if msg["role"] != "assistant":
            continue  # skip message yang bukan assistant

        parent_id = msg.get("parent_id")  # ambil ID parent (prompter)
        if not parent_id:
            continue  # skip kalau tidak punya parent (root message)

        parent = id_to_msg.get(parent_id)  # cari parent message
        if not parent:
            continue  # skip kalau parent tidak ditemukan

        # Pastikan parent-nya adalah prompter (pertanyaan manusia)
        if parent["role"] != "prompter":
            continue  # skip kalau parent bukan prompter (nested reply)

        # Ambil teks instruction dari prompter dan response dari assistant
        instruction = parent.get("text", "") or ""  # teks pertanyaan
        response = msg.get("text", "") or ""         # teks jawaban

        # Filter: skip kalau salah satu kosong
        if not instruction.strip() or not response.strip():
            continue

        rows.append({
            "instruction": instruction,  # pertanyaan dari prompter
            "response": response,        # jawaban dari assistant
        })

    dataset = Dataset.from_list(rows)  # konversi list of dict ke HuggingFace Dataset
    logger.info(f"OpenAssistant loaded & paired: {len(dataset)} Q&A pairs")
    return dataset


def load_coqa() -> Dataset:
    """
    Load CoQA dan flatten dari format story+lists menjadi per Q&A pair.

    CoQA raw: satu row = satu story + list pertanyaan + list jawaban.
    Setelah flatten: satu row = satu pasangan {question, context, answer}.
    """
    logger.info("Loading CoQA dataset...")
    raw = load_dataset("stanfordnlp/coqa", split="train")  # load raw CoQA

    rows = []  # tampung hasil flatten
    for sample in raw:  # iterasi tiap story
        story = sample["story"]                          # teks cerita sebagai context
        questions = sample["questions"]                  # list pertanyaan
        answer_texts = sample["answers"]["input_text"]   # list jawaban

        # Pair tiap pertanyaan dengan jawabannya
        for q, a in zip(questions, answer_texts):  # zip supaya index sejajar
            rows.append({
                "question": q,        # satu pertanyaan
                "context": story,     # story sebagai context
                "answer": a,          # satu jawaban
            })

    dataset = Dataset.from_list(rows)  # konversi ke HuggingFace Dataset
    logger.info(f"CoQA loaded & flattened: {len(dataset)} samples")
    return dataset


def load_all_datasets() -> dict:
    """Load semua dataset eksternal + company data dan return sebagai dict."""
    datasets = {}

    loaders = {
        "dolly": load_dolly_15k,
        "alpaca": load_alpaca,
        "openassistant": load_openassistant,  # sudah di-pair prompter→assistant
        "coqa": load_coqa,                    # sudah di-flatten per Q&A pair
    }

    for name, loader in loaders.items():  # iterasi tiap loader
        try:
            datasets[name] = loader()
        except Exception as e:
            logger.warning(f"Gagal load dataset '{name}': {e}")

    # Load company data (JSON, CSV, PDF, DOCX, TXT)
    try:
        from app.preprocessing.company_loader import load_company_data
        company_ds = load_company_data()
        if company_ds is not None and len(company_ds) > 0:
            datasets["company"] = company_ds
            logger.info(f"Company data loaded: {len(company_ds)} samples")
    except Exception as e:
        logger.warning(f"Gagal load company data: {e}")

    if not datasets:
        raise RuntimeError("Tidak ada dataset yang berhasil diload!")

    return datasets


def save_raw_datasets(datasets: dict):
    """Simpan dataset mentah ke disk."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    for name, ds in datasets.items():  # iterasi tiap dataset
        path = os.path.join(config.RAW_DATA_DIR, name)
        ds.save_to_disk(path)          # simpan ke disk
        logger.info(f"Saved {name} to {path}")