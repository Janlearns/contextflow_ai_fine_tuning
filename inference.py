"""
Entry point untuk inference ContextFlow AI.
Jalankan: python inference.py
"""
from app.inference.predictor import ContextFlowPredictor
from app.utils.logger import logger


def main():
    predictor = ContextFlowPredictor()

    examples = [
        {
            "instruction": "Apa prosedur pengajuan cuti tahunan?",
            "context": "Karyawan harus mengisi form HR-01 minimal 3 hari kerja sebelum cuti.",
        },
        {
            "instruction": "Bagaimana cara reset password sistem internal?",
            "context": "",
        },
    ]

    for ex in examples:
        logger.info(f"Instruction: {ex['instruction']}")
        response = predictor.predict(ex["instruction"], ex.get("context", ""))
        logger.info(f"Response: {response}")
        print("-" * 60)


if __name__ == "__main__":
    main()