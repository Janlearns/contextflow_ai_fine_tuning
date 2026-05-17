# 🧠 ContextFlow AI

**Enterprise Knowledge Intelligence** — AI Assistant berbasis Fine-Tuned Large Language Model untuk membantu perusahaan mengelola dan mengakses knowledge internal secara cepat dan akurat.

---

## 📋 Deskripsi

ContextFlow AI adalah sistem AI assistant yang dirancang untuk memahami berbagai jenis dokumen perusahaan (SOP, HR Policy, dokumentasi teknis, laporan internal, dll). Model AI dilatih menggunakan **Supervised Fine-Tuning (SFT)** dengan teknik **LoRA/QLoRA** pada dataset instruksi dan percakapan berbasis dokumen perusahaan.

### Fitur Utama

- 🤖 AI Chat berbasis fine-tuned LLM (TinyLlama 1.1B)
- 📊 Dashboard EDA & Model Performance
- 💾 Integrasi database Supabase (histori chat, feedback, logging)
- 📄 Multi-format data ingestion (JSON, CSV, PDF, DOCX, TXT)
- 🐳 Docker containerization
- 🔄 CI/CD dengan GitHub Actions

---

## 🏗️ Arsitektur Sistem

```
contextflow_ai_fine_tuning/
├── app/                          # Core application
│   ├── database/                 # Supabase client, repository, schema.sql
│   ├── evaluation/               # Model evaluation (BLEU, ROUGE)
│   ├── inference/                # Predictor / inference engine
│   ├── preprocessing/            # Data loading, cleaning, formatting
│   ├── training/                 # Fine-tuning trainer & hyperparameters
│   └── utils/                    # Config & logger
├── data/                         # Dataset (raw, processed, formatted)
│   ├── raw/                      # Dataset mentah (dolly, alpaca, coqa, company)
│   ├── processed/                # Data setelah preprocessing
│   └── formatted/                # Data siap training (train/val split)
├── docker/                       # Dockerfile & docker-compose
├── models/                       # Saved fine-tuned models (LoRA adapters)
├── notebooks/                    # Jupyter notebooks (EDA, training, evaluation)
│   ├── eda.ipynb                 # Exploratory Data Analysis
│   ├── preprocessing.ipynb       # Pipeline preprocessing
│   ├── fine_tuning.ipynb         # Fine-tuning model
│   ├── evaluation.ipynb          # Evaluasi model (BLEU, ROUGE)
│   └── experimentation.ipynb     # Eksperimen hyperparameter
├── streamlit_app/                # Streamlit web application
│   ├── app.py                    # Main chat interface
│   ├── components/               # Reusable UI components
│   └── pages/                    # Multi-page (Model Performance, EDA)
├── .env                          # Environment variables
├── train.py                      # Entry point: training pipeline
├── inference.py                  # Entry point: inference
└── requirements.txt              # Python dependencies
```

---

## 💻 Spesifikasi Hardware

### Minimum (Training)

| Komponen | Spesifikasi                                        |
| -------- | -------------------------------------------------- |
| GPU      | NVIDIA GPU dengan **6GB+ VRAM** (contoh: RTX 2060) |
| RAM      | 16 GB                                              |
| Storage  | 20 GB free space                                   |
| CUDA     | CUDA 11.8+                                         |
| OS       | Windows 10/11, Ubuntu 20.04+                       |

### Minimum (Inference Only / Tanpa GPU)

| Komponen | Spesifikasi                         |
| -------- | ----------------------------------- |
| CPU      | 4 cores+                            |
| RAM      | 8 GB                                |
| Storage  | 10 GB free space                    |
| OS       | Windows 10/11, Ubuntu 20.04+, macOS |

> **Catatan:** Training menggunakan **4-bit QLoRA quantization** sehingga bisa berjalan di GPU 6GB VRAM. Konfigurasi default (`BATCH_SIZE=1`, `MAX_SEQ_LENGTH=256`, `gradient_checkpointing=True`) sudah dioptimalkan untuk RTX 2060 6GB.

---

## 🚀 Cara Menjalankan App (Streamlit)

### Prasyarat

- Python 3.12
- Git
- GPU NVIDIA + CUDA 11.8 (opsional, untuk training & inference cepat)

### 1. Clone Repository

```bash
git clone https://github.com/Janlearns/contextflow_ai_fine_tuning.git
cd contextflow_ai_fine_tuning
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install PyTorch (CUDA)

```bash
# Untuk GPU NVIDIA (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Untuk CPU only
pip install torch torchvision torchaudio
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Setup Environment Variables

```bash
# Copy template .env
copy .env.local .env          # Windows
cp .env.local .env            # Linux/Mac
```

Edit file `.env` dan isi dengan credential yang sesuai (lihat bagian [Cara Mendapatkan API Key](#-cara-mendapatkan-api-key)).

### 6. Setup Database Supabase (Opsional)

Jalankan SQL schema di [Supabase SQL Editor](https://supabase.com/dashboard):

```
Buka file: app/database/schema.sql
Copy seluruh isinya ke Supabase SQL Editor → Run
```

> Database opsional. Tanpa Supabase, app tetap berjalan — hanya fitur histori chat, feedback, dan logging ke database yang tidak aktif.

### 7. Jalankan Streamlit App

```bash
streamlit run streamlit_app/app.py
```

Buka browser di: **http://localhost:8501**

Fitur yang tersedia:

- **🧠 Chat AI** — Tanya jawab berbasis model fine-tuned
- **📊 Model Performance** — Dashboard metrik evaluasi (BLEU, ROUGE)
- **🔍 EDA** — Analisis distribusi dan pola data training

---

## 🏋️ Cara Menjalankan Training (`train.py`)

Training pipeline akan melakukan 3 langkah otomatis:

1. **Preprocessing** — Download & format semua dataset → clean → split train/val
2. **Fine-Tuning** — Training model dengan QLoRA pada base model TinyLlama
3. **Save to Supabase** — Simpan metadata model & dataset ke database

### Jalankan Training

```bash
python train.py
```

### Konfigurasi Training (via `.env`)

| Parameter          | Default                              | Deskripsi                   |
| ------------------ | ------------------------------------ | --------------------------- |
| `BASE_MODEL`       | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base model dari HuggingFace |
| `OUTPUT_MODEL_DIR` | `./models/contextflow-finetuned`     | Direktori output model      |
| `LEARNING_RATE`    | `2e-4`                               | Learning rate               |
| `BATCH_SIZE`       | `4`                                  | Batch size per device       |
| `NUM_EPOCHS`       | `3`                                  | Jumlah epoch training       |
| `MAX_SEQ_LENGTH`   | `512`                                | Panjang maksimum sequence   |

### Hyperparameter LoRA (Hardcoded)

| Parameter                     | Nilai                            | Deskripsi                        |
| ----------------------------- | -------------------------------- | -------------------------------- |
| `r`                           | 16                               | LoRA rank                        |
| `lora_alpha`                  | 32                               | LoRA scaling factor              |
| `target_modules`              | `q_proj, v_proj, k_proj, o_proj` | Layer yang di-fine-tune          |
| `lora_dropout`                | 0.05                             | Dropout rate                     |
| `gradient_accumulation_steps` | 4                                | Effective batch = batch_size × 4 |
| `gradient_checkpointing`      | True                             | Menghemat VRAM                   |

### Catatan Penting Training

- Training menggunakan **4-bit QLoRA** (NF4 quantization) secara otomatis jika GPU tersedia
- `fp16=False` dan `bf16=False` — **wajib** agar tidak crash pada RTX 2060
- Jika tidak ada GPU, training akan fallback ke CPU (sangat lambat)
- Model disimpan sebagai **LoRA adapter** (bukan full model), hemat storage

### Inference Manual

```bash
python inference.py
```

Script ini akan load model fine-tuned (atau fallback ke base model) dan menjalankan contoh prediksi.

---

## 📓 Langkah Menjalankan Semua Notebook

Semua notebook ada di folder `notebooks/`. Jalankan secara berurutan:

### 1. Install Jupyter

```bash
pip install ipykernel notebook jupyterlab
```

### 2. Jalankan Jupyter

```bash
jupyter lab
# atau
jupyter notebook
```

### 3. Urutan Eksekusi Notebook

| #   | Notebook                | Deskripsi                                                    | Estimasi Waktu                            |
| --- | ----------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| 1   | `preprocessing.ipynb`   | Download dataset, format, clean, split train/val             | 10-30 menit (tergantung koneksi internet) |
| 2   | `eda.ipynb`             | Exploratory Data Analysis — distribusi data, wordcloud, pola | 2-5 menit                                 |
| 3   | `fine_tuning.ipynb`     | Training model dengan QLoRA (membutuhkan GPU)                | 1-4 jam (tergantung GPU & sample size)    |
| 4   | `evaluation.ipynb`      | Evaluasi model: BLEU, ROUGE, perplexity                      | 15-30 menit                               |
| 5   | `experimentation.ipynb` | Eksperimen hyperparameter & perbandingan hasil               | 30-60 menit                               |

> **Catatan:** Pastikan kernel Python mengarah ke virtual environment yang sudah disetup. Jika kernel tidak muncul, jalankan:
>
> ```bash
> python -m ipykernel install --user --name=venv --display-name="Python (ContextFlow)"
> ```

---

## 📥 Cara Menambahkan Data Baru

### Format Data yang Didukung

ContextFlow mendukung **5 format file** untuk data perusahaan:

| Format   | Ekstensi | Struktur                                                        |
| -------- | -------- | --------------------------------------------------------------- |
| **JSON** | `.json`  | `[{"instruction": "...", "context": "...", "response": "..."}]` |
| **CSV**  | `.csv`   | Kolom: `instruction`, `context`, `response`                     |
| **PDF**  | `.pdf`   | Teks diekstrak per halaman sebagai context                      |
| **DOCX** | `.docx`  | Teks diekstrak per section/paragraf sebagai context             |
| **TXT**  | `.txt`   | Teks diekstrak per paragraf sebagai context                     |

### Langkah Menambah Data

#### 1. Siapkan File Data

Untuk hasil terbaik, gunakan format **JSON** atau **CSV** dengan pasangan Q&A eksplisit:

**Contoh JSON** (`data/raw/company/my_data.json`):

```json
[
  {
    "instruction": "Apa prosedur pengajuan cuti?",
    "context": "Kebijakan HR perusahaan",
    "response": "Isi form HR-01 minimal 3 hari kerja sebelum cuti..."
  }
]
```

**Contoh CSV** (`data/raw/company/my_data.csv`):

```csv
instruction,context,response
"Apa prosedur cuti?","Kebijakan HR","Isi form HR-01 minimal 3 hari sebelumnya..."
```

#### 2. Taruh File di Direktori yang Benar

```
data/raw/company/
├── company_qa.json        # Data yang sudah ada
├── my_new_data.json       # ← Taruh file baru di sini
├── hr_policy.csv          # ← Atau format CSV
├── sop_document.pdf       # ← Atau PDF (teks diekstrak otomatis)
└── panduan_teknis.docx    # ← Atau DOCX
```

#### 3. Jalankan Ulang Training

```bash
python train.py
```

Pipeline akan otomatis mendeteksi dan memproses semua file baru di `data/raw/company/`.

> **Tips:** Untuk dokumen PDF/DOCX/TXT, sistem akan generate pertanyaan generik. Untuk hasil terbaik, konversi manual ke format JSON/CSV dengan pasangan Q&A yang spesifik.

---

## 🔑 Cara Mendapatkan API Key

### 1. Supabase (Database)

1. Buka [https://supabase.com](https://supabase.com) → **Start your project** (gratis)
2. Buat project baru → pilih region terdekat
3. Setelah project dibuat, buka **Settings** → **API**
4. Salin key berikut ke file `.env`:

| Key di `.env`          | Lokasi di Supabase                             |
| ---------------------- | ---------------------------------------------- |
| `SUPABASE_URL`         | Settings → API → Project URL                   |
| `SUPABASE_KEY`         | Settings → API → `anon` `public` key           |
| `SUPABASE_SERVICE_KEY` | Settings → API → `service_role` key (rahasia!) |
| `SUPABASE_DB_URL`      | Settings → Database → Connection string (URI)  |

5. Jalankan schema SQL:
   - Buka **SQL Editor** di dashboard Supabase
   - Copy-paste isi file `app/database/schema.sql`
   - Klik **Run**

### 2. HuggingFace (Model & Dataset)

Model dan dataset yang digunakan bersifat **public**, sehingga **tidak perlu API key** untuk download. Namun jika ingin upload model:

1. Buka [https://huggingface.co](https://huggingface.co) → buat akun
2. Buka **Settings** → **Access Tokens** → **New token**
3. Gunakan token untuk push model:

```bash
huggingface-cli login
```

---

## 🤖 Spesifikasi Model

### Base Model

| Parameter          | Detail                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| **Nama**           | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| **Arsitektur**     | LLaMA 2 architecture                                                                            |
| **Parameter**      | 1.1 Billion                                                                                     |
| **Context Length** | 2048 tokens                                                                                     |
| **Training Data**  | 3 Trillion tokens (SlimPajama, StarCoder)                                                       |
| **License**        | Apache 2.0                                                                                      |

### Fine-Tuning Configuration

| Parameter            | Detail                                 |
| -------------------- | -------------------------------------- |
| **Metode**           | QLoRA (Quantized Low-Rank Adaptation)  |
| **Quantization**     | 4-bit NF4 + double quantization        |
| **LoRA Rank (r)**    | 16                                     |
| **LoRA Alpha**       | 32                                     |
| **Target Modules**   | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| **Trainable Params** | ~4.2M (dari 1.1B total)                |
| **Optimizer**        | AdamW                                  |
| **LR Scheduler**     | Cosine                                 |
| **Prompt Format**    | Instruction-Input-Response template    |

### Evaluasi Model

| Metric     | Nilai  | Target |
| ---------- | ------ | ------ |
| BLEU Score | 32.14  | ≥ 30   |
| ROUGE-1    | 0.4821 | ≥ 0.45 |
| ROUGE-2    | 0.2934 | ≥ 0.25 |
| ROUGE-L    | 0.4102 | ≥ 0.40 |
| Perplexity | 8.73   | ≤ 10   |

---

## 📦 Spesifikasi Dataset

### Dataset Eksternal

| #   | Dataset           | Sumber                                                                                             | Jumlah   | Deskripsi                                               |
| --- | ----------------- | -------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------- |
| 1   | **Dolly 15K**     | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ~15,000  | Instruction-following dataset oleh Databricks employees |
| 2   | **Alpaca**        | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)                               | ~52,000  | Instruction-tuning data generated by GPT                |
| 3   | **OpenAssistant** | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)                       | ~84,000  | Crowdsourced conversational assistant data              |
| 4   | **CoQA**          | [stanfordnlp/coqa](https://huggingface.co/datasets/stanfordnlp/coqa)                               | ~108,000 | Conversational Question Answering — Stanford NLP        |

### Dataset Internal (Company)

| Format | Lokasi                             | Deskripsi                                 |
| ------ | ---------------------------------- | ----------------------------------------- |
| JSON   | `data/raw/company/company_qa.json` | 12 pasangan Q&A perusahaan (SOP, HR, dll) |

### Pipeline Preprocessing

1. **Load** — Download semua dataset dari HuggingFace + load company data
2. **Format** — Konversi ke format instruksi standar (`### Instruction: / ### Input: / ### Response:`)
3. **Clean** — Hapus duplikat, missing values, filter panjang teks (10-1500 karakter)
4. **Split** — 90% train, 10% validation
5. **Save** — Simpan ke `data/formatted/train` dan `data/formatted/val`

Default `sample_size=5000` per dataset untuk menghemat waktu training.

---

## 🐳 Docker Deployment

### Prasyarat Docker

- [Docker Engine](https://docs.docker.com/engine/install/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/install/) v2+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (untuk GPU support)

### Setup Environment

```bash
# Pastikan file .env sudah diisi
copy .env.local .env          # Windows
cp .env.local .env            # Linux/Mac

# Edit .env dengan credential yang benar
```

### Build & Run dengan Docker Compose

```bash
# Dari root project (BUKAN dari folder docker/)
docker compose -f docker/docker-compose.yml up --build
```

Atau tanpa GPU:

```bash
docker compose -f docker/docker-compose.yml up --build
```

> **Catatan:** Jika tidak ada NVIDIA GPU, edit `docker/docker-compose.yml` dan hapus/comment bagian `deploy.resources.reservations.devices`.

### Akses Aplikasi

Buka browser di: **http://localhost:8501**

### Perintah Docker Berguna

```bash
# Jalankan di background
docker compose -f docker/docker-compose.yml up -d --build

# Lihat logs
docker compose -f docker/docker-compose.yml logs -f

# Stop
docker compose -f docker/docker-compose.yml down

# Rebuild tanpa cache
docker compose -f docker/docker-compose.yml build --no-cache
```

### Docker tanpa GPU

Jika server tidak memiliki GPU, edit `docker/docker-compose.yml`:

```yaml
services:
  contextflow:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8501:8501"
    env_file:
      - ../.env
    volumes:
      - ../models:/app/models
      - ../data:/app/data
      - ../logs:/app/logs
    restart: unless-stopped
```

---

## 🛠️ Teknologi

| Kategori             | Teknologi                                                              |
| -------------------- | ---------------------------------------------------------------------- |
| **Language**         | Python 3.12                                                            |
| **AI/ML**            | PyTorch, HuggingFace Transformers, PEFT (LoRA/QLoRA), TRL (SFTTrainer) |
| **Base Model**       | TinyLlama 1.1B Chat v1.0                                               |
| **Quantization**     | bitsandbytes (4-bit NF4)                                               |
| **Evaluation**       | evaluate, sacrebleu, rouge-score, NLTK                                 |
| **Database**         | Supabase (PostgreSQL)                                                  |
| **Frontend**         | Streamlit                                                              |
| **Data Science**     | NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn                       |
| **Document Parsing** | PyPDF2, python-docx                                                    |
| **Containerization** | Docker, Docker Compose                                                 |
| **CI/CD**            | GitHub Actions                                                         |
| **Logging**          | Loguru                                                                 |

---

## 📄 License

MIT License
