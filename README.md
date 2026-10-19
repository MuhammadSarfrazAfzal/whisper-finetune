# StoreListener: ASR Dataset Discovery & Preparation

StoreListener is an offline speech transcription pipeline designed to run on resource-constrained hardware (Windows laptops with dual/quad-core CPUs, 8 GB RAM, and no NVIDIA GPU). This workspace contains the discovery, evaluation, curation, and validation tooling for preparing English and Spanish speech datasets to fine-tune OpenAI Whisper models.

---

## 1. Directory Structure

```text
d:/Finetunning/whisper-finetune/
├── .gitignore               # Excludes raw audio files, archives, and virtual environments
├── README.md                # Project overview, setup, and reproduction instructions
├── dataset_inventory.md     # In-depth evaluation of datasets, licenses, sizes, and suitability
├── dataset_report.md        # Audit of environment, hardware, licenses, and validation status
├── dataset_manifest.csv     # Master manifest of all validated audio-transcript pairs
├── train.csv                # Training split (~80%)
├── validation.csv           # Validation split (~10%)
├── test.csv                 # Held-out evaluation test split (~10%)
├── prepare_dataset.py       # Automated script to download, extract, and build manifests
├── validate_dataset.py      # Automated data integrity, decoding, and split audit script
└── storelistener.ipynb      # Jupyter notebook configured for Google Colab and local execution
```

---

## 2. Manifest Schema

All manifest CSVs adhere strictly to the following standardized schema:

| Column Header | Type | Description |
| :--- | :--- | :--- |
| `audio_path` | String | Relative path from project root to the audio clip (e.g. `datasets/audio/en/clip_001.wav`) |
| `transcript` | String | Verbatim UTF-8 reference text transcription |
| `language` | String | Target language code (`en` or `es`) |
| `dataset_source` | String | Provenance name (e.g. `Google FLEURS (en_us)`) |
| `license` | String | Verified license under which the clip was distributed (e.g. `CC BY 4.0`) |
| `split` | String | Designated dataset partition (`train`, `validation`, or `test`) |
| `duration_seconds` | Float | Clip duration in seconds |

---

## 3. Approved Initial Dataset Selection

For the initial proof-of-concept subset (target: ~500–1,000 English clips, ~500–1,000 Spanish clips):

* **English (`en`)**: **Google FLEURS (`en_us`)** — CC BY 4.0. Native 16 kHz Mono WAV audio matching Whisper's input specs.
* **Spanish (`es`)**: **Google FLEURS (`es_419`)** — CC BY 4.0. Native 16 kHz Mono WAV audio, Latin American Spanish with balanced gender distribution.
* **Supplementary English Anchor**: **LibriSpeech (`openslr.org/12`)** — CC BY 4.0.
* **Acoustic Diversity & Retail Robustness (Phase 2 Fine-Tuning)**: **Mozilla Common Voice 17.0** — CC0 1.0 (subject to Mozilla Terms of Use and Hugging Face authentication).

---

## 4. How to Execute & Reproduce

### Option A: Local Windows Execution (Once Python is installed)

1. **Install Python 3.11** (if not already installed):
   ```powershell
   winget install Python.Python.3.11 --scope user
   ```
2. **Install minimal dependencies**:
   ```powershell
   pip install datasets soundfile
   ```
3. **Download and prepare the subset**:
   ```powershell
   # Downloads 800 English and 800 Spanish clips, extracts to datasets/audio/, generates manifests
   python prepare_dataset.py --count 800
   ```
4. **Run the quality check**:
   ```powershell
   python validate_dataset.py --manifest dataset_manifest.csv --train train.csv --val validation.csv --test test.csv
   ```

### Option B: Google Colab Execution

1. Open `storelistener.ipynb` directly in [Google Colab](https://colab.research.google.com/github/MuhammadSarfrazAfzal/whisper-finetune/blob/main/storelistener.ipynb).
2. Run the preparation cells:
   ```python
   !pip install -q datasets soundfile transformers
   !python prepare_dataset.py --count 800
   !python validate_dataset.py
   ```
3. Git commit or download the generated manifests (`train.csv`, `validation.csv`, `test.csv`).

---

## 5. Deployment Constraints (Target Device)

* **Deployment Device**: Windows laptop, Intel Core i5 CPU (or AMD APU), 8 GB RAM, no dedicated GPU.
* **Runtime Strategy**:
  * Whisper Base multilingual (~74M parameters) or Whisper Small multilingual (~244M parameters).
  * Inference engine: `faster-whisper` (CTranslate2) utilizing **INT8 CPU quantization**.
  * Memory footprint: ~500 MB RAM for Whisper Base INT8; ~1.2 GB RAM for Whisper Small INT8.
  * Real-time factor (RTF): ~0.3–0.6x on modern quad-core Intel/AMD CPUs, enabling fluid offline transcription.
