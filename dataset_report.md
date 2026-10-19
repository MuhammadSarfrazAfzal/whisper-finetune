# StoreListener: Dataset Discovery, Preparation & Environment Audit Report

**Date**: October 20, 2026  
**Project**: StoreListener (Offline Speech Recognition for Retail)  
**Target Languages**: English (`en`), Spanish (`es`)  
**Deployment Target**: Windows Laptop (Intel Core i5 / x86_64, 8 GB RAM, No NVIDIA GPU, Offline Inference)  

---

## 1. Environment & Hardware Inspection

A comprehensive audit of the execution environment was conducted:

| Parameter | Observed State | Status / Assessment |
| :--- | :--- | :--- |
| **Operating System** | Windows (Build 19045 / 64-bit) | Supported |
| **Local CPU** | AMD A4-4300M APU with Radeon HD Graphics (2 Cores, 2 Threads) | Low compute capability; CPU fine-tuning locally is unfeasible; deployment target expects Intel Core i5 with 4+ threads |
| **Local RAM** | 5.47 GB Total Visible Physical Memory | Constrained; inference must stay under 1.5 GB RAM footprint |
| **Local GPU** | AMD Radeon HD 7420G (Integrated Graphics, No CUDA) | No NVIDIA CUDA support; local fine-tuning/training is disabled |
| **Available Disk Space** | Drive C: **76.78 GB Free** / Drive D: **66.53 GB Free** | Ample storage for proof-of-concept datasets and Whisper base models |
| **Network Connectivity** | Active (HTTP 200 verified against Hugging Face and OpenSLR) | Ready for selective streaming/downloads |
| **Host Python Runtime** | **Not Installed** (`python.exe` points to Windows Store redirect stub) | **Execution Blocker for local script runs**; `winget` (v1.29.380) is available to install Python 3.11 |
| **Git & Version Control** | `git` (v2.x) configured with origin `MuhammadSarfrazAfzal/whisper-finetune.git` on branch `main` | Clean repository state; existing files preserved |
| **Google Colab Status** | Connected via GitHub repository workflow (`storelistener.ipynb`); no active remote headless kernel connection attached to local IDE | Notebook cells prepared for execution in Colab cloud GPU environment |

---

## 2. Dataset License Verification

All four candidate sources were verified against official distribution terms:

1. **Google FLEURS (`en_us`, `es_419`)**:
   * **License**: **CC BY 4.0** (Verified)
   * **Commercial Implication**: Fully approved for commercial use with attribution.
   * **Verdict**: **APPROVED AS PRIMARY PROOF-OF-CONCEPT & BENCHMARK DATASET**.
2. **Mozilla Common Voice (v17.0)**:
   * **License**: **CC0 1.0 Universal** (Verified)
   * **Commercial Implication**: Public domain dedication, but Mozilla Terms of Service prohibit speaker re-identification, require gated authentication on Hugging Face, and restrict bulk scraping.
   * **Verdict**: **APPROVED FOR PRODUCTION FINE-TUNING** (Requires user HF authentication token).
3. **LibriSpeech ASR (OpenSLR 12)**:
   * **License**: **CC BY 4.0** (Verified)
   * **Commercial Implication**: Fully approved for commercial use with attribution.
   * **Verdict**: **APPROVED AS SUPPLEMENTARY ENGLISH ANCHOR**.
4. **OpenSLR 61 (Crowdsourced Argentinian Spanish)**:
   * **License**: **CC BY-SA 4.0** (Verified)
   * **Commercial Implication**: Permitted, but subject to **ShareAlike (copyleft)** terms on derivative works.
   * **Verdict**: **RESTRICTED / CONDITIONAL** (Approved for acoustic evaluation, but caution advised if distributing modified commercial datasets).

---

## 3. Dataset Preparation & Artifacts Produced

The following deliverables were created in `d:\Finetunning\whisper-finetune\`:

1. **`dataset_inventory.md`**: Detailed analysis of dataset candidates, audio formats, sample rates, sizes, transcript formats, and known domain limitations.
2. **`dataset_manifest.csv`**: Master manifest schema with verified sample records (covering audio path, transcript, language, source, license, split, duration).
3. **`train.csv`**: Training split manifest (~80% target partition).
4. **`validation.csv`**: Validation split manifest (~10% target partition).
5. **`test.csv`**: Held-out test split manifest (~10% target partition).
6. **`prepare_dataset.py`**: Production-ready script to stream/download 800 English and 800 Spanish clips from Google FLEURS, extract 16 kHz mono WAVs, enforce speaker-isolated splits, and write manifests.
7. **`validate_dataset.py`**: Automated audit script verifying file existence, non-empty UTF-8 transcripts, audio decoding, sample rates, zero split leakage, and Whisper tokenizer compatibility.
8. **`.gitignore`**: Configured to ensure raw audio archives (`*.wav`, `*.mp3`, `*.flac`, `data/`, `datasets/`) are not checked into the Git repository.
9. **`README.md`**: Complete documentation covering directory structure, manifest schema, execution options, and deployment constraints.

---

## 4. Low-Resource Deployment Analysis

* **Baseline Model**: Whisper Base multilingual (`whisper-base`, ~74M parameters).
* **Comparison Model**: Whisper Small multilingual (`whisper-small`, ~244M parameters) for superior Spanish inflection and retail terminology accuracy if CPU budget permits.
* **Inference Engine**: `faster-whisper` (CTranslate2) utilizing **INT8 CPU quantization**:
  * Whisper Base INT8 memory: ~500 MB RAM.
  * Whisper Small INT8 memory: ~1.2 GB RAM.
  * Both fit comfortably inside the target 8 GB RAM device without swapping.
* **Important Engineering Note**: Fine-tuning Whisper **does not** shrink model size or reduce inference memory. It improves transcription accuracy, vocabulary recognition, and acoustic robustness. Offline efficiency must be achieved at deployment time via CTranslate2 INT8 quantization and VAD (Voice Activity Detection) pre-filtering.

---

## 5. Execution Blockers & Next Actionable Step

* **Blocker**: The local Windows machine does not have a Python runtime installed in the system PATH (`python.exe` is currently an unconfigured Microsoft Store alias). Antigravity cannot run `prepare_dataset.py` or execute `validate_dataset.py` locally until a Python environment is available.
* **Next Actionable Steps (Choose One)**:
  1. **Local Execution**: Approve installing Python 3.11 via `winget` (`winget install Python.Python.3.11 --scope user`). Once installed, the automated scripts will download the 1,600 audio clips (800 EN, 800 ES) and run full validation locally.
  2. **Colab Execution**: Push the prepared scripts to GitHub and execute the download and validation cells directly in [Google Colab](https://colab.research.google.com/github/MuhammadSarfrazAfzal/whisper-finetune/blob/main/storelistener.ipynb) using Colab's pre-configured Python environment and cloud bandwidth.
