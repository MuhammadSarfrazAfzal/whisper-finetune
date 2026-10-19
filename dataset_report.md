# StoreListener: Dataset Discovery, Preparation & Environment Audit Report

**Date**: October 20, 2026  
**Project**: StoreListener (Offline Speech Recognition for Retail)  
**Target Languages**: English (`en`), Spanish (`es`)  
**Deployment Target**: Windows Laptop (Intel Core i5 / x86_64, 8 GB RAM, No NVIDIA GPU, Offline Inference)  

---

## 1. Environment & Hardware Execution Summary

| Parameter | Observed State | Status / Assessment |
| :--- | :--- | :--- |
| **Operating System** | Windows (Build 19045 / 64-bit) | Supported |
| **Local Host CPU** | AMD A4-4300M APU with Radeon HD Graphics (2 Cores, 2 Threads) | Low compute capability; CPU fine-tuning locally is unfeasible; deployment target expects Intel Core i5 with 4+ threads |
| **Local Host RAM** | 5.47 GB Total Visible Physical Memory | Constrained; inference must stay under 1.5 GB RAM footprint |
| **Local Host GPU** | AMD Radeon HD 7420G (Integrated Graphics, No CUDA) | No NVIDIA CUDA support; local model training is disabled |
| **Available Disk Space** | Drive C: **76.78 GB Free** / Drive D: **66.1 GB Free** | Ample storage for audio files and Whisper base models |
| **Host Python Runtime** | Python 3.15.0 installed | Direct local script execution enabled |
| **Google Colab Status** | Connected via GitHub repository workflow (`storelistener.ipynb`) | Workflow synchronized to repository |

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

## 3. Dataset Preparation Results (600 Clips Extracted)

A balanced proof-of-concept dataset was streamed, extracted, and structured from Google FLEURS:

| Partition | Total Clips | English (`en_us`) | Spanish (`es_419`) | Total Duration | Duration Range | Avg. Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **488** | 244 | 244 | 5,241.82s (~1.46 hrs) | [2.70s – 30.60s] | 10.74s |
| **Validation** | **46** | 23 | 23 | 450.66s (~0.13 hrs) | [3.96s – 17.04s] | 9.80s |
| **Test** | **66** | 33 | 33 | 672.86s (~0.19 hrs) | [3.84s – 24.18s] | 10.19s |
| **Total Master** | **600** | **300** | **300** | **6,365.34s (~1.77 hrs)** | **[2.70s – 30.60s]** | **10.61s** |

* **Audio Format**: Native 16 kHz Mono WAV (16-bit PCM), directly matching Whisper input.
* **Storage on Disk**: ~410 MB under `datasets/audio/{en,es}/` (excluded from git via `.gitignore`).
* **Transcript Normalization**: Verbatim UTF-8 with full punctuation and natural casing.
* **Split Isolation**: Grouped strictly by sentence/speaker ID. **ZERO cross-split data leakage** between train, validation, and test.

---

## 4. Quality Audit Verification (`validate_dataset.py`)

Execution output:
```text
======================================================================
STORELISTENER ASR DATASET VALIDATION AUDIT
======================================================================
[*] Validating split: train (train.csv)...
  [+] Split check PASSED (488 clips, 1.456 hrs, 16000 Hz, Mono)
[*] Validating split: validation (validation.csv)...
  [+] Split check PASSED (46 clips, 0.125 hrs, 16000 Hz, Mono)
[*] Validating split: test (test.csv)...
  [+] Split check PASSED (66 clips, 0.187 hrs, 16000 Hz, Mono)
[+] Split isolation verified: ZERO data leakage detected across train, val, and test.
[+] Combined manifest verified: 600 total records.
======================================================================
OVERALL VALIDATION STATUS: PASSED
Total Validated Clips: 600
Total Audio Duration: 1.77 hours
Summary Report Written to: validation_summary.json
======================================================================
```

---

## 5. Artifacts Produced & Verified

1. **`datasets/audio/en/`**: 300 verified 16 kHz Mono WAV files.
2. **`datasets/audio/es/`**: 300 verified 16 kHz Mono WAV files.
3. **`dataset_manifest.csv`**: Master manifest (600 validated records).
4. **`train.csv`**: Training manifest (488 records).
5. **`validation.csv`**: Validation manifest (46 records).
6. **`test.csv`**: Held-out test manifest (66 records).
7. **`validate_dataset.py`**: Reusable dataset integrity and audit script.
8. **`prepare_dataset.py`**: Zero-dependency automated stream-extractor script.
9. **`validation_summary.json`**: Machine-readable validation audit metrics.
10. **`storelistener.ipynb`**: Synchronized Colab notebook ready for future cloud fine-tuning.
