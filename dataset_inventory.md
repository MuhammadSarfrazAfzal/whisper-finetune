# StoreListener: Open-Source Speech Recognition Dataset Inventory

This document evaluates candidate open-source Automatic Speech Recognition (ASR) datasets for fine-tuning OpenAI Whisper models to support **English** and **Spanish** speech recognition in the **StoreListener** offline retail transcription application.

---

## 1. Executive Summary & Selection Matrix

| Dataset | Languages | Verified License | Commercial Use Permitted? | Native Sample Rate | Access / Distribution | Recommended Role in StoreListener |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Google FLEURS** | English (`en_us`), Spanish (`es_419`) | **CC BY 4.0** (Verified) | **Yes** (Attribution required) | 16 kHz Mono WAV | Open API & Direct HTTP (HF / GCS) | **Primary Proof-of-Concept & Evaluation Benchmark** |
| **Mozilla Common Voice** (v17.0) | English (`en`), Spanish (`es`) | **CC0 1.0 Universal** (Verified) | **Yes** (Subject to Mozilla Terms: no speaker re-id) | 48 kHz Mono MP3 | Gated (Terms acceptance required on Mozilla / HF) | **Acoustic Diversity & Retail Robustness Training** |
| **LibriSpeech ASR** (SLR12) | English (`en`) | **CC BY 4.0** (Verified) | **Yes** (Attribution required) | 16 kHz Lossless FLAC | Open Direct Mirrors (OpenSLR) | **Supplementary Clean English Acoustic Anchor** |
| **OpenSLR 61** (Crowdsourced Spanish) | Spanish (`es-AR`) | **CC BY-SA 4.0** (Verified) | **Yes** (ShareAlike copyleft obligation) | 44.1 / 48 kHz WAV | Open Direct Mirrors (OpenSLR) | **Supplementary Spanish Dialectal Training (Targeted)** |

---

## 2. Detailed Dataset Candidate Evaluations

### 2.1 Google FLEURS (Few-shot Learning Evaluation of Universal Robust Speech)
* **Dataset Name & Source URL**: Google FLEURS — [https://huggingface.co/datasets/google/fleurs](https://huggingface.co/datasets/google/fleurs)
* **Supported Languages**: English (`en_us`), Spanish (`es_419` Latin America, `es_es` Spain), and 100 other languages.
* **License & Commercial Implications**: 
  * **License**: Creative Commons Attribution 4.0 International (**CC BY 4.0**).
  * **Commercial Use**: Fully permitted for commercial use, subject only to reasonable attribution to Google Research / Conneau et al. No viral copyleft restrictions.
* **Approximate Download Size**:
  * English (`en_us`): ~380 MB compressed (~10.5 hours)
  * Spanish (`es_419`): ~350 MB compressed (~10.2 hours)
  * Selective download / streaming supported without downloading the entire 102-language corpus (~35 GB).
* **Audio Format & Sample Rate**: **16,000 Hz (16 kHz) Mono WAV**, 16-bit PCM.
* **Transcript Format**: Structured JSON / Parquet metadata containing:
  * `id`: Utterance identifier
  * `audio`: Audio file path and raw byte payload
  * `transcription`: Fully punctuated and cased reference transcript (UTF-8)
  * `raw_transcription`: Unnormalized transcript
  * `gender`: Speaker gender metadata (balanced male/female)
  * `lang_id`: Language identifier code
* **Number of Available Examples**:
  * English (`en_us`): ~3,009 recordings (~2,009 train, ~394 validation, ~606 test)
  * Spanish (`es_419`): ~3,034 recordings (~2,034 train, ~400 validation, ~600 test)
* **Suitability for Whisper Fine-Tuning**: **EXCELLENT (Highest Priority)**.
  1. *Acoustic Alignment*: Native 16 kHz mono WAV matches Whisper's mel-spectrogram input processing directly with zero audio resampling or transcoding artifacts.
  2. *Transcript Alignment*: Transcripts contain full punctuation, numbers, and capitalization, matching Whisper's training objective.
  3. *Balanced Multilingual Alignment*: Parallel coverage across English and Spanish allows balanced training without dialectal skew.
  4. *Access*: Fully open and programmatically downloadable without gated API keys or authentication roadblocks.
* **Known Limitations**:
  * Read speech from translated Wikipedia sentences; vocabulary is slightly formal and lacks ambient shop background noise or spontaneous customer hesitations.

---

### 2.2 Mozilla Common Voice (Version 17.0 / Current Archive)
* **Dataset Name & Source URL**: Mozilla Common Voice — [https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets) / [https://huggingface.co/mozilla-foundation](https://huggingface.co/mozilla-foundation)
* **Supported Languages**: English (`en`), Spanish (`es`), and over 100 languages.
* **License & Commercial Implications**:
  * **License**: **CC0 1.0 Universal** (Public Domain Dedication) for audio and text content.
  * **Commercial Use**: Permitted, but strictly conditioned by Mozilla's Common Voice Terms of Service:
    1. Users must agree never to attempt re-identification of speakers or associate demographic data with individuals.
    2. Gated access on Hugging Face requires an authenticated token and individual user agreement.
    3. Bulk scraping without adhering to Mozilla Data Collective guidelines is prohibited.
* **Approximate Download Size**:
  * English (`en`): ~85 GB compressed (>3,400 hours)
  * Spanish (`es`): ~28 GB compressed (>1,100 hours)
  * Manifest/TSV indices: ~15–50 MB.
* **Audio Format & Sample Rate**: **48,000 Hz (48 kHz) Mono MP3** (MPEG-1 Layer III).
* **Transcript Format**: Tab-Separated Values (TSV) (`validated.tsv`, `train.tsv`, `dev.tsv`, `test.tsv`):
  * `client_id`: Hashed speaker identifier (critical for preventing speaker leakage across splits)
  * `path`: MP3 audio filename
  * `sentence`: Ground-truth sentence transcript (UTF-8)
  * `up_votes`, `down_votes`: Community validation metrics
  * `age`, `gender`, `accent`, `locale`: Self-reported speaker metadata
* **Number of Available Examples**:
  * English: >2,300,000 validated clips
  * Spanish: >450,000 validated clips
* **Suitability for Whisper Fine-Tuning**: **VERY HIGH (Primary ASR Corpus for Robustness)**.
  * Captures real-world microphone variability (smartphones, laptop mics, headsets), acoustic reverb, ambient room noise, and authentic conversational cadences.
  * Excellent for preparing StoreListener for non-studio conditions.
* **Known Limitations**:
  * *Compute / Resampling Overhead*: 48 kHz MP3 requires decoding and downsampling to 16 kHz for Whisper.
  * *Access Gate*: Cannot be downloaded anonymously in automated pipelines without a registered Hugging Face access token (`HF_TOKEN`) or Mozilla Data Collective credential.
  * *Audio Artifacts*: Heavy MP3 compression, occasional clipping, background laughter, and pronunciation mistakes in a small percentage of clips despite voting.

---

### 2.3 LibriSpeech ASR Corpus (OpenSLR 12)
* **Dataset Name & Source URL**: LibriSpeech ASR Corpus — [https://www.openslr.org/12](https://www.openslr.org/12)
* **Supported Language**: English (`en`)
* **License & Commercial Implications**:
  * **License**: Creative Commons Attribution 4.0 International (**CC BY 4.0**).
  * **Commercial Use**: Fully permitted for commercial use. Attribution required (Panayotov et al., ICASSP 2015).
* **Approximate Download Size**:
  * `dev-clean.tar.gz`: 337 MB (~5.4 hours)
  * `test-clean.tar.gz`: 346 MB (~5.4 hours)
  * `train-clean-100.tar.gz`: 6.3 GB (~100 hours)
  * Full Corpus (1,000 hrs): ~60 GB
* **Audio Format & Sample Rate**: **16,000 Hz (16 kHz) Lossless FLAC**, 16-bit PCM mono.
* **Transcript Format**: Text files (`*.trans.txt`) per chapter directory formatted as:
  `<SPEAKER_ID>-<CHAPTER_ID>-<UTTERANCE_ID> <UPPERCASE_TRANSCRIPT>`
* **Number of Available Examples**:
  * `dev-clean`: 2,703 audio utterances (clean subset)
  * `train-clean-100`: 28,539 audio utterances
* **Suitability for Whisper Fine-Tuning**: **MODERATE TO HIGH (English Acoustic Baseline)**.
  * Pristine 16 kHz lossless audio provides a rock-solid acoustic anchor without resampling distortion.
  * Strict speaker separation (`speaker_id` embedded in file IDs) guarantees zero speaker leakage.
* **Known Limitations**:
  * *English Only*: Provides no Spanish speech data.
  * *Case and Punctuation*: Standard transcripts are uppercase only without commas, periods, or question marks. Fine-tuning Whisper solely on raw LibriSpeech transcripts can degrade Whisper's native punctuation capabilities unless matched against original Project Gutenberg texts.
  * *Domain Gap*: Audiobooks read aloud from 19th-century literature have little overlap with modern retail, storefront, or conversational dialogue.

---

### 2.4 OpenSLR 61 (Crowdsourced Argentinian Spanish Speech)
* **Dataset Name & Source URL**: OpenSLR 61 — [https://www.openslr.org/61/](https://www.openslr.org/61/)
* **Supported Language**: Spanish (`es-AR`)
* **License & Commercial Implications**:
  * **License**: Creative Commons Attribution-ShareAlike 4.0 International (**CC BY-SA 4.0**).
  * **Commercial Use Warning**: Commercial use is permitted, **BUT** the **ShareAlike (SA)** clause mandates that any derivative works or modified versions of the dataset that are publicly distributed must be licensed under the identical or compatible CC BY-SA license. For closed-source or proprietary enterprise deployments, CC BY-SA datasets should either be strictly isolated or avoided if derivative dataset distribution is anticipated.
* **Approximate Download Size**:
  * `es_weather_messages.zip`: 33 MB (~500 recordings)
  * `es_ar_male.zip`: 551 MB (~1,800 recordings)
  * `es_ar_female.zip`: 1.2 GB (~3,900 recordings)
* **Audio Format & Sample Rate**: WAV audio files at 44.1 kHz / 48 kHz, 16-bit PCM.
* **Transcript Format**: TSV file (`line_index_female.tsv`, `line_index_male.tsv`) containing `<FILE_ID>\t<TRANSCRIPT>`.
* **Number of Available Examples**: 5,739 recordings across native speakers.
* **Suitability for Whisper Fine-Tuning**: **MODERATE**.
  * High acoustic quality recorded on real mobile devices.
  * Good for testing dialectal robustness.
* **Known Limitations**:
  * Strongly dialect-specific (Argentinian Spanish / Rioplatense phonology: *yeísmo rehilado*, *voseo*).
  * CC BY-SA copyleft constraints require careful legal review for commercial product bundling.

---

## 3. Approved Initial Training Set Recommendation

Based on licensing verification, audio format alignment, and initial target constraints (500–1,000 clips per language):

1. **Primary Proof-of-Concept Set (Approved)**:
   * **English**: Google FLEURS (`en_us`) — 800 audio clips (CC BY 4.0)
   * **Spanish**: Google FLEURS (`es_419`) — 800 audio clips (CC BY 4.0)
   * *Rationale*: Matches Whisper native 16 kHz mono WAV input, verified CC BY 4.0 commercial license, identical collection methodology, pre-punctuated UTF-8 transcripts, and zero authentication gating.

2. **Secondary Supplement (Approved for Phase 2+ Fine-Tuning)**:
   * **English**: LibriSpeech (`dev-clean` / `train-clean-100` slice) (CC BY 4.0)
   * **Spanish & English**: Mozilla Common Voice 17.0 (CC0 + Mozilla Terms approval via user auth)
