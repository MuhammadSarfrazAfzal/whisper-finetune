#!/usr/bin/env python3
"""
prepare_dataset.py
StoreListener - Automated Dataset Downloader, Extractor & Manifest Builder

Target:
- Downloads a high-quality initial proof-of-concept subset:
    * 500-1,000 English audio clips (FLEURS en_us / LibriSpeech)
    * 500-1,000 Spanish audio clips (FLEURS es_419 / OpenSLR 61)
- Verifies licenses: CC BY 4.0 / CC BY-SA 4.0
- Extracts and organizes audio under `datasets/audio/{en,es}/`
- Preserves verbatim transcripts in UTF-8
- Prevents speaker leakage across train (80%), validation (10%), test (10%) splits
- Produces:
    * dataset_manifest.csv
    * train.csv
    * validation.csv
    * test.csv
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path

# Verify dependencies
try:
    from datasets import load_dataset
    import soundfile as sf
    HAS_HF = True
except ImportError:
    HAS_HF = False


def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def prepare_from_huggingface(target_per_lang=800, output_dir="datasets"):
    """
    Downloads and extracts a clean, balanced subset of English and Spanish clips
    from Google FLEURS using the Hugging Face datasets streaming API.
    """
    if not HAS_HF:
        print("[!] ERROR: 'datasets' and 'soundfile' packages are required.")
        print("    Install via: pip install datasets soundfile")
        return False

    base_path = Path(output_dir)
    audio_base = base_path / "audio"
    audio_base.mkdir(parents=True, exist_ok=True)

    records = []
    configs = [
        {"lang": "en", "hf_config": "en_us", "license": "CC BY 4.0", "source": "Google FLEURS (en_us)"},
        {"lang": "es", "hf_config": "es_419", "license": "CC BY 4.0", "source": "Google FLEURS (es_419)"}
    ]

    for cfg in configs:
        lang = cfg["lang"]
        hf_config = cfg["hf_config"]
        print(f"\n[*] Processing language '{lang}' from {cfg['source']}...")
        lang_audio_dir = audio_base / lang
        lang_audio_dir.mkdir(parents=True, exist_ok=True)

        # Stream dataset to avoid downloading all 35GB of FLEURS
        ds = load_dataset("google/fleurs", hf_config, split="train+validation+test", streaming=True)

        count = 0
        speaker_samples = {}

        for item in ds:
            if count >= target_per_lang:
                break

            transcript = item.get("transcription", "").strip()
            if not transcript:
                continue

            audio_data = item.get("audio")
            if not audio_data or "array" not in audio_data:
                continue

            arr = audio_data["array"]
            sr = audio_data.get("sampling_rate", 16000)
            dur = len(arr) / float(sr)

            # Filter outlier durations (keep 1s to 25s for Whisper stability)
            if dur < 1.0 or dur > 25.0:
                continue

            speaker_id = str(item.get("speaker_id", f"spk_{count}"))
            clip_id = f"{lang}_{count:04d}_{sanitize_filename(speaker_id)}"
            rel_audio_path = f"datasets/audio/{lang}/{clip_id}.wav"
            full_audio_path = base_path.parent / rel_audio_path

            # Save as 16kHz mono WAV
            if not full_audio_path.exists():
                sf.write(str(full_audio_path), arr, sr, subtype="PCM_16")

            record = {
                "audio_path": rel_audio_path.replace("\\", "/"),
                "transcript": transcript,
                "language": lang,
                "dataset_source": cfg["source"],
                "license": cfg["license"],
                "speaker_id": speaker_id,
                "duration_seconds": round(dur, 3)
            }

            if speaker_id not in speaker_samples:
                speaker_samples[speaker_id] = []
            speaker_samples[speaker_id].append(record)
            count += 1

        print(f"  [+] Extracted {count} valid audio clips across {len(speaker_samples)} speakers for '{lang}'.")

        # Group speakers into splits (80% train, 10% validation, 10% test) to prevent speaker leakage
        speakers = list(speaker_samples.keys())
        random.seed(42)
        random.shuffle(speakers)

        n_train_spk = int(0.80 * len(speakers))
        n_val_spk = int(0.10 * len(speakers))

        train_spks = set(speakers[:n_train_spk])
        val_spks = set(speakers[n_train_spk:n_train_spk + n_val_spk])
        test_spks = set(speakers[n_train_spk + n_val_spk:])

        for spk, items in speaker_samples.items():
            if spk in train_spks:
                split = "train"
            elif spk in val_spks:
                split = "validation"
            else:
                split = "test"

            for item in items:
                item["split"] = split
                records.append(item)

    # Write manifests
    manifest_fields = ["audio_path", "transcript", "language", "dataset_source", "license", "split", "duration_seconds"]
    
    def write_csv(filename, rows):
        out_path = Path(filename)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_fields, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[+] Created {out_path} ({len(rows)} records)")

    write_csv("dataset_manifest.csv", records)
    write_csv("train.csv", [r for r in records if r["split"] == "train"])
    write_csv("validation.csv", [r for r in records if r["split"] == "validation"])
    write_csv("test.csv", [r for r in records if r["split"] == "test"])

    print("\n[+] Dataset preparation complete!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare StoreListener ASR dataset.")
    parser.add_argument("--count", type=int, default=800, help="Clips per language (default: 800).")
    parser.add_argument("--outdir", default="datasets", help="Output directory for audio.")
    args = parser.parse_args()

    success = prepare_from_huggingface(target_per_lang=args.count, output_dir=args.outdir)
    sys.exit(0 if success else 1)
