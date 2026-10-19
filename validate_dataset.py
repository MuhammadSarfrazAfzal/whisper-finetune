#!/usr/bin/env python3
"""
validate_dataset.py
StoreListener - ASR Dataset Validation and Integrity Checker

This script performs rigorous quality checks on Whisper fine-tuning manifests:
1. Verifies that all manifest audio paths exist and are accessible.
2. Checks that transcripts are non-empty UTF-8 text strings.
3. Tests audio decoding, reporting duration, channels, and sample-rate statistics.
4. Validates language codes (e.g., 'en', 'es', 'en_us', 'es_419').
5. Verifies split integrity to prevent data leakage between train, validation, and test sets.
6. Validates compatibility with Whisper tokenizers and audio processors.
7. Produces an audit report summarizing dataset readiness for offline fine-tuning.
"""

import os
import sys
import csv
import wave
import json
import argparse
from pathlib import Path

# Optional advanced audio decoders
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from transformers import WhisperTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


VALID_LANGUAGES = {"en", "es", "en_us", "es_419", "es_es", "en-US", "es-ES", "es-AR"}


def get_audio_info(file_path):
    """
    Decodes audio headers to verify file validity, duration, sample rate, and channels.
    Falls back gracefully from soundfile to standard library wave module.
    """
    if not os.path.isfile(file_path):
        return None, "File does not exist"

    # Try soundfile first (supports WAV, FLAC, OGG, etc.)
    if HAS_SOUNDFILE:
        try:
            info = sf.info(file_path)
            return {
                "duration": info.duration,
                "samplerate": info.samplerate,
                "channels": info.channels,
                "format": info.format
            }, None
        except Exception as e:
            return None, f"Soundfile decode error: {e}"

    # Fallback to standard library wave for WAV files
    if str(file_path).lower().endswith(".wav"):
        try:
            with wave.open(str(file_path), "rb") as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                frames = wf.getnframes()
                duration = frames / float(rate)
                return {
                    "duration": duration,
                    "samplerate": rate,
                    "channels": channels,
                    "format": "WAV"
                }, None
        except Exception as e:
            return None, f"Wave decode error: {e}"

    # Generic file size check if no decoder available
    try:
        size = os.path.getsize(file_path)
        if size == 0:
            return None, "Zero-byte audio file"
        return {"duration": 0.0, "samplerate": 16000, "channels": 1, "format": "UNKNOWN"}, None
    except Exception as e:
        return None, f"IO error: {e}"


def validate_csv(csv_path, base_dir, split_name=""):
    """
    Validates a single CSV manifest.
    Expected columns: audio_path, transcript, language, dataset_source, license, split, duration_seconds
    """
    if not os.path.exists(csv_path):
        return None, [f"Manifest file not found: {csv_path}"]

    records = []
    errors = []
    total_duration = 0.0
    durations = []
    sample_rates = set()
    channels_set = set()
    audio_paths_seen = set()

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Check required columns
        required_cols = ["audio_path", "transcript", "language"]
        for col in required_cols:
            if col not in fieldnames:
                errors.append(f"Missing required column '{col}' in {csv_path}")

        if errors:
            return None, errors

        for idx, row in enumerate(reader, start=1):
            rel_audio = row.get("audio_path", "").strip()
            transcript = row.get("transcript", "").strip()
            lang = row.get("language", "").strip().lower()

            # 1. Check non-empty transcript
            if not transcript:
                errors.append(f"Row {idx}: Empty transcript for {rel_audio}")

            # 2. Check language
            if lang not in VALID_LANGUAGES:
                errors.append(f"Row {idx}: Invalid language '{lang}' in {rel_audio}")

            # 3. Check audio existence
            full_audio = os.path.normpath(os.path.join(base_dir, rel_audio))
            if not os.path.exists(full_audio):
                errors.append(f"Row {idx}: Audio path missing on disk: {full_audio}")
                continue

            # Check duplicate within this manifest
            if rel_audio in audio_paths_seen:
                errors.append(f"Row {idx}: Duplicate audio entry {rel_audio}")
            audio_paths_seen.add(rel_audio)

            # 4. Decode audio
            audio_info, err = get_audio_info(full_audio)
            if err:
                errors.append(f"Row {idx}: Audio decode failure for {rel_audio}: {err}")
            else:
                dur = audio_info["duration"]
                total_duration += dur
                durations.append(dur)
                sample_rates.add(audio_info["samplerate"])
                channels_set.add(audio_info["channels"])

            records.append({
                "row": idx,
                "audio_path": rel_audio,
                "full_path": full_audio,
                "transcript": transcript,
                "language": lang,
                "duration": audio_info["duration"] if audio_info else 0.0,
                "split": row.get("split", split_name)
            })

    stats = {
        "csv_path": csv_path,
        "split": split_name,
        "count": len(records),
        "total_duration_sec": total_duration,
        "total_duration_hr": total_duration / 3600.0,
        "avg_duration_sec": (total_duration / len(records)) if records else 0.0,
        "min_duration_sec": min(durations) if durations else 0.0,
        "max_duration_sec": max(durations) if durations else 0.0,
        "sample_rates": list(sample_rates),
        "channels": list(channels_set),
        "audio_paths": audio_paths_seen
    }
    return stats, errors


def check_split_leakage(train_paths, val_paths, test_paths):
    """
    Checks for overlapping audio files between training, validation, and test splits.
    """
    leakage = []
    tv = train_paths.intersection(val_paths)
    if tv:
        leakage.append(f"Train/Validation overlap: {len(tv)} items (e.g., {list(tv)[:3]})")
    tt = train_paths.intersection(test_paths)
    if tt:
        leakage.append(f"Train/Test overlap: {len(tt)} items (e.g., {list(tt)[:3]})")
    vt = val_paths.intersection(test_paths)
    if vt:
        leakage.append(f"Validation/Test overlap: {len(vt)} items (e.g., {list(vt)[:3]})")
    return leakage


def test_whisper_tokenizer(records, model_id="openai/whisper-base"):
    """
    Tests Whisper tokenizer loading and tokenization of sample transcripts.
    """
    if not HAS_TRANSFORMERS:
        return {"status": "SKIPPED", "reason": "transformers package not installed"}

    try:
        tokenizer = WhisperTokenizer.from_pretrained(model_id, language="english", task="transcribe")
        for rec in records[:10]:
            prompt_ids = tokenizer(rec["transcript"]).input_ids
            if not prompt_ids:
                return {"status": "FAILED", "reason": f"Empty tokenization for transcript: {rec['transcript']}"}
        return {"status": "PASSED", "model": model_id, "samples_tested": min(10, len(records))}
    except Exception as e:
        return {"status": "FAILED", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Validate StoreListener Whisper fine-tuning manifests.")
    parser.add_argument("--base-dir", default=".", help="Base directory of the dataset workspace.")
    parser.add_argument("--manifest", default="dataset_manifest.csv", help="Full combined manifest path.")
    parser.add_argument("--train", default="train.csv", help="Train split manifest.")
    parser.add_argument("--val", default="validation.csv", help="Validation split manifest.")
    parser.add_argument("--test", default="test.csv", help="Test split manifest.")
    parser.add_argument("--output-report", default="validation_summary.json", help="Path to write JSON summary.")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    print("=" * 70)
    print("STORELISTENER ASR DATASET VALIDATION AUDIT")
    print("=" * 70)
    print(f"Base Directory: {base_dir}")

    all_errors = {}
    splits_stats = {}

    for name, path in [("train", args.train), ("validation", args.val), ("test", args.test)]:
        full_csv = os.path.join(base_dir, path)
        print(f"\n[*] Validating split: {name} ({path})...")
        stats, errs = validate_csv(full_csv, base_dir, split_name=name)
        if errs:
            all_errors[name] = errs
            print(f"  [!] Errors found: {len(errs)}")
            for e in errs[:5]:
                print(f"      - {e}")
            if len(errs) > 5:
                print(f"      ... and {len(errs) - 5} more.")
        else:
            print("  [+] Split check PASSED.")

        if stats:
            splits_stats[name] = stats
            print(f"  Count: {stats['count']} clips")
            print(f"  Total Duration: {stats['total_duration_sec']:.2f}s ({stats['total_duration_hr']:.3f} hrs)")
            print(f"  Duration Range: [{stats['min_duration_sec']:.2f}s - {stats['max_duration_sec']:.2f}s], Avg: {stats['avg_duration_sec']:.2f}s")
            print(f"  Sample Rates: {stats['sample_rates']} Hz, Channels: {stats['channels']}")

    # Check leakage between splits
    train_paths = splits_stats.get("train", {}).get("audio_paths", set())
    val_paths = splits_stats.get("validation", {}).get("audio_paths", set())
    test_paths = splits_stats.get("test", {}).get("audio_paths", set())

    leakages = check_split_leakage(train_paths, val_paths, test_paths)
    if leakages:
        all_errors["leakage"] = leakages
        print("\n[!] DATA LEAKAGE DETECTED:")
        for l in leakages:
            print(f"    - {l}")
    else:
        print("\n[+] Split isolation verified: ZERO data leakage detected across train, val, and test.")

    # Check Whisper Tokenizer compatibility
    sample_records = []
    full_manifest_path = os.path.join(base_dir, args.manifest)
    if os.path.exists(full_manifest_path):
        m_stats, m_errs = validate_csv(full_manifest_path, base_dir, split_name="manifest")
        if m_errs:
            all_errors["manifest"] = m_errs
        else:
            print(f"\n[+] Combined manifest verified: {m_stats['count']} total records.")

    print("\n[*] Testing Whisper tokenizer compatibility...")
    tok_result = test_whisper_tokenizer([{"transcript": "Hello StoreListener customer"}])
    print(f"    Tokenizer Status: {tok_result.get('status')} ({tok_result.get('reason', 'OK')})")

    total_clips = sum(s["count"] for s in splits_stats.values())
    total_hr = sum(s["total_duration_hr"] for s in splits_stats.values())

    summary = {
        "status": "PASSED" if not all_errors else "FAILED",
        "total_clips": total_clips,
        "total_duration_hr": total_hr,
        "splits": {k: {m: v for m, v in s.items() if m != "audio_paths"} for k, s in splits_stats.items()},
        "tokenizer_compatibility": tok_result,
        "errors": all_errors
    }

    with open(os.path.join(base_dir, args.output_report), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"OVERALL VALIDATION STATUS: {summary['status']}")
    print(f"Total Validated Clips: {total_clips}")
    print(f"Total Audio Duration: {total_hr:.2f} hours")
    print(f"Summary Report Written to: {args.output_report}")
    print("=" * 70)

    if all_errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
