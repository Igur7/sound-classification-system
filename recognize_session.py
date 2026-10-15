#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sound_recognition_api import MissionSoundRecognizer


def capture_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize a top-mic capture session.")
    parser.add_argument("--session", type=Path, required=True, help="Session dir with top/capture_*.wav")
    parser.add_argument("--reference-dir", type=Path, default=Path(__file__).resolve().parent / "sounds_wav")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--window-ms", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_dir = args.session / "top"
    if not top_dir.exists():
        raise FileNotFoundError(f"Missing top mic directory: {top_dir}")

    wav_paths = sorted(top_dir.glob("capture_*.wav"), key=capture_index)
    if args.max_files is not None:
        wav_paths = wav_paths[: args.max_files]
    if not wav_paths:
        raise FileNotFoundError(f"No capture_*.wav files found in {top_dir}")

    recognizer = MissionSoundRecognizer(
        reference_dir=args.reference_dir,
        window_ms=args.window_ms,
    )
    report = recognizer.identify_session(wav_paths)

    print(
        f"FINAL label={report.result.label} "
        f"confidence={report.result.confidence:.3f} "
        f"votes={dict(report.votes)} "
        f"windows={len(report.per_window_results)}"
    )
    for wav_path, result in report.per_window_results:
        print(f"{wav_path.name}: label={result.label} confidence={result.confidence:.3f}")


if __name__ == "__main__":
    main()
