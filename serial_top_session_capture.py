#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path
import wave

import serial


MAGIC = 0x314D4350
MAGIC_BYTES = struct.pack("<I", MAGIC)
HEADER_STRUCT = struct.Struct("<IHHIIHHQ")
MIC_NAME = "top"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one top microphone PCM16 stream from ESP over UART.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=4000000)
    parser.add_argument("--output-dir", type=Path, default=Path("sessions"))
    parser.add_argument("--label", required=True, help="Session folder name, e.g. woda")
    parser.add_argument("--sample-rate", type=int, default=40100)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--prefix", default="capture")
    return parser.parse_args()


def reserve_session_dir(output_root: Path, label: str) -> Path:
    candidate = output_root / label
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = output_root / f"{label}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def open_wav(path: Path, sample_rate: int) -> wave.Wave_write:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_file = wave.open(str(path), "wb")
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    return wav_file


def write_wav_file(path: Path, sample_rate: int, payload: bytes) -> None:
    with open_wav(path, sample_rate) as wav_file:
        wav_file.writeframes(payload)


def average_abs(payload: bytes) -> int:
    sample_count = len(payload) // 2
    if sample_count == 0:
        return 0
    samples = struct.unpack("<" + "h" * sample_count, payload)
    return sum(abs(sample) for sample in samples) // sample_count


def validate_args(args: argparse.Namespace) -> None:
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")
    if args.overlap_seconds < 0:
        raise ValueError("--overlap-seconds must be >= 0")
    if args.overlap_seconds >= args.window_seconds:
        raise ValueError("--overlap-seconds must be smaller than --window-seconds")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0")


def main() -> None:
    args = parse_args()
    validate_args(args)

    bytes_per_sample = 2
    window_bytes = int(args.sample_rate * args.window_seconds * bytes_per_sample)
    overlap_bytes = int(args.sample_rate * args.overlap_seconds * bytes_per_sample)
    hop_bytes = window_bytes - overlap_bytes

    if window_bytes <= 0 or hop_bytes <= 0:
        raise ValueError("Computed window/hop size is invalid.")

    session_dir = reserve_session_dir(args.output_dir, args.label)
    window_dir = session_dir / MIC_NAME
    window_dir.mkdir(parents=True, exist_ok=True)
    serial_port = serial.Serial(args.port, args.baud, timeout=1)

    packet_buffer = bytearray()
    audio_buffer = bytearray()
    total_bytes_read = 0
    total_packets = 0
    invalid_headers = 0
    no_magic_resyncs = 0
    window_index = 1
    audio_bytes_received = 0
    started_at = time.monotonic()
    last_status_at = started_at

    print(
        f"Capturing {MIC_NAME} from {args.port} at {args.baud} baud, "
        f"window={args.window_seconds:.2f}s overlap={args.overlap_seconds:.2f}s "
        f"session={session_dir}"
    )
    print(f"Window output: {window_dir}")

    try:
        while True:
            now = time.monotonic()
            if args.duration_seconds is not None and now - started_at >= args.duration_seconds:
                print(f"Reached duration {args.duration_seconds:.2f}s, stopping.")
                break

            chunk = serial_port.read(4096)
            if chunk:
                packet_buffer.extend(chunk)
                total_bytes_read += len(chunk)

            now = time.monotonic()
            if now - last_status_at >= 2.0:
                received_seconds = audio_bytes_received / (args.sample_rate * bytes_per_sample)
                print(
                    f"status elapsed={now - started_at:.1f}s bytes={total_bytes_read} "
                    f"packets={total_packets} received={received_seconds:.1f}s "
                    f"invalid={invalid_headers} no_magic={no_magic_resyncs} "
                    f"buffer={len(packet_buffer)}"
                )
                last_status_at = now

            while True:
                start = packet_buffer.find(MAGIC_BYTES)
                if start < 0:
                    if len(packet_buffer) > 3:
                        no_magic_resyncs += 1
                        del packet_buffer[:-3]
                    break

                if start > 0:
                    del packet_buffer[:start]

                if len(packet_buffer) < HEADER_STRUCT.size:
                    break

                header = HEADER_STRUCT.unpack(packet_buffer[:HEADER_STRUCT.size])
                (
                    magic,
                    version,
                    header_size,
                    sequence,
                    sample_rate_hz,
                    sample_count,
                    _reserved,
                    timestamp_us,
                ) = header

                if magic != MAGIC or version != 1 or header_size != HEADER_STRUCT.size:
                    invalid_headers += 1
                    del packet_buffer[0]
                    continue

                payload_size = sample_count * bytes_per_sample
                packet_size = HEADER_STRUCT.size + payload_size
                if len(packet_buffer) < packet_size:
                    break

                payload = bytes(packet_buffer[HEADER_STRUCT.size:packet_size])
                del packet_buffer[:packet_size]

                if sample_rate_hz != args.sample_rate:
                    print(
                        f"Warning: packet sample rate {sample_rate_hz} "
                        f"does not match WAV sample rate {args.sample_rate}",
                        file=sys.stderr,
                    )

                total_packets += 1
                audio_buffer.extend(payload)
                audio_bytes_received += len(payload)

                while len(audio_buffer) >= window_bytes:
                    window_payload = bytes(audio_buffer[:window_bytes])
                    wav_path = window_dir / f"{args.prefix}_{window_index}.wav"
                    write_wav_file(wav_path, args.sample_rate, window_payload)
                    print(
                        f"saved_window={window_index} file={wav_path} "
                        f"seq={sequence} ts={timestamp_us} avgAbs={average_abs(window_payload)}"
                    )
                    del audio_buffer[:hop_bytes]
                    window_index += 1
    except KeyboardInterrupt:
        print("Interrupted, closing WAV file.")
    finally:
        serial_port.close()

        received_seconds = audio_bytes_received / (args.sample_rate * bytes_per_sample)
        trailing_seconds = len(audio_buffer) / (args.sample_rate * bytes_per_sample)
        print(
            f"Stats: bytes={total_bytes_read} packets={total_packets} "
            f"invalid={invalid_headers} no_magic={no_magic_resyncs}"
        )
        print(f"Saved {window_index - 1} window files in {window_dir}.")
        print(f"Received audio duration={received_seconds:.2f}s")
        print(f"Trailing rolling buffer={trailing_seconds:.2f}s")


if __name__ == "__main__":
    main()
