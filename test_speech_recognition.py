#!/usr/bin/env python3

"""
Test script for speech recognition using soundfile with pre-recorded audio
Falls back to this if microphone is not available
"""

import argparse
import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("Please install soundfile first. You can use")
    print()
    print("  pip install soundfile")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_onnx


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        required=True,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to audio file to test",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    return parser.parse_args()


def create_recognizer(args):
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        decoding_method=args.decoding_method,
        provider=args.provider,
    )
    return recognizer


def main():
    args = get_args()
    assert_file_exists(args.audio_file)

    print(f"Loading audio file: {args.audio_file}")
    samples, sample_rate = sf.read(args.audio_file, dtype="float32")

    # Resample if necessary
    if sample_rate != 16000:
        print(f"Warning: Audio sample rate is {sample_rate}, expected 16000. Model may not work optimally.")

    print("Creating recognizer...")
    recognizer = create_recognizer(args)
    print("Recognizer created successfully!")

    print("\nProcessing audio...")
    stream = recognizer.create_stream()

    # Process audio in chunks (100ms each)
    samples_per_chunk = int(0.1 * 16000)
    for i in range(0, len(samples), samples_per_chunk):
        chunk = samples[i:i + samples_per_chunk]

        stream.accept_waveform(16000, chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        if result:
            print(f"Partial result: {result}")

        is_endpoint = recognizer.is_endpoint(stream)
        if is_endpoint:
            result = recognizer.get_result(stream)
            if result:
                print(f"Final result: {result}")
            recognizer.reset(stream)

    # Get final result
    result = recognizer.get_result(stream)
    if result:
        print(f"\nFinal transcription: {result}")
    else:
        print("\nNo speech detected in audio")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
