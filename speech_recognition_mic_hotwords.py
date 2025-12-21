#!/usr/bin/env python3

"""
Speech recognition with hotword support
"""

import argparse
import sys
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first")
    sys.exit(-1)

import sherpa_onnx


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
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
        "--device",
        type=int,
        default=None,
        help="Audio device index (e.g., 4 for hw:0,6)",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for speech recognition model",
    )

    parser.add_argument(
        "--hotwords",
        type=str,
        default="",
        help="Hotwords as string (comma-separated or newline-separated)",
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="Hotword score for boosting",
    )

    parser.add_argument(
        "--bpe-vocab",
        type=str,
        default="",
        help="Path to bpe.vocab file (required for hotwords with BPE models)",
    )

    parser.add_argument(
        "--modeling-unit",
        type=str,
        default="bpe",
        help="Modeling unit (bpe, cjkchar, cjkchar+bpe)",
    )

    return parser.parse_args()


def create_recognizer(args):
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)

    kwargs = {
        "tokens": args.tokens,
        "encoder": args.encoder,
        "decoder": args.decoder,
        "joiner": args.joiner,
        "num_threads": args.num_threads,
        "sample_rate": 16000,
        "feature_dim": 80,
        "enable_endpoint_detection": True,
        "rule1_min_trailing_silence": 2.4,
        "rule2_min_trailing_silence": 1.2,
        "rule3_min_utterance_length": 300,
        "decoding_method": "modified_beam_search",  # Required for hotwords to work
    }

    if args.bpe_vocab:
        assert_file_exists(args.bpe_vocab)
        kwargs["bpe_vocab"] = args.bpe_vocab
        kwargs["modeling_unit"] = args.modeling_unit

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
    return recognizer


def main():
    args = get_args()

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print("Available devices:")
    print(devices)

    if args.device is not None:
        device_idx = args.device
        print(f'Using specified device: {devices[device_idx]["name"]}')
    else:
        device_idx = sd.default.device[0]
        print(f'Using default device: {devices[device_idx]["name"]}')

    print(f"Threads: {args.num_threads}")

    recognizer = create_recognizer(args)

    # Prepare hotwords string for stream
    hotwords_str = ""
    if args.hotwords:
        print(f"Hotwords enabled:")
        # Handle both comma-separated and newline-separated formats
        if "," in args.hotwords:
            hotwords_list = [hw.strip() for hw in args.hotwords.split(",")]
        else:
            hotwords_list = [hw.strip() for hw in args.hotwords.split("\n")]

        for hw in hotwords_list:
            if hw:
                print(f"  - {hw}")

        # Build hotwords string for create_stream()
        hotwords_str = "\n".join([hw for hw in hotwords_list if hw])
        print(f"  Score: {args.hotwords_score}")

    print("Started! Please speak")

    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 100ms

    # Create stream with hotwords
    if hotwords_str:
        stream = recognizer.create_stream(hotwords=hotwords_str, hotwords_score=args.hotwords_score)
    else:
        stream = recognizer.create_stream()

    display = sherpa_onnx.Display()

    with sd.InputStream(device=device_idx, channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)

            stream.accept_waveform(sample_rate, samples)

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            is_endpoint = recognizer.is_endpoint(stream)

            result = recognizer.get_result(stream)

            display.update_text(result)
            display.display()

            if is_endpoint:
                if result:
                    display.finalize_current_sentence()
                    display.display()

                recognizer.reset(stream)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
