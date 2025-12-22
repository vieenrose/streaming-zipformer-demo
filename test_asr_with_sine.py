#!/usr/bin/env python3

"""
Simple test to verify ASR engines produce output with sine wave test signal.
"""

import numpy as np
from asr_engine import ASREnginePool
from asr_config import RecognitionConfig

def test_asr_with_tone():
    """Test ASR with 440Hz sine tone."""
    print("\n" + "=" * 70)
    print("ASR TEST: Feeding 440Hz sine tone to all 3 models")
    print("=" * 70 + "\n")

    # Create pool and load models
    print("Loading models...")
    pool = ASREnginePool()
    success = pool.load_models()
    if not success:
        print("✗ Failed to load models")
        return False

    print("✓ Models loaded\n")

    # Generate 10 seconds of 440Hz sine tone
    sample_rate = RecognitionConfig.SAMPLE_RATE  # 16000 Hz
    duration_sec = 10
    frequency = 440
    amplitude = 0.3

    print(f"Generating {duration_sec}s of {frequency}Hz sine tone (amplitude={amplitude})...")
    total_samples = sample_rate * duration_sec
    t = np.linspace(0, duration_sec, total_samples)
    audio = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    # Feed in 100ms chunks
    chunk_duration_ms = RecognitionConfig.CHUNK_DURATION_MS
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    num_chunks = int(duration_sec * 1000 / chunk_duration_ms)

    print(f"Feeding {num_chunks} chunks of {chunk_duration_ms}ms each to ASR engines...\n")

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_samples
        end_idx = start_idx + chunk_samples
        chunk = audio[start_idx:end_idx]

        # Feed to pool
        pool.feed_audio_chunk(chunk)

        # Process
        pool.process()

        # Get results every 5 chunks
        if (chunk_idx + 1) % 5 == 0:
            results = pool.get_results()
            elapsed = (chunk_idx + 1) * chunk_duration_ms / 1000
            print(f"After {elapsed:.1f}s ({chunk_idx + 1} chunks):")
            for model_id, result in results.items():
                partial = result.partial if result.partial else "(empty)"
                final = result.final if result.final else "(empty)"
                print(f"  {model_id:18s}: partial='{partial}' final='{final}'")
            print()

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70 + "\n")

    results = pool.get_results()
    for model_id in sorted(results.keys()):
        result = results[model_id]
        print(f"{result.model_name}:")
        print(f"  Partial: {result.partial if result.partial else '(empty)'}")
        print(f"  Final:   {result.final if result.final else '(empty)'}")
        print(f"  Chunks:  {result.chunks_processed}")
        print()

    # Check if any model produced output
    has_output = any(r.partial or r.final for r in results.values())

    pool.cleanup()

    if has_output:
        print("✓ ASR produced output")
        return True
    else:
        print("✗ No output from any ASR model - may need to check:")
        print("  1. Is is_ready() being called correctly?")
        print("  2. Should we call decode_stream() in a loop?")
        print("  3. Is the audio format/amplitude correct?")
        print("  4. Do we need to signal endpoint detection?")
        return False

if __name__ == "__main__":
    success = test_asr_with_tone()
    exit(0 if success else 1)
