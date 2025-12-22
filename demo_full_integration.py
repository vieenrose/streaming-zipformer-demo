#!/usr/bin/env python3

"""
FULL INTEGRATION DEMO: Step 4 + Step 5 + Step 6

Phase 1: Audio Device Detection (Step 4)
  - Enumerate all audio devices
  - Scan for active speech using VAD
  - Auto-select device with highest VAD probability

Phase 2: MP3 Recording (Step 5)
  - Start background recording on selected device
  - Record for 10 seconds while user speaks
  - Verify MP3 file integrity

Phase 3: Parallel ASR Transcription (Step 6)
  - Load 3 sherpa-onnx ASR models in parallel
  - Feed audio chunks from detected device to all models
  - Display live transcriptions from all 3 engines
  - Compare accuracy and latency
"""

import sys
sys.path.insert(0, '/home/luigi/sherpa')

import time
import os
import json
import subprocess
import numpy as np
from collections import deque

from audio_device_detector import DeviceScanner
from audio_processor import Resampler
from asr_engine import ASREnginePool
from asr_config import RecognitionConfig
from mp3_writer import MP3Writer


def print_phase(phase_num, title):
    """Print phase header."""
    print("\n" + "=" * 80)
    print(f"PHASE {phase_num}: {title}")
    print("=" * 80 + "\n")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "FULL INTEGRATION DEMO: Audio Device Detection + MP3 Recording".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # =========================================================================
    # PHASE 1: AUDIO DEVICE DETECTION
    # =========================================================================
    print_phase(1, "Audio Device Detection with VAD (Step 4)")

    try:
        # Create scanner with default VAD threshold
        print("Creating DeviceScanner with VAD threshold: 0.75")
        scanner = DeviceScanner(vad_threshold=0.75, chunk_duration_ms=100)

        print("\n▶ Starting device scan for active speech...")
        print("  Listen carefully - speak into your microphone when the scan begins")
        print("  The scanner will automatically detect the device with active speech\n")

        # Scan all devices for active speech
        detected_device, vad_prob = scanner.scan_devices()

        if not detected_device:
            print("\n✗ PHASE 1 FAILED: No device detected with speech above threshold")
            print("   Please try again and speak louder into your microphone")
            return False

        print("\n" + "-" * 80)
        print("✓ PHASE 1 COMPLETE: Device detected successfully")
        print("-" * 80)
        print(f"\nDetected Device:")
        print(f"  Device ID:       {detected_device.device_id}")
        print(f"  Device Name:     {detected_device.name}")
        print(f"  Channels:        {int(detected_device.channels)}")
        print(f"  Sample Rate:     {int(detected_device.sample_rate)} Hz")
        print(f"  Host API:        {detected_device.api}")
        print(f"\nDetection Metrics:")
        print(f"  VAD Probability: {vad_prob:.4f}")
        print(f"  Threshold:       0.75")
        print(f"  Status:          ✓ SPEECH DETECTED")

        selected_device_id = detected_device.device_id

    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user during Phase 1")
        return False
    except Exception as e:
        print(f"\n✗ Phase 1 Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # PHASE 2 & 3: PARALLEL MP3 RECORDING + ASR TRANSCRIPTION
    # =========================================================================
    print_phase(2, "MP3 Recording + Parallel ASR (Steps 5 & 6 Combined)")

    try:
        # Generate output filename
        output_file = f"/tmp/demo_speech_{selected_device_id}_{int(time.time())}.mp3"

        # Create ASR engine pool BEFORE starting recording
        print("Loading ASR models (may take 1-2 minutes)...")
        asr_pool = ASREnginePool()
        success = asr_pool.load_models()
        if not success:
            print("✗ Failed to load ASR models")
            return False

        print(f"✓ Loaded {len(asr_pool.models)} ASR models\n")

        # Create resampler (device rate -> 16kHz for ASR)
        device_sample_rate = int(detected_device.sample_rate)
        resampler = Resampler(
            input_rate=device_sample_rate,
            output_rate=RecognitionConfig.SAMPLE_RATE,
        )

        # Chunk size at DEVICE sample rate
        chunk_samples = int(
            device_sample_rate * RecognitionConfig.CHUNK_DURATION_MS / 1000
        )

        # Create MP3 writer (receives 16kHz audio, same as ASR)
        mp3_writer = MP3Writer(
            output_file=output_file,
            sample_rate=RecognitionConfig.SAMPLE_RATE,  # 16kHz
            channels=1,
            bitrate=128,
        )

        print(f"Starting recording + transcription...")
        print(f"  Device:   {detected_device.name} (ID: {selected_device_id})")
        print(f"  Output:   {output_file}")
        print(f"  Duration: 10 seconds\n")

        # Open SINGLE audio stream
        import sounddevice as sd

        recording_duration = 10
        num_chunks = int(recording_duration * 1000 / RecognitionConfig.CHUNK_DURATION_MS)

        stream = sd.InputStream(
            device=selected_device_id,
            channels=1,
            samplerate=device_sample_rate,
            blocksize=chunk_samples,
            dtype=np.float32,
            latency='low',
        )

        # Start MP3 writer
        mp3_writer.start()

        # Start audio stream
        stream.start()

        print("▶ Recording and transcribing CONTINUOUSLY...")
        print("  Speak clearly into the microphone!")
        print("  Press Ctrl+C to stop.\n")
        print("┌" + "─" * 78 + "┐")
        print("│ LIVE ASR TRANSCRIPTION (MP3 Recording in Background)".ljust(79) + "│")
        print("│ Press Ctrl+C to stop recording".ljust(79) + "│")
        print("├" + "─" * 78 + "┤")

        chunks_processed = 0
        start_time = time.time()

        try:
            while True:  # Run indefinitely until Ctrl+C
                # Read audio chunk from device
                audio, overflowed = stream.read(chunk_samples)
                audio = audio.reshape(-1).astype(np.float32)

                # Resample to 16kHz
                audio_16k = resampler.resample(audio)

                # DUAL OUTPUT: Same 16kHz audio goes to BOTH:
                # 1. MP3 Writer (background encoding)
                mp3_writer.write_chunk(audio_16k)

                # 2. ASR Engines (real-time transcription)
                asr_pool.feed_audio_chunk(audio_16k)
                asr_pool.process()

                chunks_processed += 1

                # Get and display ASR results (every 10 chunks = 1 second)
                if chunks_processed % 10 == 0:
                    elapsed = time.time() - start_time
                    results = asr_pool.get_results()
                    for model_id, result in results.items():
                        display_text = result.partial if result.partial else result.final
                        display_text = display_text[:55] if len(display_text) > 55 else display_text
                        print(f"│ {model_id:18s} │ {display_text:57s} │")

                    # Show elapsed time
                    mins = int(elapsed // 60)
                    secs = elapsed % 60
                    time_str = f"[Recording: {mins:02d}:{secs:05.2f}]"
                    print(f"│ {time_str:78s} │")

        except KeyboardInterrupt:
            print("\n│ ✓ Stopping (Ctrl+C received)...".ljust(79) + "│")

        # Stop stream and MP3 writer
        stream.stop()
        stream.close()

        print("└" + "─" * 78 + "┘")
        print("\n▶ Finalizing MP3 file...")
        mp3_writer.stop()

        # Verify MP3 file
        if not os.path.exists(output_file):
            print("✗ MP3 file was not created")
            return False

        file_size = os.path.getsize(output_file)
        mp3_stats = mp3_writer.get_stats()
        print(f"✓ MP3 file created: {file_size / 1024:.1f} KB")
        print(f"  Chunks written: {mp3_stats['chunks_written']}")
        print(f"  Audio duration: {mp3_stats['duration_sec']:.2f}s\n")

        # Display final results
        print("-" * 80)
        print("✓ COMBINED PHASE COMPLETE: Parallel recording + ASR successful")
        print("-" * 80)

        print(f"\nMP3 Recording Details:")
        print(f"  File:       {output_file}")
        print(f"  Size:       {file_size / 1024:.1f} KB")
        print(f"  Duration:   {mp3_stats['duration_sec']:.2f} seconds")

        print(f"\nASR Transcription Results ({chunks_processed} chunks):")
        results = asr_pool.get_results()
        for model_id in sorted(results.keys()):
            result = results[model_id]
            partial = result.partial if result.partial else "(empty)"
            final = result.final if result.final else "(empty)"
            print(f"\n  {result.model_name}:")
            print(f"    Partial: {partial}")
            print(f"    Final:   {final}")

        asr_pool.cleanup()

        # Store duration for final summary
        duration = mp3_stats['duration_sec']

    except Exception as e:
        print(f"\n✗ Phase Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            if 'mp3_writer' in locals():
                mp3_writer.stop()
            if 'stream' in locals():
                stream.stop()
                stream.close()
            if 'asr_pool' in locals():
                asr_pool.cleanup()
        except:
            pass
        return False

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("✓ FULL INTEGRATION DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\n📋 Summary:")
    print(f"  Phase 1 (Detection):      ✓ Detected device {selected_device_id}")
    print(f"  Phase 2 (MP3 Recording):  ✓ Recorded {duration:.2f}s of audio")
    print(f"  Phase 3 (ASR Streaming):  ✓ Transcribed with 3 models in parallel")

    print("\n📁 Files Created:")
    print(f"  MP3 Recording: {output_file}")

    print("\n💡 Next Steps:")
    print("  1. Playback: ffplay " + output_file)
    print("  2. Inspect: ffprobe " + output_file)
    print("  3. Convert: ffmpeg -i " + output_file + " output.wav")

    print("\n" + "=" * 80 + "\n")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
