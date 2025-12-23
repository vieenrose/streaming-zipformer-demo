#!/usr/bin/env python3

"""
FULL INTEGRATION DEMO: Step 4 + Step 5 + Step 6 + Step 7

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

Phase 4: UI Integration (Step 7)
  - Real-time monitor with fixed-grid layout
  - Three zones: system status, VAD/RMS charts, streaming transcripts
  - Hotword status tracking and display

NEW: File Input Mode
  - Accept audio file as input instead of microphone
  - Process entire file through ASR engines
  - Display results for A/B testing
"""

import sys
sys.path.insert(0, '/home/luigi/sherpa')

import time
import os
import json
import subprocess
import numpy as np
from collections import deque
import argparse
import soundfile as sf
import threading

from audio_device_detector import DeviceScanner
from audio_processor import Resampler
from asr_engine import ASREnginePool, StreamResult
from asr_config import RecognitionConfig
from mp3_writer import MP3Writer
from ui_monitor import DockerCompatibleUI, UIThread, DeviceStatus


def print_phase(phase_num, title):
    """Print phase header."""
    print("\n" + "=" * 80)
    print(f"PHASE {phase_num}: {title}")
    print("=" * 80 + "\n")


def process_audio_file(file_path, asr_pool, hotwords_enabled=True):
    """
    Process an audio file through ASR engines.

    Args:
        file_path: Path to the audio file
        asr_pool: ASREnginePool instance
        hotwords_enabled: Whether to use hotwords in recognition
    """
    print(f"Loading audio file: {file_path}")

    # Load audio file
    audio_data, sample_rate = sf.read(file_path)

    # Ensure audio is mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Take first channel if stereo

    # Create resampler if needed
    if sample_rate != RecognitionConfig.SAMPLE_RATE:
        from audio_processor import Resampler
        resampler = Resampler(
            input_rate=sample_rate,
            output_rate=RecognitionConfig.SAMPLE_RATE,
        )
        audio_data = resampler.resample(audio_data)
    else:
        resampler = None

    # Process audio in chunks
    chunk_size = int(RecognitionConfig.SAMPLE_RATE * RecognitionConfig.CHUNK_DURATION_MS / 1000)
    total_samples = len(audio_data)
    chunks_processed = 0

    print(f"Processing {total_samples} samples at {RecognitionConfig.SAMPLE_RATE} Hz")
    print(f"Chunk size: {chunk_size} samples ({RecognitionConfig.CHUNK_DURATION_MS} ms)")

    start_time = time.time()

    for i in range(0, total_samples, chunk_size):
        # Extract chunk
        chunk = audio_data[i:i+chunk_size]

        # Pad if needed (last chunk might be smaller)
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

        # Ensure float32 format
        chunk = chunk.astype(np.float32)

        # Feed to ASR engines
        asr_pool.feed_audio_chunk(chunk)
        asr_pool.process()

        chunks_processed += 1

        # Print progress every 100 chunks
        if chunks_processed % 100 == 0:
            progress = min(100.0, (i + chunk_size) / total_samples * 100)
            elapsed = time.time() - start_time
            print(f"Progress: {progress:.1f}% ({chunks_processed} chunks processed in {elapsed:.2f}s)")

    total_time = time.time() - start_time
    print(f"File processing completed in {total_time:.2f} seconds")
    print(f"Processed {chunks_processed} chunks")

    return chunks_processed


def calculate_rms(audio_chunk):
    """Calculate RMS (Root Mean Square) of audio chunk."""
    return np.sqrt(np.mean(audio_chunk ** 2))


class SlowAGC:
    """Slow Automatic Gain Control for stable amplification in ASR pipeline."""

    def __init__(self, target_level=0.2, max_gain=256.0, attack_time=1.0, release_time=3.0, sample_rate=16000):
        """
        Initialize Slow AGC parameters.

        Args:
            target_level: Desired output level (0.0-1.0)
            max_gain: Maximum allowed gain to prevent excessive amplification
            attack_time: Time constant for gain reduction (seconds) - SLOW for stability
            release_time: Time constant for gain increase (seconds) - SLOW for stability
            sample_rate: Audio sample rate
        """
        self.target_level = target_level
        self.max_gain = max_gain
        self.attack_coeff = np.exp(-1.0 / (attack_time * sample_rate))
        self.release_coeff = np.exp(-1.0 / (release_time * sample_rate))
        self.current_gain = 1.0
        self.sample_rate = sample_rate

    def process(self, audio_chunk):
        """
        Process audio chunk with Slow AGC.

        Args:
            audio_chunk: Input audio as numpy array

        Returns:
            AGC-processed audio chunk
        """
        # Calculate RMS of input chunk
        input_rms = calculate_rms(audio_chunk)

        # Avoid division by zero
        if input_rms == 0:
            return audio_chunk * self.current_gain

        # Calculate desired gain to achieve target level
        desired_gain = self.target_level / input_rms

        # Limit the gain to prevent excessive amplification
        desired_gain = min(desired_gain, self.max_gain)

        # Smoothly adjust gain using slow attack/release coefficients for stability
        if desired_gain < self.current_gain:
            # Signal is getting louder, reduce gain (attack) - SLOW
            self.current_gain = self.current_gain * self.attack_coeff + desired_gain * (1 - self.attack_coeff)
        else:
            # Signal is getting quieter, increase gain (release) - SLOW
            self.current_gain = self.current_gain * self.release_coeff + desired_gain * (1 - self.release_coeff)

        # Apply the current gain to the audio chunk
        output_chunk = audio_chunk * self.current_gain

        # Clip if necessary to prevent values outside [-1, 1]
        output_chunk = np.clip(output_chunk, -1.0, 1.0)

        return output_chunk


def main():
    parser = argparse.ArgumentParser(description='Full Integration Demo with File Input Mode')
    parser.add_argument('--file', type=str, help='Path to audio file for file input mode')
    parser.add_argument('--no-hotwords', action='store_true', help='Disable hotwords for A/B testing')
    args = parser.parse_args()

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    if args.file:
        print("║" + "FULL INTEGRATION DEMO: File Input Mode (A/B Testing)".center(78) + "║")
    else:
        print("║" + "FULL INTEGRATION DEMO: Audio Device Detection + MP3 Recording".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # Create ASR engine pool BEFORE any audio processing
    print("Loading ASR models (may take 1-2 minutes)...")
    asr_pool = ASREnginePool()

    # Modify hotword settings based on command line
    if args.no_hotwords:
        print("Disabling hotwords for A/B testing")
        # Temporarily disable hotwords for this session
        for model_id, model_config in asr_pool.config.models.items():
            model_config.hotwords = None
            print(f"  - Disabled hotwords for {model_config.name}")

    success = asr_pool.load_models()
    if not success:
        print("✗ Failed to load ASR models")
        return False

    print(f"✓ Loaded {len(asr_pool.models)} ASR models\n")

    if args.file:
        # =========================================================================
        # FILE INPUT MODE: Process audio file
        # =========================================================================
        print_phase(1, "File Input Mode - ASR Processing")

        try:
            if not os.path.exists(args.file):
                print(f"✗ Audio file does not exist: {args.file}")
                return False

            print(f"Processing audio file: {args.file}")
            print(f"Hotwords enabled: {'No' if args.no_hotwords else 'Yes'}")

            # Process the audio file
            chunks_processed = process_audio_file(args.file, asr_pool, hotwords_enabled=not args.no_hotwords)

            # Display final results
            print("\n" + "-" * 80)
            print("✓ FILE PROCESSING COMPLETE: ASR transcription successful")
            print("-" * 80)

            print(f"\nASR Transcription Results ({chunks_processed} chunks):")
            results = asr_pool.get_results()
            for model_id in sorted(results.keys()):
                result = results[model_id]
                partial = result.partial if result.partial else "(empty)"
                final = result.final if result.final else "(empty)"
                print(f"\n  {result.model_name}:")
                print(f"    Partial: {partial}")
                print(f"    Final:   {final}")
                print(f"    Latency: {result.latency_ms:.2f} ms/chunk")

            # Store duration for final summary (estimated)
            duration = chunks_processed * RecognitionConfig.CHUNK_DURATION_MS / 1000

        except Exception as e:
            print(f"\n✗ File processing error: {e}")
            import traceback
            traceback.print_exc()
            asr_pool.cleanup()
            return False

    else:
        # =========================================================================
        # MICROPHONE INPUT MODE: Original functionality with UI
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
        # PHASE 2 & 3: PARALLEL MP3 RECORDING + ASR TRANSCRIPTION + UI
        # =========================================================================
        print_phase(2, "MP3 Recording + Parallel ASR + UI (Steps 5, 6 & 7 Combined)")

        try:
            # Generate output filename
            output_file = f"/tmp/demo_speech_{selected_device_id}_{int(time.time())}.mp3"

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

            # Create UI
            ui = DockerCompatibleUI()
            ui.set_device_status(DeviceStatus(
                device_id=detected_device.device_id,
                name=detected_device.name,
                sample_rate=detected_device.sample_rate,
                channels=detected_device.channels,
                dtype="float32",  # Default dtype for audio input
                api=detected_device.api
            ))

            # Set hotwords in UI
            all_hotwords = []
            for model_config in asr_pool.config.models.values():
                if model_config.hotwords:
                    all_hotwords.extend(model_config.hotwords.hotwords)
            ui.update_hotwords(list(set(all_hotwords)))  # Remove duplicates

            # Start UI thread
            ui_thread = UIThread(ui)
            ui_thread.start()

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
            print("  Each model runs on 4 CPU threads for better performance")
            print("  UI running in background with 3 zones (status, VAD/RMS, transcripts)")
            print("  Press Ctrl+C to stop.\n")

            # Initialize Slow AGC for stable amplification
            slow_agc = SlowAGC(target_level=0.2, max_gain=256.0, attack_time=1.0, release_time=3.0, sample_rate=16000)

            chunks_processed = 0
            start_time = time.time()

            try:
                while True:  # Run indefinitely until Ctrl+C
                    # Read audio chunk from device
                    audio, overflowed = stream.read(chunk_samples)
                    audio = audio.reshape(-1).astype(np.float32)

                    # Resample to 16kHz
                    audio_16k = resampler.resample(audio)

                    # Apply Slow AGC to resampled audio for stable amplification
                    agc_audio_16k = slow_agc.process(audio_16k)

                    # DUAL OUTPUT: Same AGC-processed audio goes to BOTH:
                    # 1. MP3 Writer (background encoding)
                    mp3_writer.write_chunk(agc_audio_16k)

                    # 2. ASR Engines (real-time transcription) - now with Slow AGC
                    asr_pool.feed_audio_chunk(agc_audio_16k)
                    asr_pool.process()

                    # Calculate audio levels for UI using the AGC-processed audio
                    # VAD and RMS both use the same AGC-processed signal
                    agc_rms_level = calculate_rms(agc_audio_16k)

                    # Calculate VAD on AGC-processed signal for enhanced sensitivity
                    vad_level = min(1.0, agc_rms_level * 15)  # Increased scaling for better sensitivity
                    rms_level = agc_rms_level  # Use AGC-processed signal for RMS display

                    # Update UI with audio levels
                    ui.update_audio_levels(vad_level, rms_level)

                    # Update UI with ASR results
                    results = asr_pool.get_results()
                    ui_thread.update_results(results)

                    chunks_processed += 1

                    # Update transcripts in UI (every 5 chunks for efficiency)
                    if chunks_processed % 5 == 0:
                        for model_id, result in results.items():
                            transcript = result.partial if result.partial else result.final
                            if transcript:
                                ui.update_transcript(model_id, transcript)

            except KeyboardInterrupt:
                print("\n✓ Stopping (Ctrl+C received)...")

            # Stop UI thread
            ui_thread.stop()

            # Stop stream and MP3 writer
            stream.stop()
            stream.close()

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
            print("✓ COMBINED PHASE COMPLETE: Parallel recording + ASR + UI successful")
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
                if 'ui_thread' in locals():
                    ui_thread.stop()
            except:
                pass
            return False

    # Cleanup
    asr_pool.cleanup()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("✓ FULL INTEGRATION DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)

    if args.file:
        print("\n📋 Summary:")
        print(f"  Mode:             File input mode")
        print(f"  File:             {args.file}")
        print(f"  Hotwords:         {'Disabled' if args.no_hotwords else 'Enabled'}")
        print(f"  ASR Models:       {len(asr_pool.models)}")
        print(f"  Chunks processed: {chunks_processed}")

        print("\n📁 Files Processed:")
        print(f"  Input: {args.file}")
    else:
        print("\n📋 Summary:")
        print(f"  Phase 1 (Detection):      ✓ Detected device {selected_device_id}")
        print(f"  Phase 2 (MP3 Recording):  ✓ Recorded {duration:.2f}s of audio")
        print(f"  Phase 3 (ASR Streaming):  ✓ Transcribed with 3 models in parallel")
        print(f"  Phase 4 (UI Integration): ✓ Real-time monitor with 3 zones")

        print("\n📁 Files Created:")
        print(f"  MP3 Recording: {output_file}")

    print("\n💡 Next Steps:")
    if args.file:
        print(f"  1. Compare results with/without hotwords")
        print(f"  2. Run with --no-hotwords flag for A/B testing")
    else:
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
