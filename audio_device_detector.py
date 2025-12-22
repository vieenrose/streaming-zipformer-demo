#!/usr/bin/env python3

"""
Audio Input Device Detector with VAD.

Enumerates all available audio input devices and detects which device
has active speech (VAD probability > 0.5).

Audio Processing Pipeline:
  Raw audio (device rate) -> scipy resampler -> 16kHz
  -> AGC (normalize levels) -> RMS meter (verify AGC) -> ten-vad -> device selection

Usage:
  python audio_device_detector.py [--threshold 0.75] [--chunk-duration 100]
"""

import argparse
import sys
import time
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

from audio_device_enum import enumerate_input_devices, AudioDeviceInfo, print_devices
from audio_processor import Resampler, RMSMeter, VADDetector, AGC


class DeviceScanner:
    """
    Scans all audio input devices for voice activity.
    """

    def __init__(self, vad_threshold: float = 0.75, chunk_duration_ms: int = 100):
        """
        Initialize device scanner.

        Args:
            vad_threshold: VAD probability threshold to exit (default: 0.75)
            chunk_duration_ms: Audio chunk duration in milliseconds (default: 100ms)
        """
        self.vad_threshold = vad_threshold
        self.chunk_duration_ms = chunk_duration_ms

        # Initialize components
        self.vad_detector = VADDetector(sample_rate=16000)
        self.rms_meter = RMSMeter()
        self.agc = AGC(target_level=0.1, max_gain=100.0)  # 40 dB max gain

        # State tracking
        self.devices = enumerate_input_devices()
        self.resamplers = {}
        self.streams = {}

        if not self.devices:
            raise RuntimeError("No audio input devices found!")

        print(f"\nFound {len(self.devices)} audio input device(s)")
        for device in self.devices:
            print(f"  [{device.device_id}] {device.name} ({device.channels}ch, {device.sample_rate}Hz)")

    def _open_device_stream(self, device: AudioDeviceInfo) -> Tuple[sd.InputStream, Resampler]:
        """
        Open audio input stream for a device.

        Args:
            device: Device to open

        Returns:
            Tuple of (InputStream, Resampler)
        """
        # Calculate samples per chunk based on device's native sample rate
        samples_per_chunk = int(device.sample_rate * self.chunk_duration_ms / 1000)

        try:
            stream = sd.InputStream(
                device=device.device_id,
                channels=1,
                samplerate=int(device.sample_rate),
                blocksize=samples_per_chunk,
                dtype=np.float32,
            )
            stream.start()

            # Create resampler for this device
            resampler = Resampler(
                input_rate=int(device.sample_rate),
                output_rate=16000,  # Required for ten-vad
                channels=1
            )

            return stream, resampler

        except Exception as e:
            print(f"  Warning: Failed to open device {device.device_id} ({device.name}): {e}")
            return None, None

    def scan_devices(self) -> Tuple[Optional[AudioDeviceInfo], float]:
        """
        Scan all devices for voice activity.

        In each cycle:
        1. Scans all available devices and gets VAD for each
        2. Finds the device with highest VAD in that cycle
        3. If highest VAD > threshold, exits and selects that device
        4. Otherwise, continues to next cycle

        Returns:
            Tuple of (detected_device, vad_probability)
            Returns (None, 0.0) if no device exceeds threshold
        """
        print(f"\nStarting device scan (threshold: {self.vad_threshold})...")
        print("Listening for speech... (Ctrl+C to exit)\n")

        # Open streams for all devices
        active_streams = {}
        for device in self.devices:
            stream, resampler = self._open_device_stream(device)
            if stream is not None:
                active_streams[device.device_id] = {
                    "device": device,
                    "stream": stream,
                    "resampler": resampler,
                }

        if not active_streams:
            print("Error: Could not open any device streams!")
            return None, 0.0

        print(f"Successfully opened {len(active_streams)} device stream(s)\n")

        try:
            cycle = 0
            while True:
                cycle += 1
                vad_readings = {}  # Store VAD for each device in this cycle

                print(f"[Cycle {cycle}] Scanning all devices...")

                # Scan all active devices in this cycle
                for device_id, device_info in active_streams.items():
                    device = device_info["device"]
                    stream = device_info["stream"]
                    resampler = device_info["resampler"]

                    try:
                        # Read audio chunk from device
                        chunk_samples = int(
                            device.sample_rate * self.chunk_duration_ms / 1000
                        )
                        audio, overflowed = stream.read(chunk_samples)
                        audio = audio.reshape(-1)

                        if overflowed:
                            print(f"  Warning: Buffer overflow on device {device.device_id}")

                        # Resample to 16kHz
                        audio_16k = resampler.resample(audio)

                        if len(audio_16k) == 0:
                            continue

                        # Apply AGC to normalize levels (helps VAD detect low-volume speech)
                        audio_16k_agc = self.agc.apply(audio_16k)

                        # Compute RMS on AGC-normalized signal (verify normalization)
                        rms = self.rms_meter.compute_rms(audio_16k_agc)

                        # Get VAD probability on AGC-normalized audio
                        vad_prob = self.vad_detector.get_speech_probability(audio_16k_agc)

                        # Store reading for this device
                        vad_readings[device_id] = {
                            "device": device,
                            "vad_prob": vad_prob,
                            "rms": rms
                        }

                        # Print status
                        print(
                            f"  Device {device.device_id:2d} ({device.name:30s}): "
                            f"VAD={vad_prob:.3f} RMS={rms:.4f}"
                        )

                    except Exception as e:
                        print(f"  Error reading from device {device.device_id}: {e}")
                        continue

                # Find device with highest VAD in this cycle
                if vad_readings:
                    max_device_id = max(vad_readings.keys(),
                                       key=lambda k: vad_readings[k]["vad_prob"])
                    max_reading = vad_readings[max_device_id]
                    max_device = max_reading["device"]
                    max_vad = max_reading["vad_prob"]

                    print(f"\n  → Highest VAD: Device {max_device.device_id} ({max_device.name}) = {max_vad:.3f}")

                    # Check if threshold exceeded
                    if max_vad > self.vad_threshold:
                        print(f"\n{'=' * 70}")
                        print(f"✓ DETECTED: Device {max_device.device_id} ({max_device.name})")
                        print(f"  VAD Probability: {max_vad:.3f} (threshold: {self.vad_threshold})")
                        print(f"{'=' * 70}")
                        return max_device, max_vad
                    else:
                        print(f"  (Below threshold {self.vad_threshold}, continuing...)\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            return None, 0.0

        finally:
            # Close all streams
            for device_info in active_streams.values():
                try:
                    device_info["stream"].stop()
                    device_info["stream"].close()
                except Exception:
                    pass

    def record_device(self, device_id: int, duration_sec: float = 10.0, output_file: Optional[str] = None) -> Optional[str]:
        """
        Record audio from a device and save to file.

        Args:
            device_id: Device to record from
            duration_sec: Duration of recording in seconds (default: 10s)
            output_file: Output file path (if None, auto-generate)

        Returns:
            Path to saved recording file
        """
        device = next((d for d in self.devices if d.device_id == device_id), None)
        if device is None:
            print(f"Device {device_id} not found!")
            return None

        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"device_{device_id}_{timestamp}.wav"

        print(f"\nRecording from device {device_id}: {device.name}")
        print(f"Duration: {duration_sec} seconds")
        print(f"Output file: {output_file}")
        print(f"Recording...")

        try:
            # Record at device's native sample rate
            with sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=int(device.sample_rate),
                dtype=np.float32,
            ) as stream:
                # Calculate total samples to record
                total_samples = int(device.sample_rate * duration_sec)
                audio_data = []

                while sum(len(chunk) for chunk in audio_data) < total_samples:
                    chunk_size = min(int(device.sample_rate * 0.1), total_samples - sum(len(c) for c in audio_data))
                    chunk, _ = stream.read(chunk_size)
                    audio_data.append(chunk.reshape(-1))

                # Combine all chunks
                audio = np.concatenate(audio_data)
                audio = audio[:total_samples]  # Trim to exact duration

                # Save to file
                sf.write(output_file, audio, int(device.sample_rate))
                print(f"✓ Recording saved: {output_file}")
                print(f"  Sample rate: {int(device.sample_rate)} Hz")
                print(f"  Duration: {len(audio) / device.sample_rate:.2f} seconds")
                print(f"  Samples: {len(audio)}")

                return output_file

        except Exception as e:
            print(f"Error recording: {e}")
            return None

    def test_device(self, device_id: int, duration_sec: float = 5.0) -> None:
        """
        Test a specific device for VAD detection.

        Args:
            device_id: Device to test
            duration_sec: Duration of test in seconds
        """
        device = next((d for d in self.devices if d.device_id == device_id), None)
        if device is None:
            print(f"Device {device_id} not found!")
            return

        print(f"\nTesting device {device.device_id}: {device.name}")
        print(f"Recording for {duration_sec} seconds... Speak into the microphone!")

        resampler = Resampler(input_rate=int(device.sample_rate), output_rate=16000)
        vad_probs = []

        try:
            with sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=int(device.sample_rate),
                dtype=np.float32,
            ) as stream:
                start_time = time.time()

                while time.time() - start_time < duration_sec:
                    chunk_samples = int(device.sample_rate * 0.1)  # 100ms chunks
                    audio, _ = stream.read(chunk_samples)
                    audio = audio.reshape(-1)

                    audio_16k = resampler.resample(audio)
                    if len(audio_16k) > 0:
                        vad_prob = self.vad_detector.get_speech_probability(audio_16k)
                        vad_probs.append(vad_prob)
                        rms = self.rms_meter.compute_rms(audio_16k)

                        elapsed = time.time() - start_time
                        print(
                            f"  [{elapsed:.1f}s] VAD={vad_prob:.3f} RMS={rms:.4f}",
                            end="\r"
                        )

        except Exception as e:
            print(f"Error: {e}")
            return

        print()  # Newline
        if vad_probs:
            avg_vad = np.mean(vad_probs)
            max_vad = np.max(vad_probs)
            print(f"Results:")
            print(f"  Average VAD: {avg_vad:.3f}")
            print(f"  Max VAD:     {max_vad:.3f}")
        else:
            print("No audio data captured!")


def get_args():
    parser = argparse.ArgumentParser(
        description="Detect active audio input device using VAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all audio input devices and exit",
    )

    parser.add_argument(
        "--test",
        type=int,
        metavar="DEVICE_ID",
        help="Test a specific device for VAD detection (duration: 5s)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="VAD probability threshold to detect speech (0.0 to 1.0)",
    )

    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=100,
        metavar="MS",
        help="Audio chunk duration in milliseconds",
    )

    parser.add_argument(
        "--record",
        type=int,
        metavar="DURATION_SEC",
        default=10,
        help="Record audio from detected device for N seconds (default: 10s)",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # List devices and exit
    if args.list:
        print_devices()
        return

    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        print("Error: threshold must be between 0.0 and 1.0")
        sys.exit(1)

    try:
        scanner = DeviceScanner(
            vad_threshold=args.threshold,
            chunk_duration_ms=args.chunk_duration,
        )

        # Test specific device
        if args.test is not None:
            scanner.test_device(args.test)
            return

        # Scan all devices
        detected_device, vad_prob = scanner.scan_devices()

        if detected_device:
            print(f"\nDetection complete!")
            print("=" * 70)
            print("DETECTED DEVICE DETAILS")
            print("=" * 70)
            print(f"Device ID:              {detected_device.device_id}")
            print(f"Device Name:            {detected_device.name}")
            print(f"Input Channels:         {detected_device.channels}")
            print(f"Default Sample Rate:    {detected_device.sample_rate} Hz")
            print(f"Host API:               {detected_device.api}")
            print(f"VAD Probability:        {vad_prob:.4f}")
            print(f"Threshold:              {args.threshold}")
            print(f"Status:                 ✓ SPEECH DETECTED")
            print("=" * 70)
            print(f"\nDevice {detected_device.device_id} is ready for audio capture!")

            # Record audio if requested
            if args.record and args.record > 0:
                print(f"\n" + "=" * 70)
                print("RECORDING AUDIO")
                print("=" * 70)
                output_file = scanner.record_device(
                    detected_device.device_id,
                    duration_sec=args.record
                )
                if output_file:
                    print(f"\n✓ Recording completed: {output_file}")
                    print(f"Ready for waveform inspection and analysis")
                else:
                    print(f"Failed to record audio")
        else:
            print("\nNo device detected with speech above threshold")

    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
