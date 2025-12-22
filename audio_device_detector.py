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
import threading
from typing import List, Tuple, Optional
from datetime import datetime
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
import soundfile as sf

from audio_device_enum import enumerate_input_devices, AudioDeviceInfo, print_devices
from audio_processor import Resampler, RMSMeter, VADDetector, AGC


class UIFormatter:
    """Docker-friendly UI formatter for audio device scanner."""

    # Box drawing characters
    TOP_LEFT = "┌"
    TOP_RIGHT = "┐"
    BOTTOM_LEFT = "└"
    BOTTOM_RIGHT = "┘"
    HORIZONTAL = "─"
    VERTICAL = "│"
    CROSS = "┼"
    T_DOWN = "┬"
    T_UP = "┴"
    T_LEFT = "┤"
    T_RIGHT = "├"

    # VU meter characters
    EMPTY_BAR = "░"
    FILLED_BAR = "▓"

    @staticmethod
    def detect_color_support() -> bool:
        """Check if terminal supports colors."""
        import os
        if os.environ.get("NO_COLOR"):
            return False
        return sys.stdout.isatty()

    @staticmethod
    def colorize(text: str, color: str) -> str:
        """Apply ANSI color if supported."""
        if not UIFormatter.detect_color_support():
            return text

        colors = {
            "cyan": "\033[36m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "red": "\033[31m",
            "gray": "\033[90m",
            "bold": "\033[1m",
            "reset": "\033[0m",
        }

        color_code = colors.get(color, "")
        reset = colors["reset"]
        return f"{color_code}{text}{reset}"

    @staticmethod
    def format_banner(title: str) -> str:
        """Create a bordered banner."""
        width = 70
        side = UIFormatter.VERTICAL
        top = UIFormatter.TOP_LEFT + UIFormatter.HORIZONTAL * (width - 2) + UIFormatter.TOP_RIGHT
        bottom = UIFormatter.BOTTOM_LEFT + UIFormatter.HORIZONTAL * (width - 2) + UIFormatter.BOTTOM_RIGHT
        padding = (width - 2 - len(title)) // 2
        title_line = f"{side} {title.center(width - 4)} {side}"

        return f"\n{top}\n{title_line}\n{bottom}\n"

    @staticmethod
    def format_vad_bar(vad_prob: float, threshold: float, width: int = 10) -> str:
        """Create ASCII progress bar for VAD level."""
        filled = int((vad_prob / 1.0) * width)
        bar = UIFormatter.FILLED_BAR * filled + UIFormatter.EMPTY_BAR * (width - filled)

        if vad_prob > threshold:
            return UIFormatter.colorize(bar, "green")
        elif vad_prob > threshold * 0.66:
            return UIFormatter.colorize(bar, "yellow")
        else:
            return UIFormatter.colorize(bar, "gray")

    @staticmethod
    def format_cycle_header(cycle: int, elapsed_sec: float) -> str:
        """Format cycle header with timing."""
        minutes = int(elapsed_sec // 60)
        seconds = int(elapsed_sec % 60)
        millis = int((elapsed_sec % 1) * 1000)
        time_str = f"{minutes:02d}:{seconds:02d}.{millis:03d}"

        width = 70
        header = f"Cycle {cycle}"
        padding = width - len(header) - len(time_str) - 4
        line = f"{header} {' ' * padding} {time_str}"

        top = UIFormatter.TOP_LEFT + UIFormatter.HORIZONTAL * (width - 2) + UIFormatter.TOP_RIGHT
        middle = UIFormatter.VERTICAL + line + UIFormatter.VERTICAL
        bottom = UIFormatter.BOTTOM_LEFT + UIFormatter.HORIZONTAL * (width - 2) + UIFormatter.BOTTOM_RIGHT
        return f"{top}\n{middle}\n{bottom}"

    @staticmethod
    def format_table_separator(col_widths: List[int], junction_char: Optional[str] = None) -> str:
        """Format a table separator line."""
        if junction_char is None:
            junction_char = UIFormatter.CROSS
        parts = []
        for i, width in enumerate(col_widths):
            parts.append(UIFormatter.HORIZONTAL * width)

        # Determine left and right edge characters based on junction type
        if junction_char == UIFormatter.T_DOWN:  # Top separator
            left_edge = UIFormatter.TOP_LEFT
            right_edge = UIFormatter.TOP_RIGHT
        elif junction_char == UIFormatter.CROSS:  # Middle separator
            left_edge = UIFormatter.T_RIGHT
            right_edge = UIFormatter.T_LEFT
        elif junction_char == UIFormatter.T_UP:  # Bottom separator
            left_edge = UIFormatter.BOTTOM_LEFT
            right_edge = UIFormatter.BOTTOM_RIGHT
        else:
            left_edge = UIFormatter.T_RIGHT
            right_edge = UIFormatter.T_LEFT

        return left_edge + junction_char.join(parts) + right_edge

    @staticmethod
    def format_table_row(
        values: List[str], col_widths: List[int], is_header: bool = False
    ) -> str:
        """Format a table row."""
        formatted = []
        for value, width in zip(values, col_widths):
            if is_header:
                formatted.append(value.center(width))
            else:
                formatted.append(value.ljust(width))
        return UIFormatter.VERTICAL + UIFormatter.VERTICAL.join(formatted) + UIFormatter.VERTICAL

    @staticmethod
    def format_device_table(devices_data: List[dict], threshold: float) -> str:
        """Format device scan results as table."""
        col_widths = [4, 29, 9, 10, 12]  # Total: 64 content + 7 borders/separators = 71

        output = []
        # Header
        sep_top = UIFormatter.format_table_separator(col_widths, UIFormatter.T_DOWN)
        output.append(sep_top)

        header_vals = ["ID", "Device Name", "VAD", "RMS", "Status"]
        header = UIFormatter.format_table_row(header_vals, col_widths, is_header=True)
        output.append(header)

        sep_header = UIFormatter.format_table_separator(col_widths, UIFormatter.CROSS)
        output.append(sep_header)

        # Rows
        for device_info in devices_data:
            device_id = str(device_info["device_id"]).rjust(2)
            device_name = device_info["name"][:24]
            vad_str = f"{device_info['vad']:.3f}"
            rms_str = f"{device_info['rms']:.4f}"
            # Get bar without colors for proper width calculation in table
            filled = int((device_info["vad"] / 1.0) * 10)
            status = UIFormatter.FILLED_BAR * filled + UIFormatter.EMPTY_BAR * (10 - filled)

            row_vals = [device_id, device_name, vad_str, rms_str, status]
            row = UIFormatter.format_table_row(row_vals, col_widths)
            output.append(row)

        sep_bottom = UIFormatter.format_table_separator(col_widths, UIFormatter.T_UP)
        output.append(sep_bottom)

        return "\n".join(output)

    @staticmethod
    def format_detection_summary(device: AudioDeviceInfo, vad_prob: float, cycles: int, elapsed: float) -> str:
        """Format detection success summary."""
        title = "✓ DEVICE DETECTED"
        banner = UIFormatter.format_banner(title)

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        millis = int((elapsed % 1) * 1000)
        time_str = f"{minutes:02d}:{seconds:02d}.{millis:03d}"

        summary = f"""
  Device ID:        {device.device_id}
  Device Name:      {device.name}
  Channels:         {device.channels}
  Sample Rate:      {int(device.sample_rate)} Hz
  Host API:         {device.api}

  VAD Probability:  {vad_prob:.4f}
  Cycles Elapsed:   {cycles}
  Time Elapsed:     {time_str}

  Status: SPEECH DETECTED ✓
"""
        return banner + summary


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
        self.start_time = time.time()

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

        # Display startup banner
        banner = UIFormatter.format_banner("Audio Input Device Scanner - VAD Detection")
        print(banner)
        print(f"Found {len(self.devices)} audio input device(s):")
        for device in self.devices:
            print(f"  [{device.device_id:2d}] {device.name:30s} ({int(device.channels)}ch, {int(device.sample_rate)}Hz)")

    def _read_with_timeout(self, stream: sd.InputStream, chunk_samples: int, timeout_sec: float = 1.0) -> Tuple[Optional[np.ndarray], bool]:
        """
        Read from stream with timeout to prevent hanging on unresponsive devices.

        Args:
            stream: Audio input stream
            chunk_samples: Number of samples to read
            timeout_sec: Timeout in seconds (default: 1.0)

        Returns:
            Tuple of (audio_data, success) where success=False if timeout occurred
        """
        result_queue = Queue()

        def read_thread():
            try:
                audio, overflowed = stream.read(chunk_samples)
                result_queue.put((audio, overflowed, True))
            except Exception as e:
                result_queue.put((None, False, False))

        thread = threading.Thread(target=read_thread, daemon=True)
        thread.start()

        try:
            audio, overflowed, success = result_queue.get(timeout=timeout_sec)
            return audio if success else None, overflowed if success else False
        except Empty:
            # Timeout occurred
            return None, False

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
                latency='low',
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
        # Skip ALSA plugins that are unlikely to work for real-time audio capture
        skip_devices = {'sysdefault', 'lavrate', 'samplerate', 'speexrate', 'upmix', 'vdownmix', 'speex'}
        active_streams = {}

        for device in self.devices:
            # Skip ALSA plugin devices that commonly cause issues
            if any(skip_name in device.name for skip_name in skip_devices):
                continue

            # Open device with timeout to prevent hanging
            result_queue = Queue()
            def open_thread():
                stream, resampler = self._open_device_stream(device)
                result_queue.put((stream, resampler))

            thread = threading.Thread(target=open_thread, daemon=True)
            thread.start()

            try:
                stream, resampler = result_queue.get(timeout=2.0)
                if stream is not None:
                    active_streams[device.device_id] = {
                        "device": device,
                        "stream": stream,
                        "resampler": resampler,
                    }
            except Empty:
                # Device open timed out, skip it
                pass

        if not active_streams:
            print("Error: Could not open any device streams!")
            return None, 0.0

        print(f"Scanning {len(active_streams)} device stream(s) for voice activity...\n")

        try:
            cycle = 0
            while True:
                cycle += 1
                vad_readings = {}  # Store VAD for each device in this cycle

                # Calculate elapsed time
                elapsed = time.time() - self.start_time

                # Print cycle header
                cycle_header = UIFormatter.format_cycle_header(cycle, elapsed)
                print(cycle_header)

                # Scan all active devices in this cycle
                for device_id, device_info in active_streams.items():
                    device = device_info["device"]
                    stream = device_info["stream"]
                    resampler = device_info["resampler"]

                    try:
                        # Read audio chunk from device with timeout
                        chunk_samples = int(
                            device.sample_rate * self.chunk_duration_ms / 1000
                        )
                        audio, overflowed = self._read_with_timeout(stream, chunk_samples, timeout_sec=0.5)

                        if audio is None:
                            # Device read timed out, skip this device
                            continue

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
                            "device_id": device.device_id,
                            "name": device.name,
                            "vad": vad_prob,
                            "rms": rms
                        }

                    except Exception as e:
                        print(f"  Error reading from device {device.device_id}: {e}")
                        continue

                # Format and print device table
                if vad_readings:
                    devices_data = list(vad_readings.values())
                    table = UIFormatter.format_device_table(devices_data, self.vad_threshold)
                    print(table)

                    # Find device with highest VAD in this cycle
                    max_device_id = max(vad_readings.keys(),
                                       key=lambda k: vad_readings[k]["vad"])
                    max_reading = vad_readings[max_device_id]
                    max_vad = max_reading["vad"]
                    max_device = next((d for d in self.devices if d.device_id == max_device_id), None)

                    # Print summary
                    status = "✓ DETECTED" if max_vad > self.vad_threshold else "Below threshold"
                    print(f"\n  ► Highest: Device {max_device_id} ({max_device.name}) = {max_vad:.3f} | {status} ({self.vad_threshold})")

                    # Check if threshold exceeded
                    if max_vad > self.vad_threshold:
                        # Print formatted detection summary
                        summary = UIFormatter.format_detection_summary(max_device, max_vad, cycle, elapsed)
                        print(summary)
                        return max_device, max_vad
                    else:
                        print()

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
            print(f"\n✓ Device {detected_device.device_id} is ready for audio capture!")

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
