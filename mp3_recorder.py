#!/usr/bin/env python3

"""
MP3 Recorder for streaming audio capture to MP3 format.

Records raw audio samples from an audio input device and encodes them
to MP3 format using ffmpeg in real-time via subprocess streaming.

Pipeline:
  Audio Device (raw PCM) -> sounddevice read
  -> Audio Capture Thread (pushes to queue)
  -> MP3 Encoding Thread (reads from queue)
  -> ffmpeg subprocess (encodes to MP3)
  -> MP3 file (on disk)
"""

import os
import sys
import time
import threading
import subprocess
import queue
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
import sounddevice as sd

from audio_device_enum import enumerate_input_devices, AudioDeviceInfo


class MP3Recorder:
    """
    Records audio from a device and encodes to MP3 format in real-time.

    Uses threading for:
    - Audio capture: Reads chunks from sounddevice
    - MP3 encoding: Streams chunks to ffmpeg subprocess

    Non-blocking design allows recording to run in background while
    main program continues with ASR pipeline.
    """

    def __init__(
        self,
        device_id: int,
        output_file: Optional[str] = None,
        bitrate: int = 128,
        sample_rate: Optional[int] = None,
        channels: int = 1,
        chunk_duration_ms: int = 100,
    ):
        """
        Initialize MP3Recorder.

        Args:
            device_id: Audio device ID to record from
            output_file: Output MP3 file path (auto-generated if None)
            bitrate: MP3 bitrate in kbps (default: 128, range: 96-320)
            sample_rate: Sample rate in Hz (auto-detected if None)
            channels: Number of audio channels (default: 1 = mono)
            chunk_duration_ms: Duration of audio chunks in milliseconds (default: 100)
        """
        # Validate device
        devices = enumerate_input_devices()
        self.device = next((d for d in devices if d.device_id == device_id), None)
        if self.device is None:
            raise ValueError(f"Device {device_id} not found")

        # Store configuration
        self.device_id = device_id
        self.bitrate = bitrate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms

        # Auto-detect sample rate if not provided
        if sample_rate is None:
            self.sample_rate = int(self.device.sample_rate)
        else:
            self.sample_rate = sample_rate

        # Generate output filename
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"recording_{self.device_id}_{timestamp}.mp3"
        else:
            self.output_file = output_file

        # Calculate chunk parameters
        self.chunk_samples = int(self.sample_rate * self.chunk_duration_ms / 1000)

        # State management
        self._recording = False
        self._stream = None
        self._ffmpeg_process = None

        # Threading
        self._audio_queue = queue.Queue(maxsize=10)  # Buffer up to 10 chunks
        self._capture_thread = None
        self._encoding_thread = None
        self._stop_event = threading.Event()

        # Statistics
        self._total_samples = 0
        self._chunks_captured = 0
        self._chunks_encoded = 0
        self._start_time = None

    def start(self) -> None:
        """
        Start background recording to MP3.

        Raises:
            RuntimeError: If already recording or if device cannot be opened.
        """
        if self._recording:
            raise RuntimeError("Already recording")

        print(f"\nStarting MP3 recording...")
        print(f"  Device: {self.device.device_id} ({self.device.name})")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Bitrate: {self.bitrate} kbps")
        print(f"  Output: {self.output_file}")

        # Open audio input stream
        try:
            self._stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                dtype=np.float32,
                latency='low',
            )
            self._stream.start()
        except Exception as e:
            raise RuntimeError(f"Failed to open device {self.device_id}: {e}")

        # Start ffmpeg subprocess for MP3 encoding
        try:
            self._ffmpeg_process = self._start_ffmpeg_encoder()
        except Exception as e:
            self._stream.stop()
            self._stream.close()
            raise RuntimeError(f"Failed to start ffmpeg: {e}")

        # Reset state
        self._recording = True
        self._stop_event.clear()
        self._start_time = time.time()
        self._total_samples = 0
        self._chunks_captured = 0
        self._chunks_encoded = 0

        # Start capture and encoding threads
        self._capture_thread = threading.Thread(target=self._capture_audio, daemon=True)
        self._encoding_thread = threading.Thread(target=self._encode_to_ffmpeg, daemon=True)
        self._capture_thread.start()
        self._encoding_thread.start()

        print("✓ Recording started")

    def stop(self) -> None:
        """
        Stop recording and finalize MP3 file.

        Waits for all queued audio to be encoded and closes ffmpeg process.
        """
        if not self._recording:
            return

        print(f"\nStopping MP3 recording...")

        # Signal threads to stop
        self._stop_event.set()

        # Wait for capture thread to finish
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            if self._capture_thread.is_alive():
                print("  Warning: Capture thread did not finish in time")

        # Wait for encoding thread to finish
        if self._encoding_thread is not None:
            self._encoding_thread.join(timeout=5.0)
            if self._encoding_thread.is_alive():
                print("  Warning: Encoding thread did not finish in time")

        # Finalize ffmpeg encoding
        try:
            if self._ffmpeg_process is not None:
                # Close stdin to signal ffmpeg end-of-stream
                if self._ffmpeg_process.stdin is not None:
                    self._ffmpeg_process.stdin.close()

                # Wait for ffmpeg to finish (timeout: 5 seconds)
                try:
                    self._ffmpeg_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    print("  Warning: ffmpeg did not finish in time, terminating")
                    self._ffmpeg_process.terminate()
                    self._ffmpeg_process.wait(timeout=2.0)

        except Exception as e:
            print(f"  Warning: Error closing ffmpeg: {e}")

        # Close stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"  Warning: Error closing stream: {e}")

        # Print statistics
        elapsed = time.time() - self._start_time if self._start_time else 0
        file_size = os.path.getsize(self.output_file) if os.path.exists(self.output_file) else 0

        print(f"✓ Recording stopped")
        print(f"  File: {self.output_file}")
        print(f"  Size: {file_size / 1024 / 1024:.2f} MB")
        print(f"  Duration: {elapsed:.2f} seconds")
        print(f"  Chunks: {self._chunks_captured} captured, {self._chunks_encoded} encoded")

        self._recording = False

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def get_output_path(self) -> str:
        """Get the output MP3 file path."""
        return self.output_file

    def _start_ffmpeg_encoder(self) -> subprocess.Popen:
        """
        Start ffmpeg subprocess for MP3 encoding.

        Returns:
            ffmpeg subprocess (Popen object)
        """
        # ffmpeg command for MP3 encoding
        # Input: raw PCM float32 mono audio from stdin
        # Output: MP3 file with specified bitrate
        cmd = [
            'ffmpeg',
            '-f', 'f32le',                    # Input format: 32-bit float PCM
            '-ar', str(self.sample_rate),     # Input sample rate
            '-ac', str(self.channels),        # Input channels
            '-i', 'pipe:0',                   # Input from stdin
            '-b:a', f'{self.bitrate}k',       # Output bitrate
            '-q:a', '9',                      # Quality (9 = best compression)
            self.output_file,                 # Output file
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered for real-time streaming
            )
            return process
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg with libmp3lame support")
        except Exception as e:
            raise RuntimeError(f"Failed to start ffmpeg: {e}")

    def _capture_audio(self) -> None:
        """
        Background thread: Capture audio from device and push to encoding queue.

        Runs until _stop_event is set. Handles device errors gracefully.
        """
        try:
            while not self._stop_event.is_set():
                try:
                    # Read audio chunk from device
                    audio, overflowed = self._stream.read(self.chunk_samples)

                    if overflowed:
                        print(f"  Warning: Buffer overflow on device {self.device_id}")

                    # Flatten if needed
                    audio = audio.reshape(-1)

                    # Push to encoding queue (non-blocking with timeout)
                    try:
                        self._audio_queue.put(audio, timeout=1.0)
                        self._chunks_captured += 1
                        self._total_samples += len(audio)
                    except queue.Full:
                        print("  Warning: Encoding queue full, dropping audio chunk")

                except Exception as e:
                    print(f"  Error reading from device: {e}")
                    break

        except Exception as e:
            print(f"  Fatal error in capture thread: {e}")
        finally:
            # Signal end of stream by putting sentinel value
            try:
                self._audio_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

    def _encode_to_ffmpeg(self) -> None:
        """
        Encode audio chunks from queue to MP3 via ffmpeg.

        Runs until None (sentinel) is received from queue.
        """
        try:
            while True:
                try:
                    # Get audio chunk from queue (timeout: prevent hanging)
                    audio = self._audio_queue.get(timeout=1.0)

                    # Check for sentinel value (end of stream)
                    if audio is None:
                        break

                    # Convert to float32 PCM bytes
                    audio_bytes = audio.astype(np.float32).tobytes()

                    # Write to ffmpeg stdin
                    try:
                        self._ffmpeg_process.stdin.write(audio_bytes)
                        self._ffmpeg_process.stdin.flush()
                        self._chunks_encoded += 1
                    except (BrokenPipeError, IOError) as e:
                        print(f"  Error writing to ffmpeg: {e}")
                        break

                except queue.Empty:
                    # Timeout while waiting for chunk
                    if self._stop_event.is_set():
                        # Recording stopped, drain queue and exit
                        while not self._audio_queue.empty():
                            try:
                                audio = self._audio_queue.get_nowait()
                                if audio is not None:
                                    audio_bytes = audio.astype(np.float32).tobytes()
                                    self._ffmpeg_process.stdin.write(audio_bytes)
                                    self._chunks_encoded += 1
                            except queue.Empty:
                                break
                        break

        except Exception as e:
            print(f"  Fatal error in encode thread: {e}")


# For reference/future use: Direct encoding function
def test_mp3_encoding_simple():
    """
    Simple test to verify MP3 encoding works (for debugging).
    """
    import soundfile as sf

    # Create test audio (1 second of sine wave)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Save as WAV
    test_wav = "test_audio.wav"
    sf.write(test_wav, audio, sr)

    # Convert to MP3 using ffmpeg
    test_mp3 = "test_audio.mp3"
    cmd = [
        'ffmpeg',
        '-i', test_wav,
        '-b:a', '128k',
        test_mp3,
        '-y',
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    print(f"✓ Test MP3 created: {test_mp3}")
    print(f"  Size: {os.path.getsize(test_mp3) / 1024:.1f} KB")


if __name__ == "__main__":
    # Example usage
    test_mp3_encoding_simple()
