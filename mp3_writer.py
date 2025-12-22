#!/usr/bin/env python3

"""
MP3 Writer - Accepts audio chunks and encodes to MP3 via ffmpeg.

Unlike MP3Recorder, this does NOT open an audio device.
It receives chunks from an external audio stream.

Usage:
    writer = MP3Writer(output_file, sample_rate=16000, bitrate=128)
    writer.start()

    for chunk in audio_stream:
        writer.write_chunk(chunk)

    writer.stop()
"""

import subprocess
import threading
import queue
import time
import os
from typing import Optional
import numpy as np


class MP3Writer:
    """
    Writes audio chunks to MP3 file via ffmpeg subprocess.

    Thread-safe: can be called from multiple threads.
    Non-blocking: uses internal queue for buffering.
    """

    def __init__(
        self,
        output_file: str,
        sample_rate: int = 16000,
        channels: int = 1,
        bitrate: int = 128,
    ):
        """
        Initialize MP3 writer.

        Args:
            output_file: Output MP3 file path
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono)
            bitrate: MP3 bitrate in kbps (96-320)
        """
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.channels = channels
        self.bitrate = bitrate

        # Internal state
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._write_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._is_running = False

        # Statistics
        self._chunks_written = 0
        self._total_samples = 0
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start the MP3 writer."""
        if self._is_running:
            return

        # Start ffmpeg process
        self._ffmpeg_process = self._start_ffmpeg()

        # Start write thread
        self._stop_event.clear()
        self._write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._write_thread.start()

        self._is_running = True
        self._start_time = time.time()

    def write_chunk(self, audio: np.ndarray) -> None:
        """
        Queue an audio chunk for MP3 encoding.

        Args:
            audio: Float32 audio samples, normalized [-1, 1]
        """
        if not self._is_running:
            return

        try:
            self._audio_queue.put_nowait(audio.copy())
        except queue.Full:
            pass  # Drop chunk if queue is full

    def stop(self) -> None:
        """Stop the MP3 writer and finalize file."""
        if not self._is_running:
            return

        # Signal stop
        self._stop_event.set()

        # Wait for write thread to finish
        if self._write_thread is not None:
            self._write_thread.join(timeout=5.0)

        # Close ffmpeg
        if self._ffmpeg_process is not None:
            try:
                if self._ffmpeg_process.stdin:
                    self._ffmpeg_process.stdin.close()
                self._ffmpeg_process.wait(timeout=5.0)
            except:
                self._ffmpeg_process.terminate()

        self._is_running = False

    def get_stats(self) -> dict:
        """Get writer statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "chunks_written": self._chunks_written,
            "total_samples": self._total_samples,
            "duration_sec": self._total_samples / self.sample_rate if self.sample_rate else 0,
            "elapsed_sec": elapsed,
        }

    def _start_ffmpeg(self) -> subprocess.Popen:
        """Start ffmpeg subprocess for MP3 encoding."""
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'f32le',                    # Input: 32-bit float PCM
            '-ar', str(self.sample_rate),     # Sample rate
            '-ac', str(self.channels),        # Channels
            '-i', 'pipe:0',                   # Input from stdin
            '-b:a', f'{self.bitrate}k',       # Output bitrate
            '-q:a', '2',                      # Quality
            self.output_file,
        ]

        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _write_loop(self) -> None:
        """Background thread: write queued chunks to ffmpeg."""
        while not self._stop_event.is_set():
            try:
                audio = self._audio_queue.get(timeout=0.1)

                # Convert to bytes
                audio_bytes = audio.astype(np.float32).tobytes()

                # Write to ffmpeg
                if self._ffmpeg_process and self._ffmpeg_process.stdin:
                    self._ffmpeg_process.stdin.write(audio_bytes)
                    self._chunks_written += 1
                    self._total_samples += len(audio)

            except queue.Empty:
                continue
            except Exception as e:
                break

        # Drain remaining queue
        while not self._audio_queue.empty():
            try:
                audio = self._audio_queue.get_nowait()
                audio_bytes = audio.astype(np.float32).tobytes()
                if self._ffmpeg_process and self._ffmpeg_process.stdin:
                    self._ffmpeg_process.stdin.write(audio_bytes)
                    self._chunks_written += 1
                    self._total_samples += len(audio)
            except:
                break


if __name__ == "__main__":
    # Quick test
    print("Testing MP3Writer...")

    # Generate test audio (3 seconds of 440Hz sine)
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, sr * duration)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Write to MP3
    writer = MP3Writer("/tmp/test_mp3writer.mp3", sample_rate=sr)
    writer.start()

    # Write in 100ms chunks
    chunk_size = sr // 10
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        writer.write_chunk(chunk)

    writer.stop()

    stats = writer.get_stats()
    print(f"✓ Written {stats['chunks_written']} chunks")
    print(f"  Duration: {stats['duration_sec']:.2f}s")

    if os.path.exists("/tmp/test_mp3writer.mp3"):
        size = os.path.getsize("/tmp/test_mp3writer.mp3")
        print(f"✓ File created: {size/1024:.1f} KB")
    else:
        print("✗ File not created")
