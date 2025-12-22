#!/usr/bin/env python3

"""
Audio processing components for device enumeration and VAD detection.

Pipeline:
  Raw audio (device sample rate)
  -> Resampler (scipy.signal.resample_poly) -> 16kHz
  -> AGC (Automatic Gain Control) -> normalized levels
  -> RMS Meter (on normalized signal, verify AGC)
  -> VAD Detector (ten-vad ONNX)

Uses ten-vad ONNX Python extension for high-performance voice activity detection.
Built from: https://github.com/TEN-framework/ten-vad

Note: For optimal ARM performance (Jetson Nano), consider using pysoxr
      as a drop-in replacement for the Resampler class.
"""

import numpy as np
from scipy import signal
import ten_vad_python


class Resampler:
    """
    Audio resampler using scipy.signal.resample_poly.
    Fast polyphase resampler suitable for VAD preprocessing.

    Note: For maximum performance on ARM devices, consider replacing this
          with pysoxr (python-soxr package) which uses libsoxr C library.
    """

    def __init__(self, input_rate: int, output_rate: int = 16000, channels: int = 1):
        """
        Initialize resampler.

        Args:
            input_rate: Input sample rate in Hz
            output_rate: Output sample rate in Hz (default: 16000 for VAD)
            channels: Number of audio channels (default: 1 for mono)
        """
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.channels = channels

        if input_rate == output_rate:
            # No resampling needed
            self.ratio = None
        else:
            # Compute rational resampling ratio
            from math import gcd

            g = gcd(output_rate, input_rate)
            self.up = output_rate // g
            self.down = input_rate // g
            self.ratio = (self.up, self.down)

    def resample(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Input audio samples (1D numpy array for mono)

        Returns:
            Resampled audio at output_rate
        """
        if self.ratio is None:
            # No resampling needed
            return audio

        if len(audio) == 0:
            return audio

        # scipy.signal.resample_poly for fast polyphase resampling
        resampled = signal.resample_poly(audio, self.ratio[0], self.ratio[1])

        return resampled.astype(np.float32)


class AGC:
    """
    Automatic Gain Control (AGC) for normalizing audio levels.

    Helps VAD detection by amplifying low-volume signals and
    preventing high-energy noise from dominating.
    """

    def __init__(self, target_level: float = 0.1, max_gain: float = 10.0):
        """
        Initialize AGC.

        Args:
            target_level: Target RMS level for normalization (default: 0.1)
            max_gain: Maximum gain to prevent over-amplification (default: 10.0)
        """
        self.target_level = target_level
        self.max_gain = max_gain

    def apply(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply automatic gain control to audio.

        Args:
            audio: Input audio samples (numpy array, float32)

        Returns:
            Gain-adjusted audio at target level
        """
        if len(audio) == 0:
            return audio

        # Compute RMS of input
        rms = np.sqrt(np.mean(np.square(audio)))

        if rms < 1e-6:  # Silence threshold
            return audio

        # Calculate required gain
        gain = self.target_level / rms
        gain = min(gain, self.max_gain)  # Limit maximum gain

        # Apply gain
        return (audio * gain).astype(np.float32)


class RMSMeter:
    """
    Compute Root Mean Square (RMS) of audio signal.
    Optional component for audio level monitoring.
    """

    @staticmethod
    def compute_rms(audio: np.ndarray) -> float:
        """
        Compute RMS (audio level) of audio samples.

        Args:
            audio: Input audio samples (numpy array)

        Returns:
            RMS value (0.0 to 1.0 for normalized audio)
        """
        if len(audio) == 0:
            return 0.0

        rms = np.sqrt(np.mean(np.square(audio)))
        return float(rms)

    @staticmethod
    def compute_rms_db(audio: np.ndarray, reference: float = 1.0) -> float:
        """
        Compute RMS in decibels.

        Args:
            audio: Input audio samples
            reference: Reference level (default: 1.0)

        Returns:
            RMS in dB (20 * log10(rms / reference))
        """
        rms = RMSMeter.compute_rms(audio)
        if rms == 0:
            return -np.inf

        db = 20 * np.log10(rms / reference)
        return float(db)


class VADDetector:
    """
    Voice Activity Detection using ten-vad ONNX.
    High-performance speech detection with ONNX Runtime.

    Requirements:
      - Sample rate: 16000 Hz (required by ten-vad ONNX)
      - Audio chunks: 256 samples (16ms at 16kHz)
      - Audio format: int16 PCM data
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        """
        Initialize VAD detector.

        Args:
            sample_rate: Sample rate in Hz (required: 16000 for ten-vad)
            threshold: VAD threshold for is_voice flag (0.0 to 1.0)
        """
        if sample_rate != 16000:
            raise ValueError("ten-vad requires 16000 Hz sample rate")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.chunk_size = 256  # 16ms at 16kHz

        # Initialize ONNX VAD
        self.vad = ten_vad_python.VAD(hop_size=self.chunk_size, threshold=threshold)

    def get_speech_probability(self, audio: np.ndarray) -> float:
        """
        Get probability of speech presence in audio chunk.

        For best results, provide audio chunks of exactly 256 samples.
        If audio is longer, it will be processed in 256-sample frames
        and the maximum probability returned.

        Args:
            audio: Audio samples at 16kHz (numpy array, float32 or int16)

        Returns:
            Speech probability (0.0 to 1.0)
        """
        if len(audio) == 0:
            return 0.0

        # Convert to int16 PCM format (required by ten-vad ONNX)
        if audio.dtype != np.int16:
            # Assume float32 in range [-1.0, 1.0]
            audio = audio.astype(np.float32)
            # Normalize if needed
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            # Convert to int16
            audio = (audio * 32767).astype(np.int16)

        # Process audio in 256-sample chunks
        max_probability = 0.0

        for start in range(0, len(audio), self.chunk_size):
            chunk = audio[start : start + self.chunk_size]

            # Pad chunk if it's shorter than 256 samples
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode="constant")

            # Process chunk with ONNX VAD
            # Returns (probability, is_voice) tuple
            probability, is_voice = self.vad.process(chunk)

            max_probability = max(max_probability, float(probability))

        return max(0.0, min(1.0, max_probability))  # Clamp to [0, 1]
