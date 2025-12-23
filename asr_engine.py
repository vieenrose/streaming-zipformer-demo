#!/usr/bin/env python3

"""
ASR Engine Pool for Step 6: Parallel ASR Integration

Manages 3 sherpa-onnx streaming ASR engines in parallel.
- Load models on demand
- Feed audio chunks simultaneously to all 3 models
- Return snapshot results for safe concurrent access
- Thread-safe per-model state management

Architecture:
  Audio Stream (from device)
      ↓
  ASREnginePool.feed_audio_chunk()
      ├→ ModelInstance 1 (small-bilingual) → TranscriptionState
      ├→ ModelInstance 2 (medium-bilingual) → TranscriptionState
      └→ ModelInstance 3 (multilingual) → TranscriptionState
      ↓
  ASREnginePool.get_results() → Dict[model_id, StreamResult]
"""

import threading
import time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np

import sherpa_onnx

from asr_config import ASRConfig, RecognitionConfig

# Initialize OpenCC converter (zh-TW to zh-CN)
try:
    import opencc
    OPENCC_AVAILABLE = True
    OPENCC_CONVERTER = opencc.OpenCC('t2s')  # Traditional → Simplified
except ImportError:
    print("Warning: opencc-python-reimplemented not available. zh-TW to zh-CN conversion disabled.")
    OPENCC_AVAILABLE = False
    OPENCC_CONVERTER = None
except Exception as e:
    print(f"Warning: OpenCC initialization failed: {e}. zh-TW to zh-CN conversion disabled.")
    OPENCC_AVAILABLE = False
    OPENCC_CONVERTER = None


# =========================================================================
# RESULT TRACKING CLASSES
# =========================================================================

@dataclass
class TranscriptionState:
    """Current transcription state for a single model."""
    partial_text: str = ""          # Partial transcription (being processed)
    final_text: str = ""            # Final transcription (committed)
    last_update_time: float = field(default_factory=time.time)
    chunk_count: int = 0            # Number of chunks processed
    is_endpoint_detected: bool = False
    confidence: float = 0.0         # Confidence score (if available)

    def reset(self) -> None:
        """Reset state for new utterance."""
        self.partial_text = ""
        self.final_text = ""
        self.chunk_count = 0
        self.is_endpoint_detected = False


@dataclass
class StreamResult:
    """Snapshot of transcription result (for thread-safe access)."""
    model_id: str                   # Model identifier
    model_name: str                 # Human-readable model name
    partial: str                    # Current partial transcript
    final: str                      # Final transcript
    updated_at: float               # Timestamp of last update
    chunks_processed: int           # Total chunks fed to model
    endpoint_detected: bool         # Speech endpoint detected
    confidence: float = 0.0         # Confidence score
    latency_ms: float = 0.0         # Processing latency


# =========================================================================
# MODEL INSTANCE CLASS
# =========================================================================

class ModelInstance:
    """Manages state and inference for a single ASR model."""

    @staticmethod
    def convert_hotwords_zh_tw_to_zh_cn(hotwords: List[str]) -> List[str]:
        """
        Convert hotwords from zh-TW (Traditional) to zh-CN (Simplified) before sending to ASR.

        Args:
            hotwords: List of hotwords in zh-TW format

        Returns:
            List of hotwords in zh-CN format (or original if OpenCC unavailable)
        """
        if not OPENCC_AVAILABLE or OPENCC_CONVERTER is None:
            # OpenCC not available, return hotwords as-is (zh-TW)
            print(f"Warning: zh-TW → zh-CN conversion not available for hotwords. Using zh-TW directly.")
            return hotwords

        try:
            # Convert each hotword from Traditional (zh-TW) to Simplified (zh-CN)
            converted = [OPENCC_CONVERTER.convert(hw) for hw in hotwords]
            print(f"✓ Converted {len(converted)} hotwords from zh-TW to zh-CN")
            return converted
        except Exception as e:
            print(f"Warning: zh-TW → zh-CN conversion failed: {e}. Using zh-TW hotwords directly.")
            return hotwords

    def __init__(
        self,
        model_id: str,
        model_config,
        recognizer: sherpa_onnx.OnlineRecognizer,
    ):
        """
        Initialize model instance.

        Args:
            model_id: Unique identifier (e.g., "small-bilingual")
            model_config: ModelConfig from asr_config.py
            recognizer: Initialized OnlineRecognizer instance
        """
        self.model_id = model_id
        self.model_config = model_config
        self.recognizer = recognizer

        # Create stream for this model with hotwords if available
        # Note: hotwords_score is set at recognizer level, only hotwords string passed to create_stream
        if model_config.hotwords and model_config.bpe_vocab_path:
            # Convert hotwords from zh-TW (Traditional) to zh-CN (Simplified) before ASR
            converted_hotwords = self.convert_hotwords_zh_tw_to_zh_cn(model_config.hotwords.hotwords)

            # Format hotwords as newline-separated uppercase strings (required for BPE tokenization)
            hotwords_str = "\n".join([hw.upper() for hw in converted_hotwords])
            self.stream = recognizer.create_stream(hotwords=hotwords_str)
        else:
            self.stream = recognizer.create_stream()

        # Transcription state
        self.state = TranscriptionState()
        self._state_lock = threading.RLock()

        # Processing statistics
        self._start_time = time.time()
        self._last_chunk_time = self._start_time

    def feed_audio(self, audio: np.ndarray) -> None:
        """
        Feed audio chunk to model.

        Args:
            audio: Float32 audio samples, normalized to [-1.0, 1.0]
        """
        with self._state_lock:
            # Ensure float32 format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Feed to stream (sample_rate, waveform)
            try:
                self.stream.accept_waveform(
                    RecognitionConfig.SAMPLE_RATE,
                    audio,
                )
                self.state.chunk_count += 1
                self._last_chunk_time = time.time()
            except Exception as e:
                print(f"✗ Error feeding audio to {self.model_id}: {e}")
                return

    def process(self) -> None:
        """
        Process accumulated audio and update transcription state.
        Call this after feeding audio chunks.
        """
        import json

        with self._state_lock:
            try:
                # Decode all ready samples
                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)

                # Get results as JSON (contains is_final field)
                json_str = self.recognizer.get_result_as_json_string(self.stream)
                result_data = json.loads(json_str)

                # Update state based on result
                text = result_data.get("text", "")
                is_final = result_data.get("is_final", False)

                if is_final:
                    self.state.final_text = text
                else:
                    self.state.partial_text = text

                # Check for endpoint (speech end detection)
                self.state.is_endpoint_detected = self.recognizer.is_endpoint(self.stream)

                self.state.last_update_time = time.time()

            except Exception as e:
                print(f"✗ Error processing {self.model_id}: {e}")
                import traceback
                traceback.print_exc()

    def reset(self) -> None:
        """Reset stream and state for new utterance."""
        with self._state_lock:
            try:
                # Create new stream with hotwords if available
                # Note: hotwords_score is set at recognizer level, only hotwords string passed here
                if self.model_config.hotwords and self.model_config.bpe_vocab_path:
                    hotwords_str = "\n".join([hw.upper() for hw in self.model_config.hotwords.hotwords])
                    self.stream = self.recognizer.create_stream(hotwords=hotwords_str)
                else:
                    self.stream = self.recognizer.create_stream()
                self.state.reset()
            except Exception as e:
                print(f"✗ Error resetting {self.model_id}: {e}")

    def get_result_snapshot(self) -> StreamResult:
        """Get thread-safe snapshot of current result."""
        with self._state_lock:
            # Total elapsed time since model creation
            total_time = (time.time() - self._start_time) * 1000

            # Average latency per chunk
            avg_latency = total_time / max(1, self.state.chunk_count)

            return StreamResult(
                model_id=self.model_id,
                model_name=self.model_config.name,
                partial=self.state.partial_text,
                final=self.state.final_text,
                updated_at=self.state.last_update_time,
                chunks_processed=self.state.chunk_count,
                endpoint_detected=self.state.is_endpoint_detected,
                latency_ms=avg_latency,
            )


# =========================================================================
# ASR ENGINE POOL CLASS
# =========================================================================

class ASREnginePool:
    """
    Manages 3 sherpa-onnx ASR engines for parallel transcription.

    Usage:
        pool = ASREnginePool()
        pool.load_models()

        # Feed audio from detected device
        audio_chunk = np.array([...], dtype=np.float32)
        pool.feed_audio_chunk(audio_chunk)

        # Get results
        results = pool.get_results()
        for model_id, result in results.items():
            print(f"{result.model_name}: {result.partial}")
    """

    def __init__(self, config: Optional[ASRConfig] = None):
        """
        Initialize engine pool.

        Args:
            config: ASRConfig instance (uses default if None)
        """
        self.config = config or ASRConfig()
        self.models: Dict[str, ModelInstance] = {}
        self._pool_lock = threading.Lock()

        # Processing statistics
        self._start_time = time.time()
        self._audio_chunks_fed = 0
        self._is_loaded = False

    def load_models(self) -> bool:
        """
        Load all 3 ASR models.

        Returns:
            True if all models loaded successfully, False otherwise
        """
        with self._pool_lock:
            print("\n" + "=" * 70)
            print("LOADING ASR MODELS")
            print("=" * 70)

            all_loaded = True

            for model_id, model_config in self.config.models.items():
                try:
                    print(f"\n▶ Loading {model_id}...")
                    print(f"   {model_config.name}")

                    # Create recognizer using from_transducer (recommended for zipformer)
                    kwargs = {
                        "tokens": model_config.tokens_path,
                        "encoder": model_config.encoder_path,
                        "decoder": model_config.decoder_path,
                        "joiner": model_config.joiner_path,
                        "num_threads": model_config.num_threads,
                        "provider": model_config.provider,
                        "decoding_method": RecognitionConfig.DECODING_METHOD,
                        "sample_rate": RecognitionConfig.SAMPLE_RATE,
                        "feature_dim": 80,
                        "enable_endpoint_detection": True,
                    }

                    # Add hotword support if available (requires bpe_vocab + modified_beam_search)
                    if model_config.bpe_vocab_path and model_config.hotwords:
                        kwargs["bpe_vocab"] = model_config.bpe_vocab_path
                        kwargs["modeling_unit"] = model_config.modeling_unit  # cjkchar+bpe for bilingual/multilingual
                        kwargs["hotwords_score"] = model_config.hotwords.boost_score  # Set at recognizer level
                        kwargs["decoding_method"] = "modified_beam_search"  # Required for hotwords

                    # Create recognizer
                    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)

                    # Create model instance
                    model_instance = ModelInstance(
                        model_id=model_id,
                        model_config=model_config,
                        recognizer=recognizer,
                    )

                    self.models[model_id] = model_instance
                    print(f"   ✓ Loaded successfully")

                    # Show configuration
                    print(f"     Threads: {model_config.num_threads} | Decoding: {kwargs['decoding_method']}")

                    # Show hotword status
                    if model_config.hotwords and model_config.bpe_vocab_path:
                        hotwords_list = ", ".join(model_config.hotwords.hotwords)
                        print(f"     Hotwords: ✓ ENABLED | Boost: {model_config.hotwords.boost_score}")
                        print(f"               [{hotwords_list}]")
                    else:
                        reason = "No hotwords config" if not model_config.hotwords else "No bpe.vocab"
                        print(f"     Hotwords: ✗ DISABLED ({reason})")

                except Exception as e:
                    print(f"   ✗ Failed to load: {e}")
                    import traceback
                    traceback.print_exc()
                    all_loaded = False

            self._is_loaded = all_loaded

            if all_loaded:
                print("\n✓ All models loaded successfully")
                print(f"   Total models: {len(self.models)}")
            else:
                print(f"\n⚠ {len(self.config.models) - len(self.models)} models failed to load")

            print("=" * 70 + "\n")

            return all_loaded

    def feed_audio_chunk(self, audio: np.ndarray) -> int:
        """
        Feed audio chunk to all loaded models simultaneously.

        Args:
            audio: Float32 audio samples (mono, 16kHz, 100ms chunks)

        Returns:
            Number of models that received audio chunk
        """
        if not self._is_loaded or not self.models:
            print("✗ No models loaded. Call load_models() first.")
            return 0

        count = 0
        for model_instance in self.models.values():
            try:
                model_instance.feed_audio(audio)
                count += 1
            except Exception as e:
                print(f"✗ Error feeding audio to {model_instance.model_id}: {e}")

        self._audio_chunks_fed += 1
        return count

    def process(self) -> None:
        """
        Process accumulated audio and update transcription states.
        Call this periodically after feeding chunks.
        """
        for model_instance in self.models.values():
            try:
                model_instance.process()
            except Exception as e:
                print(f"✗ Error processing {model_instance.model_id}: {e}")

    def get_results(self) -> Dict[str, StreamResult]:
        """
        Get thread-safe snapshot of all transcription results.

        Returns:
            Dict mapping model_id -> StreamResult
        """
        results = {}
        for model_id, model_instance in self.models.items():
            try:
                results[model_id] = model_instance.get_result_snapshot()
            except Exception as e:
                print(f"✗ Error getting result from {model_id}: {e}")

        return results

    def reset_all(self) -> None:
        """Reset all models for new utterance."""
        for model_instance in self.models.values():
            try:
                model_instance.reset()
            except Exception as e:
                print(f"✗ Error resetting {model_instance.model_id}: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.models.clear()
        self._is_loaded = False

    def get_statistics(self) -> Dict[str, any]:
        """Get pool statistics."""
        return {
            "models_loaded": len(self.models),
            "audio_chunks_fed": self._audio_chunks_fed,
            "uptime_ms": (time.time() - self._start_time) * 1000,
        }


# =========================================================================
# TESTING / DEBUG
# =========================================================================

def test_pool_creation():
    """Test creating and loading ASR engine pool."""
    print("\n" + "=" * 70)
    print("TEST: ASR Engine Pool Creation")
    print("=" * 70 + "\n")

    try:
        # Create pool
        pool = ASREnginePool()
        print("✓ Pool created")

        # Load models
        success = pool.load_models()
        if not success:
            print("✗ Failed to load models")
            return False

        # Get initial results
        results = pool.get_results()
        print(f"\n✓ Retrieved results from {len(results)} models:")
        for model_id, result in results.items():
            print(f"   {model_id}: {result.model_name}")

        # Cleanup
        pool.cleanup()
        print("\n✓ Cleanup successful")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pool_creation()
    exit(0 if success else 1)
