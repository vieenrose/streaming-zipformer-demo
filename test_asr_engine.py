#!/usr/bin/env python3

"""
Unit and integration tests for ASR Engine Pool (Step 6).

Tests:
1. Configuration validation
2. Model loading (all 3 models)
3. Audio chunk feeding and processing
4. Result retrieval and snapshots
5. Thread safety
6. Error handling
"""

import unittest
import time
import numpy as np
import tempfile
import os
from typing import Dict

from asr_config import ASRConfig, RecognitionConfig, ModelConfig
from asr_engine import (
    ASREnginePool,
    ModelInstance,
    TranscriptionState,
    StreamResult,
)


# =========================================================================
# TEST CONFIGURATION AND UTILITIES
# =========================================================================

class TestAudioGenerator:
    """Generate test audio data."""

    @staticmethod
    def generate_silence(duration_ms: int, sample_rate: int = 16000) -> np.ndarray:
        """Generate silent audio."""
        num_samples = int(sample_rate * duration_ms / 1000)
        return np.zeros(num_samples, dtype=np.float32)

    @staticmethod
    def generate_sine_wave(
        frequency: float = 440,
        duration_ms: int = 100,
        sample_rate: int = 16000,
        amplitude: float = 0.3,
    ) -> np.ndarray:
        """Generate sine wave audio."""
        num_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples)
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        return audio.astype(np.float32)

    @staticmethod
    def generate_noise(
        duration_ms: int = 100,
        sample_rate: int = 16000,
        amplitude: float = 0.1,
    ) -> np.ndarray:
        """Generate white noise."""
        num_samples = int(sample_rate * duration_ms / 1000)
        audio = amplitude * np.random.randn(num_samples)
        return audio.astype(np.float32)


# =========================================================================
# UNIT TESTS
# =========================================================================

class TestASRConfig(unittest.TestCase):
    """Test ASR configuration."""

    def test_config_creation(self):
        """Test creating ASR config."""
        config = ASRConfig()
        self.assertIsNotNone(config)
        self.assertEqual(len(config.models), 3)

    def test_config_validation(self):
        """Test config validation."""
        config = ASRConfig()
        is_valid, results = config.validate_all()

        self.assertTrue(is_valid, f"Config validation failed: {results}")
        for model_id, msg in results.items():
            self.assertEqual(msg, "Valid", f"{model_id} validation failed: {msg}")

    def test_model_retrieval(self):
        """Test retrieving individual models."""
        config = ASRConfig()

        model = config.get_model("small-bilingual")
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "Small Bilingual (zh-en 2023-02-16)")

    def test_model_list(self):
        """Test listing all models."""
        config = ASRConfig()
        models = config.list_models()

        self.assertEqual(len(models), 3)
        self.assertIn("small-bilingual", models)
        self.assertIn("medium-bilingual", models)
        self.assertIn("multilingual", models)

    def test_recognition_config(self):
        """Test recognition settings."""
        self.assertEqual(RecognitionConfig.SAMPLE_RATE, 16000)
        self.assertEqual(RecognitionConfig.CHUNK_DURATION_MS, 100)
        self.assertEqual(RecognitionConfig.DECODING_METHOD, "greedy_search")


class TestTranscriptionState(unittest.TestCase):
    """Test TranscriptionState class."""

    def test_initial_state(self):
        """Test initial state."""
        state = TranscriptionState()

        self.assertEqual(state.partial_text, "")
        self.assertEqual(state.final_text, "")
        self.assertEqual(state.chunk_count, 0)
        self.assertFalse(state.is_endpoint_detected)

    def test_reset(self):
        """Test resetting state."""
        state = TranscriptionState()
        state.partial_text = "hello"
        state.final_text = "hello world"
        state.chunk_count = 5
        state.is_endpoint_detected = True

        state.reset()

        self.assertEqual(state.partial_text, "")
        self.assertEqual(state.final_text, "")
        self.assertEqual(state.chunk_count, 0)
        self.assertFalse(state.is_endpoint_detected)


class TestStreamResult(unittest.TestCase):
    """Test StreamResult snapshot."""

    def test_result_creation(self):
        """Test creating result snapshot."""
        result = StreamResult(
            model_id="test-model",
            model_name="Test Model",
            partial="hello",
            final="hello world",
            updated_at=time.time(),
            chunks_processed=5,
            endpoint_detected=False,
        )

        self.assertEqual(result.model_id, "test-model")
        self.assertEqual(result.partial, "hello")
        self.assertEqual(result.final, "hello world")
        self.assertEqual(result.chunks_processed, 5)


# =========================================================================
# INTEGRATION TESTS
# =========================================================================

class TestASREnginePool(unittest.TestCase):
    """Test ASR Engine Pool integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = ASRConfig()

    def test_pool_creation(self):
        """Test creating engine pool."""
        pool = ASREnginePool(self.config)
        self.assertIsNotNone(pool)
        self.assertEqual(len(pool.models), 0)  # No models loaded yet

    def test_model_loading(self):
        """Test loading all models (INTEGRATION TEST)."""
        print("\n▶ Loading models (may take 1-2 minutes)...")
        pool = ASREnginePool(self.config)
        success = pool.load_models()

        self.assertTrue(success)
        self.assertEqual(len(pool.models), 3)

        # Verify all model IDs
        model_ids = set(pool.models.keys())
        self.assertEqual(
            model_ids,
            {"small-bilingual", "medium-bilingual", "multilingual"},
        )

    def test_audio_feeding_silence(self):
        """Test feeding silence to models (INTEGRATION TEST)."""
        print("\n▶ Testing audio feeding with silence...")
        pool = ASREnginePool(self.config)
        pool.load_models()

        # Feed 10 chunks of silence
        audio = TestAudioGenerator.generate_silence(100)
        for _ in range(10):
            count = pool.feed_audio_chunk(audio)
            self.assertEqual(count, 3)  # All 3 models received audio

        # Process
        pool.process()

        # Get results - should be empty
        results = pool.get_results()
        self.assertEqual(len(results), 3)

        for model_id, result in results.items():
            self.assertEqual(result.partial, "")
            self.assertEqual(result.final, "")
            self.assertEqual(result.chunks_processed, 10)

        pool.cleanup()

    def test_result_snapshot(self):
        """Test result snapshots (thread-safe access)."""
        print("\n▶ Testing result snapshots...")
        pool = ASREnginePool(self.config)
        pool.load_models()

        # Feed audio
        audio = TestAudioGenerator.generate_silence(100)
        pool.feed_audio_chunk(audio)
        pool.process()

        # Get snapshots
        results = pool.get_results()

        for model_id, result in results.items():
            self.assertIsInstance(result, StreamResult)
            self.assertEqual(result.model_id, model_id)
            self.assertGreater(len(result.model_name), 0)
            self.assertIsInstance(result.chunks_processed, int)
            self.assertGreater(result.latency_ms, 0)

        pool.cleanup()

    def test_multiple_chunks(self):
        """Test feeding multiple chunks (INTEGRATION TEST)."""
        print("\n▶ Testing multiple audio chunks...")
        pool = ASREnginePool(self.config)
        pool.load_models()

        num_chunks = 20
        for i in range(num_chunks):
            audio = TestAudioGenerator.generate_sine_wave(duration_ms=100)
            count = pool.feed_audio_chunk(audio)
            self.assertEqual(count, 3)

            pool.process()

            # Every 5th chunk, verify state
            if (i + 1) % 5 == 0:
                results = pool.get_results()
                for model_id, result in results.items():
                    self.assertEqual(result.chunks_processed, i + 1)

        # Final results
        results = pool.get_results()
        for model_id, result in results.items():
            self.assertEqual(result.chunks_processed, num_chunks)

        pool.cleanup()

    def test_reset_functionality(self):
        """Test resetting models for new utterance."""
        print("\n▶ Testing reset functionality...")
        pool = ASREnginePool(self.config)
        pool.load_models()

        # Feed initial chunks
        for _ in range(5):
            audio = TestAudioGenerator.generate_silence(100)
            pool.feed_audio_chunk(audio)
            pool.process()

        results_before = pool.get_results()
        for result in results_before.values():
            self.assertEqual(result.chunks_processed, 5)

        # Reset
        pool.reset_all()

        results_after = pool.get_results()
        for result in results_after.values():
            self.assertEqual(result.chunks_processed, 0)
            self.assertEqual(result.partial, "")
            self.assertEqual(result.final, "")

        pool.cleanup()

    def test_cleanup(self):
        """Test cleanup."""
        print("\n▶ Testing cleanup...")
        pool = ASREnginePool(self.config)
        pool.load_models()
        self.assertEqual(len(pool.models), 3)

        pool.cleanup()
        self.assertEqual(len(pool.models), 0)

    def test_statistics(self):
        """Test statistics tracking."""
        print("\n▶ Testing statistics...")
        pool = ASREnginePool(self.config)
        pool.load_models()

        # Feed some audio
        for _ in range(5):
            audio = TestAudioGenerator.generate_silence(100)
            pool.feed_audio_chunk(audio)

        stats = pool.get_statistics()
        self.assertEqual(stats["models_loaded"], 3)
        self.assertEqual(stats["audio_chunks_fed"], 5)
        self.assertGreater(stats["uptime_ms"], 0)

        pool.cleanup()


# =========================================================================
# STRESS TESTS
# =========================================================================

class TestASREnginePoolStress(unittest.TestCase):
    """Stress tests for ASR Engine Pool."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = ASRConfig()

    def test_rapid_feeding(self):
        """Test rapid audio chunk feeding (STRESS TEST)."""
        print("\n▶ Stress test: Rapid audio feeding...")
        pool = ASREnginePool(self.config)
        pool.load_models()

        start_time = time.time()

        # Feed 100 chunks as fast as possible
        for i in range(100):
            audio = TestAudioGenerator.generate_silence(100)
            pool.feed_audio_chunk(audio)

            # Process every 10th chunk
            if (i + 1) % 10 == 0:
                pool.process()

        elapsed = time.time() - start_time

        results = pool.get_results()
        for result in results.values():
            self.assertEqual(result.chunks_processed, 100)

        print(f"   Processed 100 chunks in {elapsed:.2f}s ({100/elapsed:.1f} chunks/sec)")

        pool.cleanup()

    def test_long_utterance(self):
        """Test processing long utterance (STRESS TEST)."""
        print("\n▶ Stress test: Long utterance (30 seconds)...")
        pool = ASREnginePool(self.config)
        pool.load_models()

        # 30 seconds of audio = 300 chunks (100ms each)
        num_chunks = 300

        for i in range(num_chunks):
            audio = TestAudioGenerator.generate_silence(100)
            pool.feed_audio_chunk(audio)

            # Process every chunk
            pool.process()

            # Print progress every 50 chunks
            if (i + 1) % 50 == 0:
                results = pool.get_results()
                elapsed_sec = (i + 1) * 100 / 1000
                print(f"   Processed {i+1}/{num_chunks} chunks ({elapsed_sec:.1f}s)")

        results = pool.get_results()
        for result in results.values():
            self.assertEqual(result.chunks_processed, num_chunks)

        pool.cleanup()


# =========================================================================
# TEST SUITE
# =========================================================================

def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestASRConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestTranscriptionState))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamResult))
    suite.addTests(loader.loadTestsFromTestCase(TestASREnginePool))
    suite.addTests(loader.loadTestsFromTestCase(TestASREnginePoolStress))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ASR ENGINE POOL TEST SUITE")
    print("=" * 70)

    success = run_tests()

    print("\n" + "=" * 70)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 70 + "\n")

    exit(0 if success else 1)
