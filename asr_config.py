#!/usr/bin/env python3

"""
ASR Configuration and Model Definitions for Step 6

Defines:
- ModelConfig: Configuration for individual ASR models
- ASRConfig: Container for all 3 models with shared settings
- Hotword configurations per model
- Recognition parameters (sample rate, decoding method, etc.)
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

# =========================================================================
# CENTRALIZED HOTWORD LIST (zh-TW: Traditional Chinese)
# =========================================================================
# All ASR models share this single hotword list
# Storage format: zh-TW (Traditional) for UI display
# Conversion: zh-TW → zh-CN (Simplified) before sending to ASR
# =========================================================================

ZH_TW_HOTWORDS = [
    # Person names in pure zh-TW
    "張偉明",  # 张伟明 (zh-CN)
    "李建國",  # 李建国 (zh-CN)
    "王曉紅",  # 王小红 (zh-CN)
    "陳志強",  # 陈志强 (zh-CN)
    "劉芳芳",  # 刘芳芳 (zh-CN)
    "楊光明",  # 杨光明 (zh-CN)
    "趙文靜",  # 赵文静 (zh-CN)
    "黃麗華",  # 黄丽华 (zh-CN)
    "周永康",  # 周永康 (zh-CN, unchanged in zh-TW)
    "吳天明",  # 吴天明 (zh-CN)

    # Person names in mixed zh-TW/en
    "Mark Ma",   # Mark Ma (unchanged)
    "Ken Li",    # Ken Li (unchanged)

    # English hotwords
    "LILY",
    "LUCY",
    "EMMA",
    "KEVIN",
    "CINDY",
    "TONY",
    "AMY",
    "DAVID",
    "JESSICA",
    "MICHAEL",
]

HOTWORD_BOOST_SCORE = 1.5


# =========================================================================
# MODEL CONFIGURATION
# =========================================================================

@dataclass
class HotwordConfig:
    """Hotword configuration for a model."""
    hotwords: List[str]  # List of hotwords to boost
    boost_score: float   # Boost score for hotwords (typically 1.0-5.0)

    def is_valid(self) -> bool:
        """Check if hotword config is valid."""
        return len(self.hotwords) > 0 and self.boost_score > 0.0


@dataclass
class ModelConfig:
    """Configuration for a single ASR model."""
    name: str                          # Display name
    encoder_path: str                  # Path to encoder ONNX model
    decoder_path: str                  # Path to decoder ONNX model
    joiner_path: str                   # Path to joiner ONNX model
    tokens_path: str                   # Path to tokens.txt
    bpe_model_path: Optional[str]      # Path to BPE model (optional)
    bpe_vocab_path: Optional[str]      # Path to BPE vocab for hotwords (optional)
    hotwords: Optional[HotwordConfig]  # Hotword configuration (optional)
    modeling_unit: str = "cjkchar+bpe" # Modeling unit for BPE tokenization (bpe, cjkchar, cjkchar+bpe)
    sample_rate: int = 16000           # Sample rate for model (IMPORTANT: ASR requires 16kHz)
    num_threads: int = 4               # Number of threads for ONNX inference (4 per model)
    provider: str = "cpu"              # ONNX execution provider ("cpu" or "cuda")

    def validate(self) -> tuple[bool, str]:
        """Validate model configuration."""
        # Check required files exist
        required_files = {
            "encoder": self.encoder_path,
            "decoder": self.decoder_path,
            "joiner": self.joiner_path,
            "tokens": self.tokens_path,
        }

        for name, path in required_files.items():
            if not os.path.exists(path):
                return False, f"Missing {name}: {path}"

        # Check optional files if specified
        if self.bpe_model_path and not os.path.exists(self.bpe_model_path):
            return False, f"Missing BPE model: {self.bpe_model_path}"

        if self.bpe_vocab_path and not os.path.exists(self.bpe_vocab_path):
            return False, f"Missing BPE vocab: {self.bpe_vocab_path}"

        return True, "Valid"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to sherpa-onnx."""
        config = {
            "encoder": self.encoder_path,
            "decoder": self.decoder_path,
            "joiner": self.joiner_path,
            "tokens": self.tokens_path,
            "num_threads": self.num_threads,
            "provider": self.provider,
        }

        if self.bpe_model_path:
            config["bpe_model"] = self.bpe_model_path

        return config


class ASRConfig:
    """Container for all ASR models and shared recognition settings."""

    def __init__(self, models_dir: str = None):
        """Initialize ASR configuration."""
        if models_dir is None:
            # Try multiple locations for models directory
            candidates = [
                # Environment variable
                os.environ.get("SHERPA_MODELS_DIR"),
                # Docker: /app/models (mounted volume)
                "/app/models",
                # Legacy: /home/luigi/sherpa/models (symlink in Docker)
                "/home/luigi/sherpa/models",
                # Host: ./models relative to this file
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
                # Host: ./models relative to project root (this file is in project root)
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
            ]
            for candidate in candidates:
                if candidate and os.path.isdir(candidate):
                    models_dir = candidate
                    break
            else:
                # Fallback to ./models (will fail validation if not found)
                models_dir = "./models"
        self.models_dir = models_dir
        self.models: Dict[str, ModelConfig] = {}

        # Define all 3 models
        self._define_models()

    def _define_models(self) -> None:
        """Define the 3 ASR models."""

        # Model 1: Small Bilingual (2023-02-16)
        # More compact, faster inference, Chinese + English
        model1_dir = os.path.join(
            self.models_dir,
            "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile"
        )
        self.models["small-bilingual"] = ModelConfig(
            name="Small Bilingual (zh-en 2023-02-16)",
            encoder_path=os.path.join(model1_dir, "encoder-epoch-99-avg-1.int8.onnx"),
            decoder_path=os.path.join(model1_dir, "decoder-epoch-99-avg-1.onnx"),
            joiner_path=os.path.join(model1_dir, "joiner-epoch-99-avg-1.int8.onnx"),
            tokens_path=os.path.join(model1_dir, "tokens.txt"),
            bpe_model_path=os.path.join(model1_dir, "bpe.model"),
            bpe_vocab_path=os.path.join(model1_dir, "bpe.vocab"),
            hotwords=HotwordConfig(
                # Use centralized ZH_TW_HOTWORDS list
                hotwords=ZH_TW_HOTWORDS,
                boost_score=HOTWORD_BOOST_SCORE
            ),
            modeling_unit="cjkchar+bpe",  # Required for bilingual zh-en with hotwords
            num_threads=4,
        )

        # Model 2: Medium Bilingual (2023-02-20)
        # Larger, better accuracy, Chinese + English
        model2_dir = os.path.join(
            self.models_dir,
            "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-mobile"
        )
        self.models["medium-bilingual"] = ModelConfig(
            name="Medium Bilingual (zh-en 2023-02-20)",
            encoder_path=os.path.join(model2_dir, "encoder-epoch-99-avg-1.int8.onnx"),
            decoder_path=os.path.join(model2_dir, "decoder-epoch-99-avg-1.onnx"),
            joiner_path=os.path.join(model2_dir, "joiner-epoch-99-avg-1.int8.onnx"),
            tokens_path=os.path.join(model2_dir, "tokens.txt"),
            bpe_model_path=os.path.join(model2_dir, "bpe.model"),  # Now we have the BPE model
            bpe_vocab_path=os.path.join(model2_dir, "bpe.vocab"),  # Now we have the bpe.vocab
            hotwords=HotwordConfig(
                # Use centralized ZH_TW_HOTWORDS list
                hotwords=ZH_TW_HOTWORDS,
                boost_score=HOTWORD_BOOST_SCORE
            ),
            modeling_unit="cjkchar+bpe",  # Required for bilingual zh-en with hotwords
            num_threads=4,
        )

        # Model 3: Multilingual (2025-02-10)
        # State-of-art, multilingual (AR, EN, ID, JA, RU, TH, VI, ZH)
        model3_dir = os.path.join(
            self.models_dir,
            "sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10"
        )
        self.models["multilingual"] = ModelConfig(
            name="Multilingual (ar_en_id_ja_ru_th_vi_zh 2025-02-10)",
            encoder_path=os.path.join(model3_dir, "encoder-epoch-75-avg-11-chunk-16-left-128.int8.onnx"),
            decoder_path=os.path.join(model3_dir, "decoder-epoch-75-avg-11-chunk-16-left-128.onnx"),
            joiner_path=os.path.join(model3_dir, "joiner-epoch-75-avg-11-chunk-16-left-128.int8.onnx"),
            tokens_path=os.path.join(model3_dir, "tokens.txt"),
            bpe_model_path=os.path.join(model3_dir, "bpe.model"),
            bpe_vocab_path=os.path.join(model3_dir, "bpe.vocab"),
            hotwords=HotwordConfig(
                # Use centralized ZH_TW_HOTWORDS list
                hotwords=ZH_TW_HOTWORDS,
                boost_score=HOTWORD_BOOST_SCORE
            ),
            modeling_unit="cjkchar+bpe",  # Required for multilingual with hotwords
            num_threads=4,
        )

    def validate_all(self) -> tuple[bool, Dict[str, str]]:
        """Validate all models."""
        results = {}
        all_valid = True

        for model_id, model in self.models.items():
            is_valid, message = model.validate()
            results[model_id] = message
            if not is_valid:
                all_valid = False

        return all_valid, results

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model by ID."""
        return self.models.get(model_id)

    def list_models(self) -> Dict[str, str]:
        """Get all model IDs and names."""
        return {model_id: model.name for model_id, model in self.models.items()}


# =========================================================================
# RECOGNITION CONFIGURATION
# =========================================================================

class RecognitionConfig:
    """Settings for speech recognition."""

    # Audio processing
    SAMPLE_RATE = 16000           # Sample rate for ASR (ONNX models require 16kHz)
    CHUNK_DURATION_MS = 100       # Audio chunk duration (100ms = 1600 samples at 16kHz)

    # Recognition behavior
    DECODING_METHOD = "greedy_search"  # "greedy_search" or "modified_beam_search"
    NUM_ACTIVE_PATHS = 4          # For beam search
    ENABLE_HOTWORDS = True        # Enable hotword biasing
    LANG_IDS = [0, 1]            # Language IDs: [0]=Chinese, [1]=English (for bilingual models)

    # Output
    RULE_FSTS = None              # Optional FST rules
    RULE_SYMBOLS = None           # Optional symbol table


# =========================================================================
# GLOBAL INSTANCES
# =========================================================================

# Create global ASR config instance
asr_config = ASRConfig()

# Validate on import
all_valid, validation_results = asr_config.validate_all()
if not all_valid:
    print("⚠ ASR Configuration Validation Warnings:")
    for model_id, message in validation_results.items():
        print(f"  {model_id}: {message}")
else:
    print("✓ All ASR models validated successfully")


# =========================================================================
# TESTING / DEBUG
# =========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ASR CONFIGURATION VALIDATION")
    print("=" * 70 + "\n")

    print("Available Models:")
    for model_id, name in asr_config.list_models().items():
        model = asr_config.get_model(model_id)
        is_valid, msg = model.validate()
        status = "✓" if is_valid else "✗"
        print(f"\n{status} {model_id}")
        print(f"   Name: {name}")
        print(f"   Sample Rate: {model.sample_rate} Hz")
        print(f"   Encoder: {os.path.basename(model.encoder_path)}")
        print(f"   BPE Vocab: {'✓' if model.bpe_vocab_path else '✗'}")
        print(f"   Hotwords: {model.hotwords.hotwords if model.hotwords else '✗'}")
        if not is_valid:
            print(f"   Error: {msg}")

    print("\n" + "=" * 70)
    print("Recognition Settings:")
    print("=" * 70)
    print(f"Sample Rate: {RecognitionConfig.SAMPLE_RATE} Hz")
    print(f"Chunk Duration: {RecognitionConfig.CHUNK_DURATION_MS} ms")
    print(f"Decoding Method: {RecognitionConfig.DECODING_METHOD}")
    print(f"Hotwords Enabled: {RecognitionConfig.ENABLE_HOTWORDS}")
    print()
