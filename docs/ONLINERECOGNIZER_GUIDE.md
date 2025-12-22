# sherpa_onnx.OnlineRecognizer API Guide

Complete reference for using the `sherpa_onnx.OnlineRecognizer` API for streaming speech recognition based on examples from the sherpa repository.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage Pattern](#basic-usage-pattern)
3. [Creating Recognizers](#creating-recognizers)
4. [Stream Operations](#stream-operations)
5. [Processing Audio](#processing-audio)
6. [Getting Results](#getting-results)
7. [Hotwords Support](#hotwords-support)
8. [Advanced Usage](#advanced-usage)
9. [Complete Examples](#complete-examples)

---

## Quick Start

The most basic workflow for streaming speech recognition:

```python
import sherpa_onnx

# 1. Create recognizer
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="path/to/tokens.txt",
    encoder="path/to/encoder.onnx",
    decoder="path/to/decoder.onnx",
    joiner="path/to/joiner.onnx",
    num_threads=1,
    sample_rate=16000,
)

# 2. Create stream
stream = recognizer.create_stream()

# 3. Feed audio samples
stream.accept_waveform(sample_rate=16000, waveform=audio_samples)

# 4. Process when ready
while recognizer.is_ready(stream):
    recognizer.decode_stream(stream)

# 5. Get result
result = recognizer.get_result(stream)
print(result.text)

# 6. Check for endpoint
if recognizer.is_endpoint(stream):
    recognizer.reset(stream)  # Reset for next utterance
```

---

## Basic Usage Pattern

The streaming recognition workflow follows this pattern:

```
Create Recognizer
      ↓
Create Stream
      ↓
[Loop] Feed Audio Chunks
      ↓
      Check if ready to decode
      ↓
      Decode (process) the stream
      ↓
      Get partial result
      ↓
      Check for endpoint (speech end)
      ↓
      If endpoint → Reset for next utterance
```

---

## Creating Recognizers

### Method 1: from_transducer (Recommended - Simple)

Use this for most zipformer models:

```python
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="path/to/tokens.txt",
    encoder="path/to/encoder.onnx",
    decoder="path/to/decoder.onnx",
    joiner="path/to/joiner.onnx",
    num_threads=1,
    sample_rate=16000,           # IMPORTANT: Must be 16000
    feature_dim=80,              # Typically 80 for ASR
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,  # End utterance after 2.4s silence
    rule2_min_trailing_silence=1.2,
    rule3_min_utterance_length=300,   # Minimum utterance length (ms)
    decoding_method="greedy_search",  # Or "modified_beam_search"
    provider="cpu",              # Or "cuda", "coreml"
)
```

### Method 2: OnlineRecognizerConfig (Advanced - Full Control)

Use this for more complex configurations:

```python
config = sherpa_onnx.OnlineRecognizerConfig(
    model_type="zipformer",  # Model architecture
    zipformer=sherpa_onnx.OnlineZipformerModelConfig(
        encoder="path/to/encoder.onnx",
        decoder="path/to/decoder.onnx",
        joiner="path/to/joiner.onnx",
    ),
    tokens="path/to/tokens.txt",
    num_threads=1,
    provider="cpu",
    decoding_method="greedy",
    # Optional: for hotword support
    bpe_vocab="path/to/bpe.vocab",
)

recognizer = sherpa_onnx.OnlineRecognizer(config)
```

---

## Stream Operations

### Creating a Stream

```python
# Basic stream creation
stream = recognizer.create_stream()

# Stream with hotwords (see Hotwords section below)
stream = recognizer.create_stream(
    hotwords="HOTWORD1\nHOTWORD2",  # newline-separated
    hotwords_score=1.5
)
```

### Feeding Audio

Feed audio in chunks (typically 100ms at a time):

```python
# Audio must be:
# - Sample rate matching recognizer (16000 Hz)
# - Mono (1 channel)
# - Float32 or Int16 format
# - Typically 100ms chunks

samples = audio_array  # shape: (16000 * 0.1,) = (1600,)
stream.accept_waveform(sample_rate=16000, waveform=samples)
```

### Checking Stream Status

```python
# Check if stream has enough data to process
if recognizer.is_ready(stream):
    recognizer.decode_stream(stream)  # or decode()

# Check if speech endpoint detected (speech ended)
if recognizer.is_endpoint(stream):
    # User has stopped speaking
    result = recognizer.get_result(stream)
    print(f"Final: {result.text}")
    recognizer.reset(stream)  # Reset for next utterance
```

### Resetting Stream

```python
# Reset stream for a new utterance
recognizer.reset(stream)
```

---

## Processing Audio

### The Processing Loop

```python
import sounddevice as sd
import numpy as np

# Recognizer setup
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(...)
stream = recognizer.create_stream()

# Audio stream setup
sample_rate = 48000  # Microphone sample rate
chunk_duration = 0.1  # 100ms
samples_per_chunk = int(sample_rate * chunk_duration)

with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as audio_stream:
    while True:
        # 1. Read audio chunk from microphone
        samples, _ = audio_stream.read(samples_per_chunk)
        samples = samples.reshape(-1)

        # 2. Feed to stream (note: sherpa uses 16000 Hz internally)
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)

        # 3. Process when ready
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        # 4. Get partial result
        result = recognizer.get_result(stream)
        print(f"Partial: {result.text}")

        # 5. Check for endpoint
        if recognizer.is_endpoint(stream):
            result = recognizer.get_result(stream)
            print(f"Final: {result.text}")
            recognizer.reset(stream)
```

---

## Getting Results

### The Result Object

```python
result = recognizer.get_result(stream)

# Access result properties
print(result.text)              # Transcription text
print(result.is_final)          # True if endpoint detected
print(result.tokens)            # Token IDs (if available)
print(result.timestamps)        # Timestamps (if available)
print(result.lang)              # Language ID (for multilingual models)
```

### Interpretation

```python
# Partial result (still speaking)
if not result.is_final:
    print(f"Partial: {result.text}")

# Final result (endpoint detected)
if result.is_final:
    print(f"Final: {result.text}")
```

---

## Hotwords Support

### Overview

Hotwords allow you to boost recognition of specific terms (person names, product names, etc.). This is useful for improving accuracy on domain-specific vocabulary.

### Prerequisites

1. **Model must have BPE vocab**: Check for `bpe.vocab` file in the model directory
2. **Decoding method**: Must use `"modified_beam_search"` (not `"greedy_search"`)
3. **Hotword format**: ALL UPPERCASE, space-separated for multi-token hotwords

### Configuration

When creating the recognizer, enable hotword support:

```python
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="path/to/tokens.txt",
    encoder="path/to/encoder.onnx",
    decoder="path/to/decoder.onnx",
    joiner="path/to/joiner.onnx",
    num_threads=1,
    sample_rate=16000,
    decoding_method="modified_beam_search",  # CRITICAL: Required for hotwords
    bpe_vocab="path/to/bpe.vocab",           # BPE vocabulary for hotword processing
    modeling_unit="bpe",                      # "bpe", "cjkchar", or "cjkchar+bpe"
    # ... other parameters
)
```

### Using Hotwords

#### Method 1: Per-Stream Hotwords

```python
hotwords_str = """KEN LI
JOHN SMITH
MARK MA"""

stream = recognizer.create_stream(
    hotwords=hotwords_str,
    hotwords_score=2.0  # Boost score (1.5-5.0 typical range)
)
```

#### Method 2: Hotwords File

```python
# Create hotwords.txt file with one hotword per line (all uppercase)
# KEN LI
# JOHN SMITH
# MARK MA

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    # ... other parameters
    hotwords_file="path/to/hotwords.txt",
    hotwords_score=2.0
)

stream = recognizer.create_stream()
```

### Hotword Format Rules

**CRITICAL**: These rules must be followed exactly:

1. **ALL UPPERCASE** - "KEN LI" works, "Ken Li" doesn't
2. **Space-separated for multi-token** - "KEN LI" (with space), not "KENLI"
3. **No comments in file** - Only hotwords, no descriptions
4. **Valid tokens** - All tokens must be in model's vocabulary
5. **Single-token also works** - "KEN" alone is fine
6. **Chinese characters work** - "特雷危" (no uppercase needed)

Example hotwords file:

```
KEN LI
JOHN SMITH
ALICE JOHNSON
MARK MA
特雷危
林志玲
```

### Modeling Unit Selection

Different models have different tokenization:

- **"bpe"**: Byte-Pair Encoding (default for most models)
- **"cjkchar"**: Character-level for CJK (Chinese, Japanese, Korean)
- **"cjkchar+bpe"**: Hybrid for multilingual models mixing English and CJK

Check your model documentation to determine which to use.

### Expected Improvements

- **Targeted improvement**: Up to 25% accuracy improvement when hotword applies
- **General improvement**: 2-8% overall accuracy improvement
- **Performance**: Minimal overhead (same inference time)

---

## Advanced Usage

### Multiple Streams (Parallel Recognition)

For recognizing from multiple sources simultaneously:

```python
class ModelInstance:
    def __init__(self, model_id, recognizer_config):
        self.model_id = model_id
        self.recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
        self.stream = self.recognizer.create_stream()
        self._lock = threading.RLock()

    def feed_audio(self, audio):
        with self._lock:
            self.stream.accept_waveform(sample_rate=16000, waveform=audio)

    def process(self):
        with self._lock:
            if self.stream.is_ready(self.recognizer):
                self.recognizer.decode(self.stream)

    def get_result(self):
        with self._lock:
            return self.stream.get_result()
```

### Dynamic Hotword Updates

```python
# Create stream with initial hotwords
stream = recognizer.create_stream(
    hotwords="INITIAL_HOTWORD",
    hotwords_score=2.0
)

# Later, create a new stream with different hotwords
stream = recognizer.create_stream(
    hotwords="UPDATED_HOTWORD",
    hotwords_score=2.0
)
```

### Custom Audio Sources

```python
import soundfile as sf

# From file
data, sample_rate = sf.read("audio.wav")
stream.accept_waveform(sample_rate, data)

# From network stream
response = requests.get("https://example.com/audio.wav", stream=True)
for chunk in response.iter_content(chunk_size=3200):  # 100ms at 16kHz
    stream.accept_waveform(16000, chunk)
```

---

## Complete Examples

### Example 1: Microphone Speech Recognition (Basic)

**File**: `/home/luigi/sherpa/speech_recognition_mic.py`

```python
#!/usr/bin/env python3

import sounddevice as sd
import sherpa_onnx

def create_recognizer(tokens, encoder, decoder, joiner, num_threads=1):
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        decoding_method="greedy_search",
    )
    return recognizer

recognizer = create_recognizer(
    tokens="models/tokens.txt",
    encoder="models/encoder.onnx",
    decoder="models/decoder.onnx",
    joiner="models/joiner.onnx"
)

stream = recognizer.create_stream()
display = sherpa_onnx.Display()

sample_rate = 48000
samples_per_read = int(0.1 * sample_rate)  # 100ms chunks

with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
    while True:
        samples, _ = s.read(samples_per_read)
        samples = samples.reshape(-1)

        stream.accept_waveform(sample_rate, samples)

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        is_endpoint = recognizer.is_endpoint(stream)
        result = recognizer.get_result(stream)

        display.update_text(result)
        display.display()

        if is_endpoint:
            if result:
                display.finalize_current_sentence()
                display.display()
            recognizer.reset(stream)
```

### Example 2: Microphone with Hotwords

**File**: `/home/luigi/sherpa/speech_recognition_mic_hotwords.py`

```python
#!/usr/bin/env python3

import sounddevice as sd
import sherpa_onnx

def create_recognizer(tokens, encoder, decoder, joiner,
                     bpe_vocab=None, modeling_unit="bpe", num_threads=4):
    kwargs = {
        "tokens": tokens,
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner,
        "num_threads": num_threads,
        "sample_rate": 16000,
        "feature_dim": 80,
        "enable_endpoint_detection": True,
        "decoding_method": "modified_beam_search",  # Required for hotwords
    }

    if bpe_vocab:
        kwargs["bpe_vocab"] = bpe_vocab
        kwargs["modeling_unit"] = modeling_unit

    return sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)

recognizer = create_recognizer(
    tokens="models/tokens.txt",
    encoder="models/encoder.onnx",
    decoder="models/decoder.onnx",
    joiner="models/joiner.onnx",
    bpe_vocab="models/bpe.vocab",
    modeling_unit="bpe"
)

# Create stream with hotwords
hotwords = "KEN LI\nJOHN SMITH\nMARK MA"
stream = recognizer.create_stream(hotwords=hotwords, hotwords_score=2.0)

# ... rest of processing loop same as Example 1
```

### Example 3: Parallel ASR Engines (Advanced)

**File**: `/home/luigi/sherpa/asr_engine.py` (Simplified excerpt)

```python
import threading
import sherpa_onnx

class ASREnginePool:
    def __init__(self):
        self.models = {}
        self._pool_lock = threading.Lock()

    def load_models(self):
        """Load multiple recognizers in parallel."""
        model_configs = {
            "small": {...},
            "medium": {...},
            "large": {...}
        }

        for model_id, config in model_configs.items():
            recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
                model_type="zipformer",
                zipformer=sherpa_onnx.OnlineZipformerModelConfig(
                    encoder=config["encoder"],
                    decoder=config["decoder"],
                    joiner=config["joiner"],
                ),
                tokens=config["tokens"],
                num_threads=config["num_threads"],
                provider=config["provider"],
                bpe_vocab=config.get("bpe_vocab"),
            )

            recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
            self.models[model_id] = {
                "recognizer": recognizer,
                "stream": recognizer.create_stream()
            }

    def feed_audio_chunk(self, audio):
        """Feed same audio to all models."""
        for model_id, model in self.models.items():
            model["stream"].accept_waveform(16000, audio)

    def process(self):
        """Process all streams."""
        for model_id, model in self.models.items():
            recognizer = model["recognizer"]
            stream = model["stream"]

            if stream.is_ready(recognizer):
                recognizer.decode(stream)

    def get_results(self):
        """Get results from all models."""
        results = {}
        for model_id, model in self.models.items():
            result = model["stream"].get_result()
            results[model_id] = result.text
        return results
```

---

## Key Parameters Reference

### Recognizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokens` | str | - | Path to tokens.txt file (REQUIRED) |
| `encoder` | str | - | Path to encoder ONNX model (REQUIRED) |
| `decoder` | str | - | Path to decoder ONNX model (REQUIRED) |
| `joiner` | str | - | Path to joiner ONNX model (REQUIRED) |
| `num_threads` | int | 1 | Number of threads for inference |
| `sample_rate` | int | 16000 | Audio sample rate (must be 16000) |
| `feature_dim` | int | 80 | Feature dimension (typically 80) |
| `enable_endpoint_detection` | bool | True | Enable speech endpoint detection |
| `decoding_method` | str | "greedy_search" | "greedy_search" or "modified_beam_search" |
| `rule1_min_trailing_silence` | float | 2.4 | Silence to end utterance (seconds) |
| `rule2_min_trailing_silence` | float | 1.2 | Alternative silence threshold |
| `rule3_min_utterance_length` | int | 300 | Minimum utterance length (ms) |
| `provider` | str | "cpu" | ONNX provider: "cpu", "cuda", "coreml" |
| `bpe_vocab` | str | None | Path to bpe.vocab (for hotwords) |
| `modeling_unit` | str | "bpe" | Tokenization: "bpe", "cjkchar", "cjkchar+bpe" |

### Stream Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `create_stream()` | hotwords (str), hotwords_score (float) | stream | Create new recognition stream |
| `accept_waveform()` | sample_rate (int), waveform (array) | None | Feed audio chunk |
| `is_ready()` | recognizer | bool | Check if ready to decode |
| `decode_stream()` | stream | None | Decode/process accumulated audio |
| `decode()` | stream | None | Alternative decode method |
| `is_endpoint()` | stream | bool | Check if speech endpoint detected |
| `get_result()` | stream | result | Get current transcription result |
| `reset()` | stream | None | Reset stream for new utterance |

---

## Common Issues and Solutions

### Issue: Model fails to load
**Cause**: Missing ONNX files
**Solution**: Verify all 4 files exist: encoder, decoder, joiner, tokens.txt

### Issue: Hotwords not working
**Cause**: Not using "modified_beam_search" decoding method
**Solution**: Change `decoding_method="modified_beam_search"`

### Issue: Hotwords not recognized
**Cause**: Wrong case (not ALL UPPERCASE) or extra spaces
**Solution**: Check hotword format strictly: "KEN LI" works, "Ken Li" doesn't

### Issue: No results returned
**Cause**: Not calling `recognizer.decode_stream(stream)` when ready
**Solution**: Always check `is_ready()` and call decode in a loop

### Issue: Audio cuts off
**Cause**: Not calling `reset()` after endpoint detection
**Solution**: Always reset stream when endpoint is detected

---

## References

- **Official docs**: https://k2-fsa.github.io/sherpa/onnx/index.html
- **ASR models**: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
- **Hotwords guide**: https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html
- **Example scripts**: https://github.com/k2-fsa/sherpa-onnx/tree/master/python-api-examples

---

## File Locations in This Repository

- **Basic example**: `/home/luigi/sherpa/speech_recognition_mic.py`
- **Hotwords example**: `/home/luigi/sherpa/speech_recognition_mic_hotwords.py`
- **Parallel engines**: `/home/luigi/sherpa/asr_engine.py`
- **Configuration**: `/home/luigi/sherpa/asr_config.py`
- **Hotwords documentation**: `/home/luigi/sherpa/docs/HOTWORDS_WORKING.md`

