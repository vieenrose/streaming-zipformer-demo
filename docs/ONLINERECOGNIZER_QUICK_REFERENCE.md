# OnlineRecognizer Quick Reference

Fast lookup guide for common sherpa_onnx.OnlineRecognizer usage patterns.

---

## Minimal Working Example (30 seconds)

```python
import sherpa_onnx
import sounddevice as sd

# 1. Create recognizer
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="model/tokens.txt",
    encoder="model/encoder.onnx",
    decoder="model/decoder.onnx",
    joiner="model/joiner.onnx",
)

# 2. Create stream
stream = recognizer.create_stream()

# 3. Feed audio (100ms chunks at 16kHz)
samples = next(audio_source)  # 1600 float32 samples
stream.accept_waveform(16000, samples)

# 4. Decode when ready
if recognizer.is_ready(stream):
    recognizer.decode_stream(stream)

# 5. Get result
result = recognizer.get_result(stream)
print(result.text)
```

---

## Recognizer Creation Patterns

### Pattern 1: Simple (Most Common)

```python
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="model/tokens.txt",
    encoder="model/encoder.onnx",
    decoder="model/decoder.onnx",
    joiner="model/joiner.onnx",
)
```

### Pattern 2: With Configuration

```python
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="model/tokens.txt",
    encoder="model/encoder.onnx",
    decoder="model/decoder.onnx",
    joiner="model/joiner.onnx",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,
    rule2_min_trailing_silence=1.2,
    rule3_min_utterance_length=300,
    decoding_method="greedy_search",
)
```

### Pattern 3: With Hotwords

```python
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="model/tokens.txt",
    encoder="model/encoder.onnx",
    decoder="model/decoder.onnx",
    joiner="model/joiner.onnx",
    decoding_method="modified_beam_search",  # REQUIRED for hotwords
    bpe_vocab="model/bpe.vocab",            # REQUIRED for hotwords
    modeling_unit="bpe",
)
```

### Pattern 4: Advanced Config Object

```python
config = sherpa_onnx.OnlineRecognizerConfig(
    model_type="zipformer",
    zipformer=sherpa_onnx.OnlineZipformerModelConfig(
        encoder="model/encoder.onnx",
        decoder="model/decoder.onnx",
        joiner="model/joiner.onnx",
    ),
    tokens="model/tokens.txt",
    num_threads=1,
    provider="cpu",
    decoding_method="greedy",
)
recognizer = sherpa_onnx.OnlineRecognizer(config)
```

---

## Stream Operations

### Create Stream

```python
# Basic
stream = recognizer.create_stream()

# With hotwords
stream = recognizer.create_stream(
    hotwords="HOTWORD1\nHOTWORD2",
    hotwords_score=2.0
)
```

### Feed Audio

```python
# Numpy array (most common)
import numpy as np
samples = np.array([...], dtype=np.float32)  # or int16
stream.accept_waveform(sample_rate=16000, waveform=samples)

# From sounddevice
samples, _ = audio_stream.read(samples_per_chunk)
stream.accept_waveform(sample_rate, samples.reshape(-1))
```

### Process Loop

```python
# Classic pattern
while recognizer.is_ready(stream):
    recognizer.decode_stream(stream)

# Alternative (same thing)
while stream.is_ready(recognizer):
    recognizer.decode(stream)
```

### Get Result

```python
# Get current result (can be partial or final)
result = recognizer.get_result(stream)
print(result.text)

# Check if final
if result.is_final:
    print("Final:", result.text)
else:
    print("Partial:", result.text)
```

### Endpoint Detection

```python
if recognizer.is_endpoint(stream):
    # Speech has ended, get final result
    result = recognizer.get_result(stream)
    print("Utterance:", result.text)

    # Reset for next utterance
    recognizer.reset(stream)
```

### Reset Stream

```python
recognizer.reset(stream)
```

---

## Audio Input Patterns

### From Microphone (sounddevice)

```python
import sounddevice as sd

with sd.InputStream(channels=1, dtype="float32", samplerate=48000) as s:
    while True:
        samples, _ = s.read(samples_per_chunk)
        stream.accept_waveform(48000, samples.reshape(-1))
```

### From File (soundfile)

```python
import soundfile as sf

data, sr = sf.read("audio.wav")
stream.accept_waveform(sr, data)
```

### From Bytes (e.g., network)

```python
import numpy as np

# From raw bytes
audio_bytes = b'\x00\x01\x02\x03...'
audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
stream.accept_waveform(16000, audio_array)
```

### Chunk-by-Chunk (Memory Efficient)

```python
chunk_size = 1600  # 100ms at 16kHz

for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    stream.accept_waveform(16000, chunk)

    # Decode immediately after each chunk
    if recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
```

---

## Hotwords Quick Reference

### Enable Hotwords (3 Steps)

1. **Create recognizer with hotword support**:
```python
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="model/tokens.txt",
    encoder="model/encoder.onnx",
    decoder="model/decoder.onnx",
    joiner="model/joiner.onnx",
    decoding_method="modified_beam_search",  # ← CRITICAL
    bpe_vocab="model/bpe.vocab",             # ← REQUIRED
    modeling_unit="bpe",
)
```

2. **Create stream with hotwords**:
```python
hotwords = "KEN LI\nJOHN SMITH\nMARK MA"
stream = recognizer.create_stream(
    hotwords=hotwords,
    hotwords_score=2.0
)
```

3. **Use normally**:
```python
stream.accept_waveform(16000, samples)
if recognizer.is_ready(stream):
    recognizer.decode_stream(stream)
result = recognizer.get_result(stream)
```

### Hotword Format Rules (STRICT!)

```
✓ CORRECT:
KEN LI          # All uppercase, space-separated
JOHN SMITH      # Multi-word hotwords OK
特雷危           # Chinese characters OK
林志玲 MARK      # Mixed English + Chinese OK

✗ WRONG:
Ken Li          # Wrong case!
ken li          # Wrong case!
KENLI           # No space (treated as single token)
"KEN LI"        # No quotes!
KEN LI #name    # No comments!
```

### Update Hotwords Dynamically

```python
# Create new stream with different hotwords
old_stream = stream
stream = recognizer.create_stream(
    hotwords="NEW_HOTWORD1\nNEW_HOTWORD2",
    hotwords_score=2.0
)
# Continue using stream...
```

---

## Configuration Reference

### Decoding Methods

```python
# Decoding method affects behavior:
decoding_method="greedy_search"         # Faster, simpler
# → Use for real-time, low-latency
# → Hotwords NOT supported

decoding_method="modified_beam_search"  # Slower, more accurate
# → Use for higher accuracy
# → Hotwords REQUIRED
```

### Endpoint Detection Rules

```python
# Three rules for detecting speech endpoint:
rule1_min_trailing_silence=2.4   # End if 2.4s silence
rule2_min_trailing_silence=1.2   # OR if 1.2s silence + threshold
rule3_min_utterance_length=300   # AND utterance >= 300ms
```

### Provider Options

```python
provider="cpu"      # CPU inference
provider="cuda"     # NVIDIA GPU
provider="coreml"   # Apple CoreML
```

### Modeling Units

```python
modeling_unit="bpe"           # Byte-pair encoding (default)
modeling_unit="cjkchar"       # Character-level for CJK
modeling_unit="cjkchar+bpe"   # Hybrid for multilingual
```

---

## Common Workflows

### Real-Time Transcription (Streaming)

```python
import sounddevice as sd
import sherpa_onnx

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="model/tokens.txt",
    encoder="model/encoder.onnx",
    decoder="model/decoder.onnx",
    joiner="model/joiner.onnx",
)

stream = recognizer.create_stream()

with sd.InputStream(channels=1, dtype="float32", samplerate=48000) as s:
    while True:
        samples, _ = s.read(1600)  # 100ms at 48kHz
        stream.accept_waveform(48000, samples.reshape(-1))

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        if result.text:
            print(f"\rPartial: {result.text}", end="")

        if recognizer.is_endpoint(stream):
            print()  # newline
            recognizer.reset(stream)
```

### Batch File Processing

```python
import soundfile as sf
import sherpa_onnx

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="model/tokens.txt",
    encoder="model/encoder.onnx",
    decoder="model/decoder.onnx",
    joiner="model/joiner.onnx",
)

for filename in ["audio1.wav", "audio2.wav"]:
    audio, sr = sf.read(filename)
    stream = recognizer.create_stream()

    # Feed all audio at once
    stream.accept_waveform(sr, audio)

    # Process
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    # Get final result
    result = recognizer.get_result(stream)
    print(f"{filename}: {result.text}")
```

### Parallel Model Comparison

```python
models = {}
recognizers = {}

# Load multiple models
for model_id in ["small", "medium", "large"]:
    config = load_model_config(model_id)
    recognizer = sherpa_onnx.OnlineRecognizer(config)
    recognizers[model_id] = recognizer
    models[model_id] = recognizer.create_stream()

# Feed same audio to all
for chunk in audio_chunks:
    for model_id, stream in models.items():
        stream.accept_waveform(16000, chunk)

# Process all
for model_id, stream in models.items():
    recognizer = recognizers[model_id]
    if stream.is_ready(recognizer):
        recognizer.decode(stream)

# Compare results
for model_id, stream in models.items():
    recognizer = recognizers[model_id]
    result = recognizer.get_result(stream)
    print(f"{model_id}: {result.text}")
```

### Hotword-Enhanced Streaming

```python
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="model/tokens.txt",
    encoder="model/encoder.onnx",
    decoder="model/decoder.onnx",
    joiner="model/joiner.onnx",
    decoding_method="modified_beam_search",
    bpe_vocab="model/bpe.vocab",
)

hotwords = "KEN LI\nJOHN SMITH\nMARK MA"
stream = recognizer.create_stream(
    hotwords=hotwords,
    hotwords_score=2.0
)

# Rest same as real-time transcription...
```

---

## Debugging Tips

### Check If Recognizer Loaded

```python
try:
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(...)
    print("✓ Recognizer loaded successfully")
except Exception as e:
    print(f"✗ Failed to load recognizer: {e}")
```

### Verify Files Exist

```python
from pathlib import Path

required_files = {
    "tokens": "model/tokens.txt",
    "encoder": "model/encoder.onnx",
    "decoder": "model/decoder.onnx",
    "joiner": "model/joiner.onnx",
}

for name, path in required_files.items():
    if Path(path).exists():
        print(f"✓ {name}: {path}")
    else:
        print(f"✗ {name} missing: {path}")
```

### Check Stream State

```python
print(f"Stream ready: {recognizer.is_ready(stream)}")
print(f"Endpoint detected: {recognizer.is_endpoint(stream)}")

result = recognizer.get_result(stream)
print(f"Text: {result.text}")
print(f"Is final: {result.is_final}")
```

### Audio Format Verification

```python
import numpy as np

samples = np.array([...])
print(f"Dtype: {samples.dtype}")  # Should be float32 or int16
print(f"Shape: {samples.shape}")  # Should be (n_samples,)
print(f"Min: {samples.min():.4f}, Max: {samples.max():.4f}")

# Check sample rate
expected_sr = 16000
actual_sr = 48000
# If different, note that accept_waveform takes the actual sample rate
```

---

## Result Object Properties

```python
result = recognizer.get_result(stream)

# Key properties:
result.text              # Transcription text (str)
result.is_final          # True if endpoint detected (bool)
result.tokens            # Token IDs (list, if available)
result.timestamps        # Timestamps (list, if available)
result.lang              # Language ID (int, for multilingual models)

# Example:
if result.is_final:
    print(f"Final transcription: {result.text}")
else:
    print(f"Partial transcription: {result.text}")
```

---

## Performance Tips

1. **Set appropriate `num_threads`**: Match your CPU core count
2. **Use GPU if available**: Set `provider="cuda"` for NVIDIA
3. **Greedy search is faster**: Use for real-time, modified_beam_search for accuracy
4. **Batch audio chunks**: Process multiple chunks together if possible
5. **Reset streams properly**: Don't keep stale streams in memory

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: encoder.onnx` | Missing model file | Verify all model files exist |
| `Hotwords not working` | Not using modified_beam_search | Change `decoding_method` parameter |
| `No results returned` | Not decoding | Call `recognizer.decode_stream(stream)` in loop |
| `Audio cuts off` | Not resetting | Call `recognizer.reset(stream)` after endpoint |
| `"Ken Li" not recognized as hotword` | Wrong case | Use "KEN LI" (all uppercase) |
| `ONNX Runtime error` | Provider not available | Try `provider="cpu"` |

---

## See Also

- Full guide: `ONLINERECOGNIZER_GUIDE.md`
- Hotwords details: `docs/HOTWORDS_WORKING.md`
- Configuration: `asr_config.py`
- Examples: `speech_recognition_mic.py`, `speech_recognition_mic_hotwords.py`
- Advanced: `asr_engine.py`

