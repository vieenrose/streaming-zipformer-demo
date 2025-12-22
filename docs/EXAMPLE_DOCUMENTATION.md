# Example Documentation - OnlineRecognizer Usage in Sherpa

Detailed analysis of the actual example files in this repository, showing real-world usage patterns.

---

## File 1: speech_recognition_mic.py

**Location**: `/home/luigi/sherpa/speech_recognition_mic.py`
**Purpose**: Basic streaming speech recognition from microphone input
**Complexity**: Beginner
**Model**: Transducer (Zipformer)

### Overview

This script demonstrates the simplest, most straightforward usage of OnlineRecognizer:
1. Create recognizer from transducer model
2. Create stream
3. Feed audio from microphone in chunks
4. Decode when ready
5. Display results
6. Reset on endpoint detection

### Key Code Sections

#### Recognizer Creation

```python
def create_recognizer(args):
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        sample_rate=16000,              # Internal ASR sample rate
        feature_dim=80,                 # Standard for speech
        enable_endpoint_detection=True, # Auto-detect speech end
        rule1_min_trailing_silence=2.4, # 2.4s silence = end
        rule2_min_trailing_silence=1.2, # Alternative rule
        rule3_min_utterance_length=300, # Min 300ms utterance
        decoding_method=args.decoding_method,  # "greedy_search" or "modified_beam_search"
        provider=args.provider,         # "cpu", "cuda", or "coreml"
    )
    return recognizer
```

**Key Points**:
- Uses `from_transducer()` factory method (recommended)
- All parameters passed as keyword arguments
- `sample_rate=16000` is the internal ASR rate (model requirement)
- Audio from microphone is 48000 Hz but gets resampled internally

#### Main Recognition Loop

```python
recognizer = create_recognizer(args)
print("Started! Please speak")

sample_rate = 48000                           # Microphone sample rate
samples_per_read = int(0.1 * sample_rate)    # 100ms chunks = 4800 samples

stream = recognizer.create_stream()
display = sherpa_onnx.Display()

with sd.InputStream(device=device_idx, channels=1, dtype="float32", samplerate=sample_rate) as s:
    while True:
        # 1. Read audio chunk from microphone
        samples, _ = s.read(samples_per_read)
        samples = samples.reshape(-1)

        # 2. Feed to recognizer (with original microphone sample rate)
        stream.accept_waveform(sample_rate, samples)

        # 3. Decode when stream has enough data
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        # 4. Check for endpoint (speech ended)
        is_endpoint = recognizer.is_endpoint(stream)

        # 5. Get result
        result = recognizer.get_result(stream)

        # 6. Display
        display.update_text(result)
        display.display()

        # 7. On endpoint, finalize and reset
        if is_endpoint:
            if result:
                display.finalize_current_sentence()
                display.display()
            recognizer.reset(stream)
```

**Key Points**:
- Microphone runs at 48000 Hz (typical)
- 100ms chunks = 4800 samples at 48000 Hz
- `accept_waveform()` takes actual sample rate (48000), not internal rate (16000)
- `is_ready()` checks if stream has enough data to decode
- `decode_stream()` processes the accumulated audio
- `is_endpoint()` detects when user stops speaking
- `reset()` prepares stream for next utterance

### Usage Example

```bash
python speech_recognition_mic.py \
  --tokens models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile/tokens.txt \
  --encoder models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile/encoder-epoch-99-avg-1.int8.onnx \
  --decoder models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile/decoder-epoch-99-avg-1.onnx \
  --joiner models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile/joiner-epoch-99-avg-1.int8.onnx \
  --num-threads 4 \
  --device 4
```

### When to Use

- ✓ Simple microphone transcription needed
- ✓ Real-time streaming recognition
- ✓ No hotwords required
- ✓ Learning the API
- ✗ Hotwords needed (use speech_recognition_mic_hotwords.py instead)
- ✗ Multilingual support required (see asr_engine.py)

---

## File 2: speech_recognition_mic_hotwords.py

**Location**: `/home/luigi/sherpa/speech_recognition_mic_hotwords.py`
**Purpose**: Streaming speech recognition with hotword boosting
**Complexity**: Intermediate
**Special Features**: Hotwords (contextual biasing)

### Overview

Extends the basic example with hotword support. Hotwords improve recognition accuracy for domain-specific terms (names, product names, etc.).

### Key Differences from Basic Example

#### Recognizer Creation with Hotwords

```python
def create_recognizer(args):
    kwargs = {
        "tokens": args.tokens,
        "encoder": args.encoder,
        "decoder": args.decoder,
        "joiner": args.joiner,
        "num_threads": args.num_threads,
        "sample_rate": 16000,
        "feature_dim": 80,
        "enable_endpoint_detection": True,
        "rule1_min_trailing_silence": 2.4,
        "rule2_min_trailing_silence": 1.2,
        "rule3_min_utterance_length": 300,
        "decoding_method": "modified_beam_search",  # ← CRITICAL: Required for hotwords!
    }

    if args.bpe_vocab:
        assert_file_exists(args.bpe_vocab)
        kwargs["bpe_vocab"] = args.bpe_vocab
        kwargs["modeling_unit"] = args.modeling_unit

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
    return recognizer
```

**Key Differences**:
- `decoding_method="modified_beam_search"` (NOT "greedy_search")
- Passes `bpe_vocab` parameter (BPE vocabulary)
- Passes `modeling_unit` parameter ("bpe", "cjkchar", or "cjkchar+bpe")
- Uses kwargs dict for flexibility

#### Stream Creation with Hotwords

```python
# Prepare hotwords string from command line
hotwords_str = ""
if args.hotwords:
    print(f"Hotwords enabled:")
    # Handle both comma-separated and newline-separated formats
    if "," in args.hotwords:
        hotwords_list = [hw.strip() for hw in args.hotwords.split(",")]
    else:
        hotwords_list = [hw.strip() for hw in args.hotwords.split("\n")]

    for hw in hotwords_list:
        if hw:
            print(f"  - {hw}")

    # Build hotwords string for create_stream()
    hotwords_str = "\n".join([hw for hw in hotwords_list if hw])
    print(f"  Score: {args.hotwords_score}")

# Create stream WITH hotwords
if hotwords_str:
    stream = recognizer.create_stream(
        hotwords=hotwords_str,
        hotwords_score=args.hotwords_score
    )
else:
    stream = recognizer.create_stream()
```

**Key Points**:
- Hotwords passed as newline-separated string to `create_stream()`
- Hotwords must be ALL UPPERCASE: "KEN LI" works, "Ken Li" doesn't
- Multi-token hotwords need spaces: "KEN LI" not "KENLI"
- `hotwords_score` controls boost magnitude (typical: 1.5-5.0)
- Same microphone loop as basic example

### Usage Examples

```bash
# Single hotword
python speech_recognition_mic_hotwords.py \
  --tokens model/tokens.txt \
  --encoder model/encoder.onnx \
  --decoder model/decoder.onnx \
  --joiner model/joiner.onnx \
  --hotwords "KEN" \
  --hotwords-score 2.0 \
  --bpe-vocab model/bpe.vocab

# Multiple hotwords (comma-separated)
python speech_recognition_mic_hotwords.py \
  --tokens model/tokens.txt \
  --encoder model/encoder.onnx \
  --decoder model/decoder.onnx \
  --joiner model/joiner.onnx \
  --hotwords "KEN,JOHN,MARK" \
  --hotwords-score 2.0 \
  --bpe-vocab model/bpe.vocab

# Multi-token hotwords
python speech_recognition_mic_hotwords.py \
  --tokens model/tokens.txt \
  --encoder model/encoder.onnx \
  --decoder model/decoder.onnx \
  --joiner model/joiner.onnx \
  --hotwords "KEN LI,JOHN SMITH" \
  --hotwords-score 2.0 \
  --bpe-vocab model/bpe.vocab
```

### Hotwords Format Reference

```
CORRECT FORMAT:
✓ KEN LI            # Uppercase with space
✓ JOHN SMITH        # Multi-word OK
✓ 特雷危             # Chinese characters OK (no uppercase needed)
✓ KEN,JOHN          # Comma-separated list
✓ "KEN\nJOHN"       # Newline-separated

INCORRECT FORMAT:
✗ Ken Li            # Wrong case
✗ ken li            # Lowercase
✗ KENLI             # No space (parsed as single token)
✗ "KEN LI"          # Quotes not needed
✗ KEN LI # name     # Comments not allowed
✗ KEN LI,JOHN SMITH # Mix spaces and commas inconsistently
```

### Requirements for Hotwords

1. **Model must have BPE vocab**: Check for `bpe.vocab` file
   ```
   model_dir/
   ├── encoder.onnx
   ├── decoder.onnx
   ├── joiner.onnx
   ├── tokens.txt
   ├── bpe.model      ← BPE model
   └── bpe.vocab      ← Required for hotwords
   ```

2. **Must use modified_beam_search**: Greedy search doesn't support hotwords
   ```python
   decoding_method="modified_beam_search"  # ← Required
   ```

3. **Valid tokens**: All tokens in hotword must be in model's vocabulary
   - Check: `grep "^TOKEN_NAME$" tokens.txt`
   - Chinese characters usually valid if model supports Chinese

### When to Use

- ✓ Need to boost recognition of specific terms
- ✓ Have person names, product names, etc.
- ✓ Model has bpe.vocab file
- ✓ Can sacrifice some speed for accuracy
- ✗ Need maximum speed (use basic example instead)
- ✗ Model doesn't have bpe.vocab (export first)

---

## File 3: asr_engine.py

**Location**: `/home/luigi/sherpa/asr_engine.py`
**Purpose**: Parallel ASR engine pool for simultaneous multi-model recognition
**Complexity**: Advanced
**Special Features**: Threading, multiple recognizers, state management

### Overview

Manages 3 ASR recognizers in parallel, feeding the same audio to all models simultaneously. Returns results from all models for comparison.

### Architecture

```
Audio Input
    ↓
ASREnginePool.feed_audio_chunk()
    ├→ ModelInstance 1 (small)
    │   ├→ recognizer: OnlineRecognizer
    │   ├→ stream: OnlineStream
    │   └→ state: TranscriptionState
    │
    ├→ ModelInstance 2 (medium)
    │   ├→ recognizer: OnlineRecognizer
    │   ├→ stream: OnlineStream
    │   └→ state: TranscriptionState
    │
    └→ ModelInstance 3 (multilingual)
        ├→ recognizer: OnlineRecognizer
        ├→ stream: OnlineStream
        └→ state: TranscriptionState
            ↓
ASREnginePool.get_results()
    → Dict[model_id, StreamResult]
```

### Key Classes

#### ModelInstance

```python
class ModelInstance:
    def __init__(self, model_id, model_config, recognizer):
        self.model_id = model_id
        self.model_config = model_config
        self.recognizer = recognizer
        self.stream = recognizer.create_stream()
        self.state = TranscriptionState()
        self._state_lock = threading.RLock()

    def feed_audio(self, audio: np.ndarray) -> None:
        """Feed audio chunk to model."""
        with self._state_lock:
            # Convert to int16 if needed
            if audio.dtype == np.float32:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio.astype(np.int16)

            # Feed to stream
            self.stream.accept_waveform(
                sample_rate=RecognitionConfig.SAMPLE_RATE,
                waveform=audio_int16,
            )
            self.state.chunk_count += 1

    def process(self) -> None:
        """Process accumulated audio and update state."""
        with self._state_lock:
            if self.stream.is_ready(self.recognizer):
                self.recognizer.decode(self.stream)

            result = self.stream.get_result()
            if result.is_final:
                self.state.final_text = result.text
            else:
                self.state.partial_text = result.text
```

**Key Points**:
- Each model runs independently with its own stream
- Thread-safe using `threading.RLock()`
- State tracking: partial text, final text, chunk count
- Audio format conversion (float32 → int16)

#### ASREnginePool

```python
class ASREnginePool:
    def __init__(self, config=None):
        self.config = config or ASRConfig()
        self.models = {}
        self._pool_lock = threading.Lock()

    def load_models(self) -> bool:
        """Load all 3 ASR models."""
        with self._pool_lock:
            for model_id, model_config in self.config.models.items():
                try:
                    recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
                        model_type="zipformer",
                        zipformer=sherpa_onnx.OnlineZipformerModelConfig(
                            encoder=model_config.encoder_path,
                            decoder=model_config.decoder_path,
                            joiner=model_config.joiner_path,
                        ),
                        tokens=model_config.tokens_path,
                        num_threads=model_config.num_threads,
                        provider=model_config.provider,
                        decoding_method=RecognitionConfig.DECODING_METHOD,
                        bpe_vocab=model_config.bpe_vocab_path if model_config.hotwords else None,
                    )

                    recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
                    model_instance = ModelInstance(
                        model_id=model_id,
                        model_config=model_config,
                        recognizer=recognizer,
                    )
                    self.models[model_id] = model_instance

                except Exception as e:
                    print(f"Failed to load {model_id}: {e}")
                    return False

        return True

    def feed_audio_chunk(self, audio: np.ndarray) -> int:
        """Feed audio to all models."""
        count = 0
        for model_instance in self.models.values():
            try:
                model_instance.feed_audio(audio)
                count += 1
            except Exception as e:
                print(f"Error feeding audio: {e}")

        self._audio_chunks_fed += 1
        return count

    def process(self) -> None:
        """Process all streams."""
        for model_instance in self.models.values():
            try:
                model_instance.process()
            except Exception as e:
                print(f"Error processing: {e}")

    def get_results(self) -> Dict[str, StreamResult]:
        """Get snapshot of all results."""
        results = {}
        for model_id, model_instance in self.models.items():
            try:
                results[model_id] = model_instance.get_result_snapshot()
            except Exception as e:
                print(f"Error getting result: {e}")

        return results
```

**Key Points**:
- Uses advanced config (OnlineRecognizerConfig + OnlineZipformerModelConfig)
- Thread-safe operations with locks
- Models loaded on demand in `load_models()`
- Results returned as thread-safe snapshots

### Usage Pattern

```python
# 1. Create pool
pool = ASREnginePool()

# 2. Load all models
pool.load_models()

# 3. Feed audio from microphone
with sd.InputStream(...) as s:
    while True:
        samples, _ = s.read(samples_per_chunk)
        pool.feed_audio_chunk(samples)

        # 4. Process
        pool.process()

        # 5. Get results from all models
        results = pool.get_results()
        for model_id, result in results.items():
            print(f"{model_id}: {result.partial}")

# 6. Cleanup
pool.cleanup()
```

### Configuration Integration

Uses `asr_config.py` to define 3 models:

```python
models_config = {
    "small-bilingual": ModelConfig(
        name="Small Bilingual (zh-en 2023-02-16)",
        encoder_path="models/.../encoder.onnx",
        decoder_path="models/.../decoder.onnx",
        joiner_path="models/.../joiner.onnx",
        tokens_path="models/.../tokens.txt",
        bpe_vocab_path="models/.../bpe.vocab",
        hotwords=HotwordConfig(["KEN LI", "JOHN SMITH"], boost_score=2.0),
    ),
    "medium-bilingual": ModelConfig(...),
    "multilingual": ModelConfig(...),
}
```

### When to Use

- ✓ Compare multiple ASR models
- ✓ Need redundancy (if one model fails, others still work)
- ✓ Want to measure model differences
- ✓ Have sufficient computational resources
- ✗ Need minimum latency (3 models = slower)
- ✗ Have limited GPU/CPU resources

---

## File 4: asr_config.py

**Location**: `/home/luigi/sherpa/asr_config.py`
**Purpose**: Centralized configuration for all 3 ASR models
**Complexity**: Configuration/Data Structure
**Related to**: asr_engine.py (uses this config)

### Structure

```python
# Model-specific configuration
@dataclass
class ModelConfig:
    name: str                          # Display name
    encoder_path: str                  # Path to encoder ONNX
    decoder_path: str                  # Path to decoder ONNX
    joiner_path: str                   # Path to joiner ONNX
    tokens_path: str                   # Path to tokens.txt
    bpe_model_path: Optional[str]      # Path to BPE model
    bpe_vocab_path: Optional[str]      # Path to BPE vocab
    hotwords: Optional[HotwordConfig]  # Hotword configuration
    sample_rate: int = 16000           # IMPORTANT: 16kHz
    num_threads: int = 1               # Thread count
    provider: str = "cpu"              # ONNX provider

# Container for all models
class ASRConfig:
    def __init__(self, models_dir="/home/luigi/sherpa/models"):
        self.models_dir = models_dir
        self.models = {}
        self._define_models()

    def _define_models(self):
        # Define 3 models here
```

### 3 Predefined Models

1. **Small Bilingual (zh-en 2023-02-16)**
   - Compact, fast
   - Chinese + English
   - Has bpe.vocab for hotwords
   - 1 thread

2. **Medium Bilingual (zh-en 2023-02-20)**
   - Medium size, better accuracy
   - Chinese + English
   - No bpe.vocab (hotwords not available)
   - 1 thread

3. **Multilingual (ar_en_id_ja_ru_th_vi_zh 2025-02-10)**
   - State-of-art
   - 8 languages
   - Has bpe.vocab for hotwords
   - 1 thread

### Usage

```python
from asr_config import ASRConfig

config = ASRConfig()

# Get all models
for model_id, name in config.list_models().items():
    print(f"{model_id}: {name}")

# Get specific model
small_model = config.get_model("small-bilingual")

# Validate all
all_valid, results = config.validate_all()
```

---

## Comparison Table

| Feature | speech_recognition_mic.py | speech_recognition_mic_hotwords.py | asr_engine.py |
|---------|---------------------------|-------------------------------------|---------------|
| **Complexity** | Beginner | Intermediate | Advanced |
| **Models** | 1 (single) | 1 (single) | 3 (parallel) |
| **Hotwords** | ✗ | ✓ | ✓ (per model) |
| **Threading** | None | None | ✓ (thread-safe) |
| **Config** | CLI args | CLI args | asr_config.py |
| **Real-time** | ✓ | ✓ | ✓ (slightly slower) |
| **Best for** | Learning | Domain-specific terms | Model comparison |

---

## Quick Summary

### For Learning
Use: **speech_recognition_mic.py**
- Simplest code
- Shows all fundamental concepts
- Easy to modify

### For Production with Hotwords
Use: **speech_recognition_mic_hotwords.py**
- Add `--hotwords` argument
- Small accuracy boost
- Requires bpe.vocab

### For Research/Comparison
Use: **asr_engine.py**
- Run 3 models in parallel
- Compare outputs
- More resource-intensive

---

## Integration Points

All examples use:
1. `sherpa_onnx.OnlineRecognizer` - Core API
2. `sounddevice` - Microphone input
3. `numpy` - Audio array handling

Only asr_engine.py uses:
- `threading` - Parallel execution
- `asr_config.py` - Centralized config
- `dataclasses` - Type-safe config

