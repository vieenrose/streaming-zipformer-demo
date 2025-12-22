# OnlineRecognizer Documentation Index

Complete documentation for using `sherpa_onnx.OnlineRecognizer` in the sherpa repository.

---

## Documentation Files

### 1. **ONLINERECOGNIZER_GUIDE.md** - Complete Reference
**For**: Comprehensive understanding of the OnlineRecognizer API
**Contains**:
- Complete API reference
- All methods and parameters explained
- Hotwords detailed guide
- Multiple usage patterns
- Key parameters reference table
- Common issues and solutions
- Full examples with explanations

**When to read**: 
- First time learning the API
- Need detailed explanation of a feature
- Want to understand all capabilities

---

### 2. **ONLINERECOGNIZER_QUICK_REFERENCE.md** - Fast Lookup
**For**: Quick code snippets and patterns
**Contains**:
- Minimal working example (30 seconds)
- Recognizer creation patterns (4 variants)
- Stream operation snippets
- Audio input patterns
- Hotwords quick setup (3 steps)
- Common workflows
- Configuration reference
- Debugging tips
- Common errors and fixes

**When to read**:
- Need quick code snippet
- Forgot specific parameter
- Need to copy-paste example
- Quick reference while coding

---

### 3. **EXAMPLE_DOCUMENTATION.md** - Real-World Examples
**For**: Understanding actual examples in this repository
**Contains**:
- speech_recognition_mic.py (Basic)
- speech_recognition_mic_hotwords.py (With hotwords)
- asr_engine.py (Parallel processing)
- asr_config.py (Configuration)
- Detailed analysis of each file
- Usage examples
- Comparison table
- When to use each example

**When to read**:
- Want to use an example file
- Need to understand example code
- Deciding which example to start with
- Want to adapt example for your use case

---

### 4. **HOTWORDS_WORKING.md** (Existing)
**Location**: `/home/luigi/sherpa/docs/HOTWORDS_WORKING.md`
**For**: Hotwords-specific details and testing
**Contains**:
- Hotwords format rules
- Working examples with proof
- Multi-token hotwords
- Chinese character support
- Implementation details
- Performance metrics

**When to read**:
- Using hotwords feature
- Hotwords not working
- Want proof hotwords work
- Need to troubleshoot hotwords

---

## Quick Navigation

### Getting Started (First Time)

1. Read: **ONLINERECOGNIZER_QUICK_REFERENCE.md** → "Minimal Working Example"
2. Read: **EXAMPLE_DOCUMENTATION.md** → "speech_recognition_mic.py"
3. Try: Run `/home/luigi/sherpa/speech_recognition_mic.py`
4. Read: **ONLINERECOGNIZER_GUIDE.md** → sections as needed

### Adding Hotwords

1. Read: **ONLINERECOGNIZER_QUICK_REFERENCE.md** → "Hotwords Quick Reference"
2. Read: **ONLINERECOGNIZER_GUIDE.md** → "Hotwords Support"
3. Read: **EXAMPLE_DOCUMENTATION.md** → "speech_recognition_mic_hotwords.py"
4. Try: Run `/home/luigi/sherpa/speech_recognition_mic_hotwords.py`

### Parallel Model Comparison

1. Read: **EXAMPLE_DOCUMENTATION.md** → "asr_engine.py"
2. Read: **EXAMPLE_DOCUMENTATION.md** → "asr_config.py"
3. Check: **ONLINERECOGNIZER_GUIDE.md** → "Advanced Usage" → "Multiple Streams"

### Troubleshooting

1. Check: **ONLINERECOGNIZER_QUICK_REFERENCE.md** → "Common Errors and Fixes"
2. Check: **ONLINERECOGNIZER_GUIDE.md** → "Common Issues and Solutions"
3. Check: **HOTWORDS_WORKING.md** → (if hotword-related)

---

## File Locations

### Documentation Files (You are reading these)
```
/home/luigi/sherpa/
├── ONLINERECOGNIZER_GUIDE.md           ← Complete reference
├── ONLINERECOGNIZER_QUICK_REFERENCE.md ← Quick lookup
├── EXAMPLE_DOCUMENTATION.md            ← Real examples explained
├── ONLINERECOGNIZER_INDEX.md           ← This file
└── docs/
    └── HOTWORDS_WORKING.md             ← Hotwords details
```

### Example Python Files
```
/home/luigi/sherpa/
├── speech_recognition_mic.py           ← Basic (Start here)
├── speech_recognition_mic_hotwords.py  ← With hotwords
├── asr_engine.py                       ← Parallel processing
├── asr_config.py                       ← Configuration
└── demo_full_integration.py            ← Full integration demo
```

### Models Directory
```
/home/luigi/sherpa/models/
├── sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile/
│   ├── tokens.txt
│   ├── encoder-epoch-99-avg-1.int8.onnx
│   ├── decoder-epoch-99-avg-1.onnx
│   ├── joiner-epoch-99-avg-1.int8.onnx
│   ├── bpe.model
│   └── bpe.vocab
├── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-mobile/
│   └── (similar files)
└── sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10/
    └── (similar files)
```

---

## API Cheat Sheet

### Creating Recognizer
```python
# Simple (recommended)
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="path/tokens.txt",
    encoder="path/encoder.onnx",
    decoder="path/decoder.onnx",
    joiner="path/joiner.onnx",
)

# Advanced (with config)
config = sherpa_onnx.OnlineRecognizerConfig(
    model_type="zipformer",
    zipformer=sherpa_onnx.OnlineZipformerModelConfig(...),
    tokens="path/tokens.txt",
)
recognizer = sherpa_onnx.OnlineRecognizer(config)
```

### Stream Lifecycle
```python
stream = recognizer.create_stream()                    # 1. Create
stream.accept_waveform(16000, samples)                # 2. Feed audio
while recognizer.is_ready(stream):                    # 3. Check ready
    recognizer.decode_stream(stream)                  # 4. Decode
result = recognizer.get_result(stream)                # 5. Get result
if recognizer.is_endpoint(stream):                    # 6. Check endpoint
    recognizer.reset(stream)                          # 7. Reset
```

### With Hotwords
```python
config = sherpa_onnx.OnlineRecognizerConfig(
    ...,
    decoding_method="modified_beam_search",  # CRITICAL
    bpe_vocab="path/bpe.vocab",              # REQUIRED
)
recognizer = sherpa_onnx.OnlineRecognizer(config)
stream = recognizer.create_stream(
    hotwords="HOTWORD1\nHOTWORD2",
    hotwords_score=2.0
)
```

---

## Key Concepts

### Sample Rate
- **Microphone**: Any rate (typically 44100, 48000 Hz)
- **ASR Model**: Always 16000 Hz (internal)
- **Accept_waveform**: Pass actual microphone sample rate
  ```python
  stream.accept_waveform(sample_rate=48000, waveform=samples)  # ✓ Correct
  stream.accept_waveform(sample_rate=16000, waveform=samples)  # ✗ Wrong if mic is 48kHz
  ```

### Audio Chunks
- **Duration**: Typically 100ms (standard)
- **Sample count**: sample_rate * duration_seconds
  - At 48000 Hz: 100ms = 4800 samples
  - At 16000 Hz: 100ms = 1600 samples

### Endpoint Detection
- **Concept**: Automatically detect when user stops speaking
- **Rules**: 3 configurable rules for determining endpoint
- **Usage**: Triggers `is_endpoint()` to return True
- **Action**: Always `reset()` stream after endpoint

### Hotwords
- **Purpose**: Boost recognition of specific terms
- **Format**: ALL UPPERCASE, space-separated
  - "KEN LI" ✓, "Ken Li" ✗, "KENLI" ✗
- **Requirements**: modified_beam_search + bpe.vocab
- **Performance**: 2-8% general improvement, up to 25% targeted

### Decoding Methods
- **greedy_search**: Faster, less accurate, no hotwords
- **modified_beam_search**: Slower, more accurate, supports hotwords

---

## Workflow Decision Tree

```
┌─ START: Need speech recognition?
│
├─ Just learning the API?
│  └─→ Read: ONLINERECOGNIZER_QUICK_REFERENCE.md
│      Try: speech_recognition_mic.py
│
├─ Need hotword boosting?
│  ├─ Know hotword format rules?
│  │  └─→ Read: ONLINERECOGNIZER_QUICK_REFERENCE.md → "Hotwords Quick Reference"
│  │      Try: speech_recognition_mic_hotwords.py
│  │
│  └─ Need to understand hotwords deeply?
│     └─→ Read: docs/HOTWORDS_WORKING.md
│         Read: ONLINERECOGNIZER_GUIDE.md → "Hotwords Support"
│
├─ Want to compare multiple models?
│  └─→ Read: EXAMPLE_DOCUMENTATION.md → "asr_engine.py"
│      Try: asr_engine.py
│
├─ Need detailed API reference?
│  └─→ Read: ONLINERECOGNIZER_GUIDE.md
│
├─ Looking for code snippet?
│  └─→ Check: ONLINERECOGNIZER_QUICK_REFERENCE.md
│
├─ Have error/problem?
│  ├─ Check: ONLINERECOGNIZER_QUICK_REFERENCE.md → "Common Errors and Fixes"
│  └─ Check: ONLINERECOGNIZER_GUIDE.md → "Common Issues and Solutions"
│
└─ END: Use documentation to implement
```

---

## Most Common Tasks

### Task 1: Basic Microphone Transcription
**File**: `EXAMPLE_DOCUMENTATION.md` → speech_recognition_mic.py
**Time**: 5 minutes to understand
**Code lines**: ~50

### Task 2: Add Hotwords to Your System
**File**: `ONLINERECOGNIZER_QUICK_REFERENCE.md` → "Hotwords Quick Reference"
**Time**: 10 minutes
**Changes**: ~10 lines (3-step setup)

### Task 3: Debug Why Hotwords Aren't Working
**File 1**: `ONLINERECOGNIZER_QUICK_REFERENCE.md` → "Common Errors and Fixes"
**File 2**: `docs/HOTWORDS_WORKING.md` → "Important Notes"
**Common issue**: Wrong case or not using modified_beam_search

### Task 4: Run Multiple Models in Parallel
**File**: `EXAMPLE_DOCUMENTATION.md` → "asr_engine.py"
**Time**: 20 minutes to understand
**Complexity**: Advanced (but example code works)

### Task 5: Convert from Another ASR System
**File**: `ONLINERECOGNIZER_GUIDE.md` → "Complete Examples"
**Key concept**: Stream-based (not batch-based)

---

## Important Notes

### DO
- ✓ Always reset stream after endpoint detection
- ✓ Use "modified_beam_search" for hotwords
- ✓ Keep hotwords ALL UPPERCASE
- ✓ Pass actual microphone sample rate to accept_waveform()
- ✓ Feed audio in regular chunks (100ms is standard)
- ✓ Check is_ready() before decoding

### DON'T
- ✗ Use "greedy_search" for hotwords (won't work)
- ✗ Mix case in hotwords ("Ken Li" won't work)
- ✗ Forget to pass bpe_vocab for hotwords
- ✗ Pass hardcoded 16000 sample rate when mic is different
- ✗ Call decode_stream() without checking is_ready()
- ✗ Forget to reset stream on endpoint

---

## Support and References

### Official Documentation
- [sherpa-onnx docs](https://k2-fsa.github.io/sherpa/onnx/index.html)
- [ASR models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
- [Hotwords guide](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
- [Example scripts](https://github.com/k2-fsa/sherpa-onnx/tree/master/python-api-examples)

### This Repository
- **Guide**: ONLINERECOGNIZER_GUIDE.md (this directory)
- **Quick ref**: ONLINERECOGNIZER_QUICK_REFERENCE.md (this directory)
- **Examples**: speech_recognition_mic.py, speech_recognition_mic_hotwords.py, asr_engine.py
- **Config**: asr_config.py
- **Hotwords**: docs/HOTWORDS_WORKING.md

---

## Document Version

- **Created**: 2025-12-22
- **Based on**: Repository at /home/luigi/sherpa
- **Examples analyzed**: 4 files (speech_recognition_mic.py, hotwords variant, asr_engine.py, asr_config.py)
- **Models documented**: 3 models (small-bilingual, medium-bilingual, multilingual)

---

## Questions This Documentation Answers

**API Basics**
- How do I create a recognizer?
- How do I create a stream?
- What parameters are required/optional?

**Audio Handling**
- What sample rate should I use?
- How do I feed audio from microphone?
- What chunk size is best?

**Results**
- How do I get transcription results?
- What's partial vs final?
- When should I reset?

**Hotwords**
- How do I add hotword support?
- What's the correct format?
- Why aren't hotwords working?

**Advanced**
- How do I use multiple models?
- How do I make it thread-safe?
- How do I integrate with my system?

**Troubleshooting**
- Model fails to load - why?
- Results are empty - why?
- Hotwords not recognized - why?

---

See individual files for detailed information on each topic.
