# Synthetic Test Audio Generation with gTTS

## Overview

This enhancement adds **Google Text-to-Speech (gTTS)** synthetic audio generation to the hotword testing framework, enabling:

- **Deterministic, reproducible test data** - Same text always produces consistent audio
- **Complete ground truth** - Exact text reference for WER calculation
- **Batch generation** - Create multiple test utterances programmatically
- **Multiple languages** - Support for English, Chinese, and other languages
- **Cost-free generation** - Uses gTTS service, no license required

## Why Synthetic Audio?

### Advantages over Recorded Audio
- **Consistency**: No speaker variation, background noise, or recording artifacts
- **Reproducibility**: Run tests multiple times with identical audio
- **Scalability**: Generate hundreds of test cases programmatically
- **Known Ground Truth**: Text input = exact reference for accuracy calculation
- **Language Purity**: Consistent pronunciation and enunciation

### Use Cases

1. **Baseline Testing**: Establish hotword effectiveness under ideal conditions
2. **Regression Testing**: Verify performance consistency after code changes
3. **Comparative Testing**: Compare natural vs synthetic audio results
4. **Batch Validation**: Test multiple hotwords systematically
5. **CI/CD Integration**: Automated test suites with deterministic inputs

## Installation

### Install gTTS

```bash
pip install gtts
```

The script will auto-install gTTS if not present.

## Usage

### Generate Synthetic Audio

```bash
python3 generate_synthetic_test_audio.py
```

**What it does:**

1. **Phase 1**: Generates 9 English person name utterances
   - Ken Li, John Smith, Alice Johnson
   - Multiple sentences per person (variation)

2. **Phase 2**: Generates 9 mixed language utterances
   - Mark Ma, Lucy Chen, Peter Wang
   - English speech with person names

3. **Phase 3**: Generates 9 Chinese utterances (Traditional Chinese - zh-TW)
   - 特雷威 (Terwei), 林志玲 (Zhi Ling Lin), 王小明 (Xiaoming Wang)
   - Multiple sentences per name

4. **Phase 4**: Generates metadata.json
   - Maps utterance IDs to reference text
   - Categorizes by language/type
   - Tracks expected hotwords

**Output:**

```
Output directory: tests/test_audio_synthetic/

Generated Files:
  English names:       9 MP3 files
  Mixed language:      9 MP3 files
  Chinese names:       9 MP3 files
  ─────────────────────────
  Total:              27 MP3 files
  Total Size:         ~15-20 MB
  Metadata:           metadata.json
```

### Test with Synthetic Audio

```bash
# Use enhanced test with automatic audio source detection
python3 test_hotword_impact_enhanced.py

# Or specify synthetic directory
python3 test_hotword_impact_enhanced.py
```

**What it does:**

1. Discovers synthetic audio files and metadata
2. Loads 27 synthetic utterances
3. Runs A/B test: WITH vs WITHOUT hotwords
4. Computes WER improvements for each model
5. Generates report: `hotword_impact_report_synthetic.json`

### Compare Synthetic vs Natural

```bash
# Test 1: Generate and test synthetic audio
python3 generate_synthetic_test_audio.py
python3 test_hotword_impact_enhanced.py

# Test 2: Test with natural audio (if available)
# Rename to force natural audio test or modify script

# Compare results
diff <(jq .summary.overall_avg_improvement hotword_impact_report_synthetic.json) \
     <(jq .summary.overall_avg_improvement hotword_impact_report_natural.json)
```

## Test Utterances

### English Person Names (9 utterances)

Each person has 3 variations:

**Ken Li**
1. "My name is Ken Li"
2. "I work with Ken Li"
3. "Can I speak with Ken Li"

**John Smith**
1. "John Smith is my colleague"
2. "Please contact John Smith"
3. "John Smith called earlier"

**Alice Johnson**
1. "Alice Johnson is available"
2. "Please schedule with Alice Johnson"
3. "Alice Johnson will handle it"

### Mixed Language Utterances (9 utterances)

**Mark Ma**
1. "Mark Ma is here"
2. "We need Mark Ma for the meeting"
3. "Mark Ma is the project manager"

**Lucy Chen**
1. "Lucy Chen presented today"
2. "Lucy Chen is our designer"
3. "Contact Lucy Chen for details"

**Peter Wang**
1. "Peter Wang leads the team"
2. "Peter Wang sent the email"
3. "I spoke with Peter Wang yesterday"

### Chinese Utterances (9 utterances - Traditional Chinese)

**特雷危 (Terwei)**
1. "特雷危"
2. "我叫特雷危"
3. "特雷威在這裡"

**林志玲 (Zhi Ling Lin)**
1. "林志玲"
2. "我是林志玲"
3. "林志玲很有才華"

**王小明 (Xiaoming Wang)**
1. "王小明"
2. "王小明來了"
3. "我叫王曉明"

## Generated Metadata Format

**metadata.json:**

```json
{
  "generated": "2025-12-22T16:45:00.123456",
  "total_utterances": 27,
  "utterances": [
    {
      "id": "ken_li_01",
      "text": "My name is Ken Li",
      "lang": "en",
      "category": "english",
      "hotwords": ["KEN LI"],
      "speed": 1.0
    },
    ...
  ],
  "categories": {
    "english": ["ken_li_01", "ken_li_02", ...],
    "mixed": ["mark_ma_01", "mark_ma_02", ...],
    "chinese": ["te_lei_wei_01", "te_lei_wei_02", ...]
  }
}
```

## Expected Results with Synthetic Audio

### By Category

| Category | Expected | Reason |
|----------|----------|--------|
| English Names | +8-15% | Clear person names |
| Mixed Language | +10-20% | Language-specific boost |
| Chinese Names | +0-5% | May be in training data |

### By Model

| Model | Expected | Notes |
|-------|----------|-------|
| Small Bilingual | +7-12% | Good baseline |
| Medium Bilingual | +10-15% | Best performance |
| Multilingual | +8-14% | Excellent generalization |

### Synthetic vs Natural Comparison

**Synthetic Audio Characteristics:**
- Higher baseline accuracy (no background noise)
- Consistent speech rate and pronunciation
- Clean audio conditions
- May show higher hotword impact

**Natural Audio Characteristics:**
- More realistic conditions
- Speaker variations
- Background noise challenges
- Shows real-world effectiveness

**Comparison Insight:**
```
Synthetic - Natural = Impact of ideal conditions
Typically: Synthetic 5-10% higher baseline accuracy
```

## Configuration

### Generate with Different Speech Rate

Edit `generate_synthetic_test_audio.py` to adjust utterance speed:

```python
UtteranceDefinition(
    id="ken_li_01",
    text="My name is Ken Li",
    lang="en",
    category="english",
    hotwords=["KEN LI"],
    speed=0.8  # Slow speech (0.5-2.0 range)
)
```

### Add Custom Utterances

Extend the utterance lists:

```python
ENGLISH_UTTERANCES.append(
    UtteranceDefinition(
        id="custom_01",
        text="Your custom text here",
        lang="en",
        category="english",
        hotwords=["YOUR HOTWORD"],
        speed=1.0
    )
)
```

### Specify Language/Locale

gTTS supports many languages:

```python
# Use specific accent
"en"      # Generic English
"en-US"   # US English
"en-GB"   # British English
"en-IN"   # Indian English
"zh-TW"   # Traditional Chinese
"zh-CN"   # Simplified Chinese
```

## API Details

### generate_synthetic_test_audio.py

**Functions:**

- `synthesize_utterance(utterance, output_dir, slow=False)` - Generate audio for single utterance
- `generate_metadata(utterances, output_dir)` - Create metadata.json file

**Output:**
- MP3 files: `{utterance_id}.mp3`
- Metadata: `metadata.json`

### test_hotword_impact_enhanced.py

**Features:**

- Auto-detects synthetic audio via metadata.json
- Falls back to natural audio if synthetic unavailable
- Generates source-specific report names
- Tracks audio source in results

**Key Difference from Original:**

```python
# Original: Fixed directory
test_utterances = [...]

# Enhanced: Auto-discover + configurable
synthetic_dir = Path(".../test_audio_synthetic")
natural_dir = Path(".../test_audio_hotwords")
test_files = discover_audio()  # Smart discovery
```

## Troubleshooting

### Issue: gTTS not found

```bash
pip install gtts
```

### Issue: Slow generation

gTTS makes HTTP requests per utterance. This is normal:
- Expected time: 10-30 seconds for 27 utterances
- Depends on network speed
- Consider batch generation in future

### Issue: Language not supported

Ensure language code is valid:

```python
# Valid
"en", "zh-TW", "ja", "fr", "es"

# Invalid (use standard codes)
"english"  # ✗
"en-US"    # ✓
```

### Issue: Audio quality low

gTTS can have quality variations. To improve:

1. Use slower speech rate: `slow=True` or `speed=0.8`
2. Simplify text (shorter sentences)
3. Use TLD setting: `tld='com'` (default) or `'co.uk'`

## Integration with Hotword Testing

### Workflow

```
1. Generate Synthetic Audio
   python3 generate_synthetic_test_audio.py
   ↓
2. Run A/B Test (Synthetic)
   python3 test_hotword_impact_enhanced.py
   ↓
3. Review Results
   cat hotword_impact_report_synthetic.json
   ↓
4. Compare with Natural Audio (if available)
   python3 test_hotword_impact_with_files.py
   ↓
5. Analyze Differences
   # Synthetic vs Natural WER comparison
```

### Sample Analysis Script

```python
import json

# Load reports
with open('hotword_impact_report_synthetic.json') as f:
    synthetic = json.load(f)
with open('hotword_impact_report_natural.json') as f:
    natural = json.load(f)

# Compare
syn_improvement = synthetic['summary']['overall_avg_improvement']
nat_improvement = natural['summary']['overall_avg_improvement']

print(f"Synthetic: {syn_improvement*100:+.1f}%")
print(f"Natural:   {nat_improvement*100:+.1f}%")
print(f"Difference: {(syn_improvement - nat_improvement)*100:+.1f}%")
```

## Advanced: Batch Testing

Create test matrix combining:
- **Audio Sources:** Synthetic, Natural, Mixed
- **Hotword Boost Scores:** 1.5, 2.0, 2.5, 3.0
- **Models:** All 3 ASR models
- **Categories:** English, Mixed, Chinese

```python
# Example: Test boost score variations
for boost_score in [1.5, 2.0, 2.5, 3.0]:
    RecognitionConfig.HOTWORDS_BOOST_SCORE = boost_score
    run_test()  # Generate report for each boost level
```

## Dependencies

**New dependency:**
- `gtts` - Google Text-to-Speech library
- Install: `pip install gtts`

**Existing dependencies (unchanged):**
- numpy
- soundfile
- sherpa-onnx

**Total additions:** 1 lightweight library (~50 KB)

## Summary

This enhancement adds **cost-free, deterministic, reproducible test audio generation** using gTTS:

✓ 27 synthetic test utterances (English, Mixed, Chinese)
✓ Complete metadata with ground truth references
✓ Perfect for baseline hotword testing
✓ Easy comparison with natural audio results
✓ Foundation for automated CI/CD testing
✓ Minimal new dependencies (only gTTS)

---

**Quick Start:**

```bash
# Generate synthetic audio
python3 generate_synthetic_test_audio.py

# Run hotword impact test
python3 test_hotword_impact_enhanced.py

# View results
cat hotword_impact_report_synthetic.json | python3 -m json.tool
```

---

**Implementation Date:** December 22, 2025
**Status:** Complete and Ready for Use
**Next Step:** Compare synthetic vs natural audio effectiveness
