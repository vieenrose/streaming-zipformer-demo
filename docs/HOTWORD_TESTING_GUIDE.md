# Hotword Impact Testing Guide

## Overview

This guide explains the hotword impact testing suite for proving the effectiveness of hotword biasing on ASR (Automatic Speech Recognition) accuracy.

Two main test scripts are provided:

1. **test_hotword_impact_with_files.py** - Comprehensive hotword impact analysis using test audio files
2. **demo_full_integration_file.py** - Enhanced demo supporting both microphone and file input modes

## Test Methodology

### What is Hotword Biasing?

Hotword biasing is a technique that increases the probability of recognizing specific words or phrases during transcription. By providing a list of "hotwords" to the ASR model, we can:

- Improve recognition accuracy for person names
- Handle multilingual content (English + Chinese)
- Boost rare or uncommon words in the domain

### Test Design

The test suite compares recognition accuracy in two conditions:

1. **WITH Hotwords**: ASR models running with hotword biasing enabled
2. **WITHOUT Hotwords**: Same models running without hotword biasing

This A/B testing approach allows us to quantitatively measure the impact using Word Error Rate (WER).

### Test Audio Categories

The test suite covers three categories of utterances:

#### 1. English Person Names
- "My name is Ken Li"
- "I work with John Smith"
- "Please call Alice Johnson"

**Hotwords:** KEN LI, JOHN SMITH, ALICE JOHNSON

#### 2. Mixed Language (English + Chinese)
- "Mark Ma is here"
- "Lucy Chen called"
- "Peter Wang joined"

**Hotwords:** MARK MA, LUCY CHEN, PETER WANG

#### 3. Pure Chinese Names
- "特雷威" (Terwei)
- "林志玲" (Zhi Ling Lin)
- "王晓明" (Xiaoming Wang)

**Hotwords:** 特雷危, 林志玲, 王小明

## Usage

### 1. File-Based Hotword Impact Test

Run the comprehensive hotword impact analysis:

```bash
python3 test_hotword_impact_with_files.py
```

**What it does:**
- Loads test audio files from `tests/test_audio_hotwords/`
- Initializes two ASR engine pools (with and without hotwords)
- Processes each audio file through both pools
- Computes WER (Word Error Rate) for each model
- Generates detailed impact analysis and statistics
- Saves JSON report to `hotword_impact_report.json`

**Expected output:**
- Per-test results showing recognition text
- WER metrics comparing with vs without hotwords
- Per-model improvement statistics
- Per-category (English/Chinese/Mixed) analysis
- Overall improvement percentage

### 2. File Input Demo

Test with your own audio files:

```bash
# Use microphone (original behavior)
python3 demo_full_integration_file.py

# Use file input
python3 demo_full_integration_file.py --file path/to/audio.mp3

# Specify output file
python3 demo_full_integration_file.py --file audio.mp3 --output output.mp3

# Specify recording duration
python3 demo_full_integration_file.py --duration 15
```

**Features:**
- Supports both microphone input (with VAD device detection) and file input
- Streams audio through 3 ASR models in parallel
- Records MP3 output in background
- Displays live transcription results
- Shows per-model latency metrics

## Metrics Explained

### Word Error Rate (WER)

WER is calculated using Levenshtein distance on word sequences:

```
WER = (S + D + I) / N

Where:
  S = number of substitutions (wrong word recognized)
  D = number of deletions (word missing)
  I = number of insertions (extra word added)
  N = number of words in reference text
```

**Interpretation:**
- WER = 0.0: Perfect recognition (no errors)
- WER = 0.5: 50% of words were incorrect
- WER = 1.0 or higher: Most/all words were incorrect

### Improvement Metric

```
Improvement = WER_without_hotwords - WER_with_hotwords
```

**Interpretation:**
- +0.2 = 20% improvement (WER reduced by 0.2)
- 0.0 = No change
- Negative = Worse with hotwords (rare)

## Expected Results

Based on sherpa-onnx documentation and similar systems:

### English Names
- **Expected improvement:** 5-15%
- **Best models:** Multilingual models (3+ languages)
- **Reason:** Better generalization to person names

### Mixed Language
- **Expected improvement:** 10-20%
- **Best models:** Multilingual and bilingual zh-en models
- **Reason:** Direct language selection via hotwords

### Pure Chinese
- **Expected improvement:** 0-5%
- **Note:** Model training data may already include these names
- **Reason:** High baseline accuracy already

## ASR Models Tested

The test suite uses 3 sherpa-onnx streaming models:

### Model 1: Small Bilingual (zh-en)
- Name: `sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16`
- File size: ~25 MB
- Speed: Fast (good for real-time)
- Accuracy: Baseline

### Model 2: Medium Bilingual (zh-en)
- Name: `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`
- File size: ~50 MB
- Speed: Medium
- Accuracy: Higher than Model 1

### Model 3: Multilingual (7 languages)
- Name: `sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10`
- File size: ~200 MB
- Speed: Slower but highest accuracy
- Languages: Arabic, English, Indonesian, Japanese, Russian, Thai, Vietnamese, Chinese

## Hotword Configuration

### Hotword Settings

Located in `asr_config.py`:

```python
RecognitionConfig.ENABLE_HOTWORDS = True        # Enable hotword biasing
RecognitionConfig.HOTWORDS_BOOST_SCORE = 2.0    # Boost score (1.5-3.0 range)
RecognitionConfig.DECODING_METHOD = "modified_beam_search"  # Required for hotwords
```

### Hotword List Format

Hotwords are:
- **Uppercase** (e.g., "KEN LI", not "ken li")
- **Space-separated** for multi-token names
- **Without OOV (Out-of-Vocabulary) tokens** - all subwords must be in model's vocabulary

### Per-Model Hotword Configuration

Each model can have different hotword boost scores:

```python
hotwords=HotwordConfig(
    hotwords=["KEN LI", "JOHN SMITH", "特雷危"],
    boost_score=2.0  # Can be adjusted per model
)
```

## Interpreting Results

### Example Output

```
Per-Model Impact:
─────────────────────────────────────────────────────────
  small-bilingual-zh-en     : +8.50% avg | +15.20% max | 15/18 tests improved
  medium-bilingual-zh-en    : +12.30% avg | +22.50% max | 17/18 tests improved
  multilingual-7lang        : +10.20% avg | +18.90% max | 16/18 tests improved

Per-Category Impact:
─────────────────────────────────────────────────────────
  english                   : +10.50% avg (18 comparisons)
  mixed                     : +14.80% avg (18 comparisons)
  chinese                   : +2.30% avg (18 comparisons)
```

**Analysis:**
1. **Overall improvement is positive** - hotwords help on average
2. **English and mixed language benefit most** - expected, as person names are context
3. **Medium model benefits most** - good balance of accuracy and model capacity
4. **Multilingual model consistent** - good generalization

## Troubleshooting

### Issue: "No test audio files found"
**Solution:** Ensure audio files exist in `tests/test_audio_hotwords/`

### Issue: "Failed to load ASR models"
**Solution:** Models should be in `models/` directory. Run step 2 from README if missing.

### Issue: Low or negative improvements
**Possible causes:**
- Hotword list has OOV tokens (words not in model vocabulary)
- Boost score is too low (try increasing to 3.0)
- Decoding method not set to modified_beam_search
- Model training data already contains these names

### Issue: Timeout during processing
**Solution:** Reduce number of test utterances or increase timeout value in code

## Advanced: Custom Hotword Testing

To test with your own hotwords:

1. Edit `test_hotword_impact_with_files.py`
2. Modify the `test_utterances` list:

```python
test_utterances = [
    TestUtterance(
        name="my_test",
        audio_file="path/to/audio.mp3",
        reference_text="Expected transcription",
        category="english",
        hotword_matches=["YOUR HOTWORD", "ANOTHER NAME"]
    ),
]
```

3. Update hotwords in `asr_config.py`:

```python
hotwords=HotwordConfig(
    hotwords=["YOUR HOTWORD", "ANOTHER NAME"],
    boost_score=2.5  # Try different values
)
```

4. Run the test and observe results

## Report Format

The JSON report saved to `hotword_impact_report.json` contains:

```json
{
  "metadata": {
    "timestamp": "2025-12-22 16:30:00",
    "num_tests": 7,
    "num_models": 3,
    "total_comparisons": 21
  },
  "summary": {
    "overall_avg_improvement": 0.098,
    "improvement_by_model": {
      "small-bilingual-zh-en": {
        "avg": 0.085,
        "max": 0.152,
        "min": -0.050,
        "improved_count": 15,
        "tests_count": 18
      }
    },
    "improvement_by_category": {
      "english": 0.105,
      "mixed": 0.148,
      "chinese": 0.023
    }
  },
  "details": [
    {
      "utterance": "ken_li_01",
      "reference": "My name is Ken Li",
      "category": "english",
      "hotwords_expected": ["KEN LI"],
      "results": {
        "small-bilingual-zh-en": {
          "with_hotwords": "MY NAME IS KEN LI",
          "without_hotwords": "MY NAME IS KEN LE",
          "wer_with": 0.0,
          "wer_without": 0.2,
          "improvement": 0.2
        }
      }
    }
  ]
}
```

## References

- [sherpa-onnx Hotwords Documentation](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
- [Hotwords for Multilingual Models](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html#modeling-unit-is-cjkchar-bpe)
- [Word Error Rate (WER) Explanation](https://en.wikipedia.org/wiki/Word_error_rate)

## Next Steps

1. **Run the impact test** to get baseline metrics
2. **Analyze results** to identify which hotwords help most
3. **Fine-tune boost scores** based on per-model results
4. **Test with microphone input** using `demo_full_integration_file.py`
5. **Integrate into live demo UI** (Step 7) with real-time hotword indication

---

**Last Updated:** December 22, 2025
**Version:** 1.0
