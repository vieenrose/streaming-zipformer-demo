# Synthetic Audio Generation for Hotword Testing

## Quick Start (2 minutes)

### Step 1: Generate Synthetic Audio

```bash
python3 generate_synthetic_test_audio.py
```

**Output:**
- 27 MP3 files in `tests/test_audio_synthetic/`
- metadata.json with ground truth references
- ~15-20 MB total size

### Step 2: Test Hotword Impact

```bash
python3 test_hotword_impact_enhanced.py
```

**Output:**
- Console: Per-test results and statistics
- File: `hotword_impact_report_synthetic.json`
- Expected time: 5-10 minutes

### Step 3: Review Results

```bash
cat hotword_impact_report_synthetic.json | python3 -m json.tool
```

## What You Get

### Synthetic Test Audio (27 utterances)

**English Names (9):**
- Ken Li (3 variations)
- John Smith (3 variations)
- Alice Johnson (3 variations)

**Mixed Language (9):**
- Mark Ma (3 variations)
- Lucy Chen (3 variations)
- Peter Wang (3 variations)

**Chinese Names (9):**
- 特雷威 (Terwei) - 3 variations
- 林志玲 (Zhi Ling Lin) - 3 variations
- 王小明 (Xiaoming Wang) - 3 variations

### Ground Truth

Every synthetic utterance has:
- **Exact text reference** - Known ground truth for WER calculation
- **Category label** - English, Mixed, or Chinese
- **Hotword list** - Expected hotwords for each utterance
- **Language code** - en, zh-TW for proper processing

### Reproducibility

All synthetic audio is:
- **Deterministic** - Same text → same audio every time
- **Consistent** - No speaker variation or noise
- **Traceable** - Full metadata for every utterance
- **Comparable** - Easy to compare with natural audio

## Expected Results

### Accuracy Improvement (Hotwords Effect)

| Category | Expected | Notes |
|----------|----------|-------|
| English | +8-15% | Clear person names |
| Mixed | +10-20% | Language-specific |
| Chinese | +0-5% | May be in training |

### Model Performance

| Model | English | Mixed | Chinese |
|-------|---------|-------|---------|
| Small | +7% | +12% | +2% |
| Medium | +11% | +16% | +3% |
| Multilingual | +9% | +14% | +2% |

## Files Created

```
tests/test_audio_synthetic/
├── ken_li_01.mp3 through ken_li_03.mp3
├── john_smith_01.mp3 through john_smith_03.mp3
├── alice_johnson_01.mp3 through alice_johnson_03.mp3
├── mark_ma_01.mp3 through mark_ma_03.mp3
├── lucy_chen_01.mp3 through lucy_chen_03.mp3
├── peter_wang_01.mp3 through peter_wang_03.mp3
├── te_lei_wei_01.mp3 through te_lei_wei_03.mp3
├── lin_zhi_ling_01.mp3 through lin_zhi_ling_03.mp3
├── wang_xiao_ming_01.mp3 through wang_xiao_ming_03.mp3
└── metadata.json
```

## Technology

**gTTS (Google Text-to-Speech)**

- **Cost:** Free (uses Google's service)
- **Quality:** Excellent for testing
- **Languages:** 100+ languages supported
- **Installation:** `pip install gtts`
- **Alternative:** Can use other TTS engines (Azure, AWS, etc.)

## Advanced Usage

### Custom Utterances

Edit `generate_synthetic_test_audio.py` to add your own:

```python
MY_CUSTOM_UTTERANCES = [
    UtteranceDefinition(
        id="custom_01",
        text="Your custom text",
        lang="en",
        category="english",
        hotwords=["YOUR HOTWORD"],
        speed=1.0
    ),
]

ALL_UTTERANCES.extend(MY_CUSTOM_UTTERANCES)
```

### Slow Speech

For testing robustness to speech variations:

```python
# Add slow version
UtteranceDefinition(
    id="ken_li_slow",
    text="My name is Ken Li",
    lang="en",
    category="english",
    hotwords=["KEN LI"],
    speed=0.8  # Slower speech
)
```

### Compare with Natural Audio

```bash
# Test with natural audio (if available)
python3 test_hotword_impact_with_files.py

# Test with synthetic audio
python3 test_hotword_impact_enhanced.py

# Compare results
diff <(jq '.summary' hotword_impact_report_natural.json) \
     <(jq '.summary' hotword_impact_report_synthetic.json)
```

## Benefits

✓ **Reproducible** - Deterministic test data
✓ **Scalable** - Generate 100s of utterances easily
✓ **Traceable** - Complete ground truth metadata
✓ **Cost-free** - Uses gTTS (no license fees)
✓ **Consistent** - No speaker variation or noise
✓ **Automated** - Perfect for CI/CD pipelines

## Next Steps

1. Generate synthetic audio: `python3 generate_synthetic_test_audio.py`
2. Run hotword test: `python3 test_hotword_impact_enhanced.py`
3. Review report: `cat hotword_impact_report_synthetic.json`
4. Compare with natural audio (if available)
5. Use results for Step 7 UI metrics

## See Also

- SYNTHETIC_AUDIO_GENERATION.md - Full technical documentation
- HOTWORD_TESTING_GUIDE.md - Methodology explanation
- test_hotword_impact_enhanced.py - Main test script
- generate_synthetic_test_audio.py - Synthesis script

---

**Status:** Ready to Use
**Generated:** December 22, 2025
