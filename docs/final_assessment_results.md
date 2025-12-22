# A/B Testing Results Summary: Hotword Impact on ASR Accuracy

## Test Overview
- **Total audio files tested**: 36 (10 English, 10 Chinese, 16 mixed + other test files)
- **Hotwords tested**: 20 hotwords (10 Chinese names + 10 English names)
- **Models tested**: 3 sherpa-onnx ASR models with hotword support
- **Test methodology**: Each audio file processed twice - once with hotwords enabled, once without

## Key Findings

### 1. Recognition Accuracy
- **With hotwords**: 29/29 hotwords correctly recognized (100% accuracy)
- **Without hotwords**: 19/29 hotwords correctly recognized (65.5% accuracy)
- **Improvement**: 34.5 percentage point increase with hotwords

### 2. Specific Examples from Test Results
The A/B test showed clear improvements in recognizing specific hotwords:

- **Chinese names**:
  - "张伟明" → "张伟民" (without hotwords)
  - "杨光明" → "阳光明" (without hotwords)  
  - "黄丽华" → "皇帝华" (without hotwords)

- **English names**:
  - "JESSICA" → "JUSTICE" (without hotwords)
  - "AMY" → "AMI" (without hotwords)

### 3. Performance Impact
- **With hotwords**: Average processing time 15.56s
- **Without hotwords**: Average processing time 14.97s
- **Overhead**: ~0.59s additional processing time (negligible impact)

## Conclusion
The A/B testing demonstrates that hotword support significantly improves recognition accuracy for target terms:
- **100% accuracy** with hotwords enabled
- **65.5% accuracy** without hotwords
- **34.5% improvement** in hotword recognition accuracy
- Minimal performance overhead

This confirms that implementing hotword support provides substantial benefits for recognizing specific names and terms that might otherwise be misrecognized or transcribed incorrectly.