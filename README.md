# Live DEMO: Compare Multiple sherpa-onnx Streaming STT Engines

A Docker Compose demo that compares 3 sherpa-onnx ASR models side-by-side with real-time audio analysis and transcription.

## Quick Start

```bash
# Build and run with Docker Compose
docker compose up

# Or run directly on host (requires Python dependencies)
python demo_full_integration.py
```

## Features

1. **Auto Device Detection** - Scans all audio input devices, uses VAD to find active microphone
2. **Multi-Model Comparison** - Runs 3 ASR engines in parallel for direct comparison
3. **Real-time UI** - Fixed-grid text UI with VAD/RMS bar charts and streaming transcripts
4. **MP3 Recording** - Background recording to MP3 during session
5. **Hotword Support** - Multi-token hotwords with zh-TW/EN mixed language support
6. **Docker Ready** - Containerized with ALSA device passthrough

## Objective

The demo performs the following workflow:

1. **Device Detection** - Enumerate audio input devices, scan for active speech via VAD
2. **Model Loading** - Preload 3 ASR models (small-bilingual, medium-bilingual, multilingual)
3. **Real-time Processing**
   - Record audio to MP3 in background
   - Display audio analysis in text UI (device status, VAD/RMS bars, transcripts)
   - Compare transcription quality across models
   - Detect hotwords (English, zh-TW, mixed)

## Dependencies

### Python Modules

```
sherpa-onnx>=1.12.20
ten-vad (built from source, ONNX backend)
sounddevice>=0.5.3
soundfile>=0.13.1
numpy>=2.3.5
scipy>=1.13.0
opencc-python-reimplemented>=0.1.7
wcwidth>=0.2.14
```

### ASR Models

| Model | Description |
|-------|-------------|
| [csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10) | Multilingual (11 languages) |
| [sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile.tar.bz2) | Small bilingual (zh-en) |
| [sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-mobile](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-mobile.tar.bz2) | Medium bilingual (zh-en) |

## Hotwords

Multi-token hotwords with no OOV tokens:

| Type | Examples |
|------|----------|
| Pure English | Ken Li, Mark Ma, LILY, KEVIN, DAVID |
| Pure zh-TW | 張偉明, 李建國, 王曉紅, 陳志強, 劉芳芳 |
| Mixed (zh-TW/EN) | Mark Ma, Ken Li |

Hotwords are converted from zh-TW to zh-CN at input, transcripts converted from zh-CN to zh-TW at output.

## Architecture

### Phase 1: Device Scanning (VAD-based Auto-detection)

At startup, the system enumerates all available audio input devices via `sounddevice.query_devices()`, filtering out ALSA plugin devices (sysdefault, lavrate, samplerate, speexrate, etc.) that are known to cause issues.

For each remaining device, the scanner opens an InputStream at the device's native sample rate (typically 44.1kHz or 48kHz), reads a 100ms chunk as float32, resamples to 16kHz using scipy's polyphase resampler, and applies AGC normalization. The processed audio is then fed to the ten-vad ONNX model (requiring 16kHz int16 input) which outputs a speech probability score between 0.0 and 1.0.

The scanner runs in cycles, comparing VAD scores across all devices. When a device exceeds the threshold (default 0.75), that device is selected and the scanning phase exits. If no device reaches the threshold after a configurable number of cycles, the device with the highest score is selected.

**Output**: `device_id`, `sample_rate`, `channels` for the selected device.

### Phase 2: Real-time Audio Processing Pipeline

Once a device is selected, the system opens a dedicated InputStream and begins continuous processing. Audio is captured in 100ms chunks at the device's native sample rate as float32 samples.

**Preprocessing**: Each chunk is resampled to 16kHz (required by both ten-vad and sherpa-onnx models) using `scipy.signal.resample_poly`, then optionally passed through a slow AGC to normalize RMS levels. An RMS meter continuously measures audio levels for UI display.

**Parallel Processing**: The preprocessed audio stream forks into three parallel paths:

1. **MP3 Recorder** (background thread): Encodes audio to MP3 via ffmpeg and writes to `/recordings/*.mp3`

2. **VAD Detector** (ten-vad): Converts float32 to int16, feeds to ten-vad ONNX model, outputs speech probability (0.0-1.0) for the VAD bar chart in the UI

3. **ASR Engine Pool**: Three sherpa-onnx models run in parallel (small-bilingual, medium-bilingual, multilingual), each with 4 CPU threads. Audio is fed to all models simultaneously using `accept_waveform()`. Hotwords are first converted from zh-TW to zh-CN (via opencc) for model input. Decoding uses `modified_beam_search` with hotword boost set to 1.5. Transcript output in zh-CN is converted back to zh-TW for display.

**Text UI**: Three-zone fixed-grid display updated continuously:
- Zone 1: Device status (name, ID, sample rate, channels, data type)
- Zone 2: VAD and RMS bar charts (dynamic, 82-column width)
- Zone 3: Left-truncated streaming transcripts from each model (most recent text only)

### Data Flow Summary

| Stage | Input | Output | Component |
|-------|-------|--------|-----------|
| Capture | ALSA `/dev/snd` | float32, native rate | sounddevice |
| Resample | float32, native | float32, 16kHz | scipy.signal.resample_poly |
| VAD | float32, 16kHz | speech prob (0-1) | ten-vad ONNX |
| ASR | float32, 16kHz | zh-CN text | sherpa-onnx (3 models) |
| Output | zh-CN text | zh-TW text | opencc |

### Key Design Decisions

* **ten-vad over silero-vad**: More accurate for 16kHz speech, ONNX backend
* **Modified beam search**: Better accuracy than greedy for hotword boosting
* **Parallel ASR models**: Direct comparison, each with 4 CPU threads
* **zh-TW ↔ zh-CN conversion**: Hotwords (input) and transcripts (output)
* **Docker resource limits**: 6 CPUs, 4GB RAM for 12 ASR threads total

## Docker Configuration

```yaml
# Resource limits for 3 ASR models × 4 threads = 12 threads
deploy:
  resources:
    limits:
      cpus: '6.0'
      memory: 4G

# ALSA device passthrough
devices:
  - /dev/snd:/dev/snd
network_mode: host
```

## References

* [sherpa-onnx documentation](https://k2-fsa.github.io/sherpa/onnx/index.html)
* [sherpa-onnx ASR models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
* [Hotwords (Contextual biasing)](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
* [Hotwords for Multilingual Models](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html#modeling-unit-is-cjkchar-bpe)
* [ten-vad](https://github.com/TEN-framework/ten-vad)

## Implementation Plan

| Step | Description | Status |
|------|-------------|--------|
| 1 | Verify baseline microphone transcription on host | ✓ |
| 2 | Download and validate all ASR models | ✓ |
| 3 | Implement OOV-free hotwords with BPE vocab export | ✓ |
| 4 | Auto device detection via VAD scanning | ✓ |
| 5 | Background MP3 recording + audio streaming | ✓ |
| 6 | Multi-engine ASR orchestration with hotword support | ✓ |
| 7 | Docker-compatible text UI with 3-zone layout | ✓ |
| 8 | zh-TW ↔ zh-CN conversion for hotwords and transcripts | ✓ |
| 9 | Port to Docker Compose with ALSA passthrough | ✓ |
