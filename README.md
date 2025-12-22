# Live DEMO to Compare Multiple [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) Streaming STT engines in Docker Compose

## Objective

A live demo built in docker compose, that

1. First, enumerate all available audio input devices on the systems at start. Then in a persistant loop, 
. find the device with the highest VAD probability in current scam cycle, exits immediately when the top device's VAD > 0.5. 
2. Seconds, pre-downloads all ASR models listed in *zh-CN/en Multilingual ASR models* section
3. in the next, 
3.1 records raw audio samples from the microphone in MP3 format in background until the end of session
3.2 provide real-time monitor that displays audio analysis in *nice text-based UI* with fixed-grid layout with three zones (see 3.0, 3.1, 3.2)
3.2.0 header showing system status: specification on the audio capturing device like device name / id, raw sample rate, data type, etc.
3.2.1 2 rows of dynamic bar charts for VAD and RMS
3.2.2 N rows of left-truncated, per-engine streaming transcripts, showing only the most recent text, for comparison.
3.2.2.1 multi-token hotword testing: design a list of multi-token hotwords (make sure there is no OOV tokens in each hotword for all ASR models' tokenizers) containing at least one example for one of the types described below then show it for user:

* a. person names in pure English (e.g. Ken Li),
* b. person names in pure zh-TW (e.g. 特雷危),
* c. person names in zh-TW/en mixing language (e.g. Mark Ma).

P.S. Redesign new hotwords according to models' tokenziers to avoid OOV tokens

3.2.2.2 adapt hotword list to models' tokenizers, adjust decoding method, Fine-tune or set appropriate boost score to make hotwords realy work. (if needed, make individuel experiments to learn how it works and prove it works as expected)
3.2.2.3 an additional requirement to enable hotword for multilingual models, check *Hotwords for Multilingual Models* in references section, bpe.vocab file is required.
3.2.2.4 display live transcripts in zh-TW instead of in zh-CN  

## Dependencies

### Python modules

* [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) CPU-only version
* [ten-vad](https://github.com/TEN-framework/ten-vad) or [silero-vad-lite](https://github.com/daanzu/py-silero-vad-lite)
* sounddevice
* opencc-python-reimplemented

P.S. silero-vad-lite supports both 8kHz and 16kHz while ten-vad requires 16kHz input.

### zh-CN/en Multilingual ASR models

* [csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10)
* [sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16-mobile.tar.bz2)
* [sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-mobile](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-mobile.tar.bz2)

## References

* [sherpa-onnx documentation](https://k2-fsa.github.io/sherpa/onnx/index.html)
* [sherpa-onnx ASR models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
* [Hotwords (Contextual biasing)](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
* [Hotwords for Multilingual Models](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html#modeling-unit-is-cjkchar-bpe)
* [speech-recognition-from-microphone-with-endpoint-detection.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py)
* [online_recognizer.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/online_recognizer.py)
* [sherpa-onnx/python/csrc](https://github.com/k2-fsa/sherpa-onnx/tree/master/sherpa-onnx/python/csrc)

## Plan & Checkpoints

1. take the existing speech-recognition-from-microphone-with-endpoint-detection.py script, run it on the host system ensuring it functions correctly by validating that it can accurately transcribe speech from the microphone; if any issues arise during microphone-based transcription, switch to using soundfile to test with a known audio file that has established ground truth for verification.
2. download and extract all required ASR models (multilingual and bilingual models), then test each of them with the microphone-based speech recognition script from step 1 to validate they function properly with real audio input
3. redesign hotwords to satisfy 3.2.2. for models which are not delivred with bpe.vocab, make sure to export their own one with official [tool]](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/export_bpe_vocab.py). then make hotword work with hotwords file on script obtained in step 1 then in the next with stream-level dynamic hotword list. 
4. enumerate all available audio input devices on the system and automatically detect the active microphone through Voice Activity Detection (VAD) in real-time scanning cycles; exit when speech is detected above a configurable threshold (default 0.75) and display formatted device metrics (VAD probability, RMS levels, cycle timing) in a docker-compatible text-based UI with unicode table formatting and progress bar indicators.
5. create MP3 recorder
6. add UIs
7. port into docker compose
