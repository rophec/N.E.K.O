# Qwen3-TTS GGUF / CustomVoice integration

## Scope

N.E.K.O. treats the local Qwen3-TTS GGUF service as a preset-speaker provider.
The selected `voice` is a `speaker_name` exported by a single-speaker SFT or
another CustomVoice checkpoint. Runtime synthesis does not require reference
audio.

Provider defaults:

- provider: `qwen3_tts_gguf`
- endpoint: `ws://127.0.0.1:8091/v1`
- model: `Qwen3-TTS` (must match the sidecar's configured model ID)
- voice: `default` (resolved by the sidecar to the loaded model's default)

## Kept from the experimental implementation

- full-duplex `/v1/audio/speech/stream` transport;
- llama.cpp b9333 Vulkan inference support;
- ONNX Runtime execution-provider support;
- configurable GGUF GPU layers;
- speaker-name mappings and preset speaker selection;
- step and timing diagnostics that do not log private dialogue through a
  logger.

## Deliberately excluded

- Base-model ICL registration and preview flows;
- reference-audio and reference-text routing from N.E.K.O.;
- fixed speaker embeddings, fixed codec codes, x-vector-only modes, and
  decoder-continuation experiments;
- fixed seeds or temperatures as a voice-stability mechanism;
- debug audio exports, cached anchors, generated WAV/JSON/NPY results, local
  models, binaries, and virtual environments.

The current experimental sidecar is not copied into this branch because its
source files combine the production transport with the excluded experiments.
Until a clean sidecar is derived, this branch connects to the already-running
service over the configured endpoint.

## Checkpoint/export contract

After selecting a stable single-speaker SFT checkpoint, export all three
checkpoint-derived components together:

1. embeddings, including the trained `codec_embedding.weight` speaker row;
2. talker;
3. predictor.

The model `config.json` must map `speaker_name` to that trained row. A sidecar
may expose the mapping through its config, but it must not infer a named voice
from the count of non-zero embedding rows.

## Validation order

1. Establish the official 1.7B BF16 VoiceClone baseline in a CUDA PyTorch
   environment.
2. If Base remains unstable, train and evaluate the single-speaker CustomVoice
   checkpoint.
3. Export embeddings, talker, and predictor from the same checkpoint.
4. Validate the GGUF/ONNX sidecar directly with multiple sentences.
5. Select `qwen3_tts_gguf` in N.E.K.O. and set `ttsVoiceId` to the exported
   `speaker_name` (or `default`).
