# N.E.K.O. Qwen3-TTS GGUF Local Server

这是 N.E.K.O. 官方维护的 Qwen3-TTS GGUF 本地 sidecar。它负责提供 N.E.K.O. 使用的 HTTP/WebSocket API、运行环境和日志；它不是 HaujetZhao/Qwen3-TTS-GGUF 上游仓库的一部分。

## 边界

- 上游 `HaujetZhao/Qwen3-TTS-GGUF`：负责模型转换、导出和原始推理工具。
- 本目录：负责 N.E.K.O. 支持的定制推理运行层和 `/v1/audio/speech/stream` 协议。
- `model/`：只存放导出或下载得到的模型资产，不提交到 Git。
- `inference/bin/`：只存放匹配版本的 llama.cpp 动态库，不提交到 Git。

不要再把本目录的 `server/` 或 `inference/` 覆盖到上游仓库中。N.E.K.O. 只保证 sidecar 内成对版本的 `inference/` 与 `server/` 兼容。

## 目录准备

模型目录至少应包含：

```text
model/Qwen3-TTS-12Hz-1.7B-Base-GGUF/
  tokenizer.json
  qwen3_tts_talker.*.gguf
  qwen3_tts_predictor.*.gguf
  qwen3_tts_codec_encoder.*.onnx
  qwen3_tts_decoder.*.onnx
  qwen3_tts_speaker_encoder.*.onnx
  embeddings/
```

将 llama.cpp b9333 对应平台的动态库放入 `inference/bin/`。Windows 至少需要 `llama.dll`、`ggml.dll`、`ggml-base.dll` 以及所选 CPU/CUDA 后端依赖。

也可以不复制资产，使用环境变量指向现有目录：

```powershell
$env:QWEN3_TTS_MODEL_DIR="F:\path\to\Qwen3-TTS-12Hz-1.7B-Base-GGUF"
$env:QWEN3_TTS_LLAMA_BIN_DIR="F:\path\to\llama-b9333-bin"
```

## 启动

在本目录执行：

```powershell
Copy-Item config.yaml.example config.yaml
uv sync
.\start.ps1
```

sidecar 只运行官方 `1.7B-Base-GGUF`。Base 没有内置预制音色，用户先在 N.E.K.O.
Voice Clone 页面注册参考音频，随后所有发声都由同一个 Base 模型完成。

或直接使用环境变量：

```powershell
$env:QWEN3_TTS_MODEL_DIR="F:\path\to\Qwen3-TTS-12Hz-1.7B-Base-GGUF"
$env:QWEN3_TTS_LLAMA_BIN_DIR="F:\path\to\llama-b9333-bin"
$env:QWEN3_TTS_PORT="8091"
uv run python -m server.main
```

启动成功后：

- 健康检查：`http://127.0.0.1:8091/health`
- 音色目录：`http://127.0.0.1:8091/v1/audio/voices`
- WebSocket：`ws://127.0.0.1:8091/v1/audio/speech/stream`

## N.E.K.O. 配置

在“自定义 API > TTS 模型配置”中填写：

```text
服务商：Qwen3-TTS-GGUF
API URL：ws://127.0.0.1:8091/v1
模型 ID：Qwen3-TTS-Base
API Key：留空（除非 sidecar 配置了 api_key）
Voice ID：default（注册克隆音色后，为角色选择生成的 Voice ID）
```

不要将这个地址填入“实时模型配置”；那一栏是 Realtime 对话模型，不是 TTS。

## 克隆并发声

1. 启动 sidecar，确认日志显示 `模型类型: base`。
2. 在 Voice Clone 页面选择 `Qwen3-TTS-GGUF（本地克隆）`。
3. 上传参考音频，并逐字填写参考音频原文。
4. 注册音色并将生成的 Voice ID 绑定到角色。
5. 后续 TTS 会把保存的参考音频和原文发送到同一个 `8091` Base 服务。

未绑定克隆音色时不会调用 Base 合成，因为 Base 没有可直接使用的内置音色。

## 协议

客户端依次发送：

1. `session.config`（克隆模式包含稳定的 `voice` ID、前端 `voice_name`、
   `ref_audio` 和 `ref_text`）
2. 一个或多个 `input.text`
3. `input.done`

服务端返回 `audio.start`、二进制 24 kHz PCM、`audio.done` 和 `session.done`。该协议属于 N.E.K.O. sidecar，不代表上游仓库原生提供同一服务接口。

克隆音色在首次验证、试听、选择或正式合成时登记到 sidecar 的本地音色目录。
`GET /v1/audio/voices` 会同时返回模型预制音色和已登记的克隆音色，例如：

```json
{
  "voices": [
    {
      "voice_id": "qwen3-tts-gguf-clone-ch-...",
      "name": "我的音色",
      "type": "clone",
      "audio_key": "d885c4a1f897..."
    }
  ]
}
```
