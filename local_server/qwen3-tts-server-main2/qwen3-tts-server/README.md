# Qwen3-TTS Server

基于 Qwen3-TTS 的 OpenAI 兼容 TTS 服务，支持多音色、声音克隆、流式合成。跨平台支持 Linux / Windows。

## 快速开始

### Linux / macOS

```bash
# 安装依赖
pip install -r requirements.txt

# 下载模型到 model/ 目录
#  └─ model/
#      ├─ qwen3_tts_talker.q5_k.gguf
#      ├─ qwen3_tts_predictor.q8_0.gguf
#      ├─ qwen3_tts_decoder.fp16.onnx
#      ├─ qwen3_tts_codec_encoder.fp16.onnx
#      ├─ qwen3_tts_speaker_encoder.fp16.onnx
#      └─ tokenizer.json

# 启动服务
bash tts_server.sh start
```

### Windows

```bat
# 安装依赖
.venv\python.exe -m pip install -r requirements.txt

# 下载模型到 model\ 目录，启动
start.bat
```

启动后服务运行在 `http://localhost:8091`。

## 技术架构

服务由三个推理阶段组成：

- **Talker**（大师模型）— 根据文本和音色生成语义隐层，决定每帧的语调和语气
- **Predictor**（工匠模型）— 根据语义隐层逐帧预测 16 层声学码本，决定音质细节
- **Decoder**（渲染模型）— 将声学码本解码为 24kHz PCM 波形音频

Talker 和 Predictor 运行在 GPU（llama.cpp GGUF），Decoder 运行在 ONNX Runtime（支持 CUDA / CPU），编码器（Codec Encoder、Speaker Encoder）运行在 ONNX Runtime。

### llama.cpp 编译

`inference/bin/` 目录（已加入 `.gitignore`）需自行编译，编译产物因平台和 CUDA 版本不同而互不适配。

```bash
# Linux (CUDA 12.x)
cd inference/lib && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . --config Release -j$(nproc)

# Windows (CUDA 13.0)
cd inference\lib && mkdir build && cd build
cmake .. -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . --config Release
```

编译产物（`.so` / `.dll` / `.exe`）需放入 `inference/bin/`。

### ONNX Runtime GPU 加速

如需 GPU 加速 Decoder，需安装 `onnxruntime-gpu` 并配置 cuDNN 9.x：

```bash
pip install onnxruntime-gpu nvidia-cudnn-cu13 nvidia-cublas nvidia-cuda-nvrtc
```

Windows 下需将 nvidia pip 包的 `bin/` 目录加入 PATH（`start.bat` 已处理）。

## 性能参数

以下数据基于 GB10 GPU 实测（24kHz 输出）。性能因硬件和 CUDA 版本不同会有差异。

| 项目 | 数值 |
|------|------|
| 模型参数量 | 1.7B |
| 音频采样率 | 24kHz |
| 支持并发 | 1 流（串行） |

**离线推理（非流式 WAV 一次性返回）：**

| 文本长度 | 生成音频时长 | TTFB / 总耗时 | 实时率 (RTF) |
|----------|-------------|---------------|-------------|
| 5 字     | 1.6s        | 0.58s         | 0.36x       |
| 40 字    | 7.2s        | 2.25s         | 0.31x       |
| 100 字   | 15.4s       | 4.63s         | 0.30x       |
| 200 字   | 24.0s       | 6.45s         | 0.27x       |

> 非流式模式下 TTFB 等于总耗时（全部音频生成完毕后一次性返回），RTF 随文本变长降低——固定开销被摊薄，长文本效率更高。

**流式推理（PCM 逐 chunk 返回）：**

| 文本长度 | 生成音频时长 | 首 chunk TTFB | 总耗时 | 实时率 (RTF) | Chunk 数 |
|----------|-------------|--------------|-------|-------------|---------|
| 5 字     | 1.4s        | 0.28s        | 0.53s | 0.39x       | 5       |
| 40 字    | 8.2s        | 0.28s        | 2.53s | 0.31x       | 34      |
| 100 字   | 16.6s       | 0.27s        | 4.70s | 0.28x       | 69      |
| 200 字   | 24.0s       | 0.26s        | 6.64s | 0.28x       | 100     |

> 流式模式下首 chunk TTFB 约 **0.26~0.28s**，与文本长度无关——这是 Talker 首帧推理的固定开销。后续 chunk 持续到达，总耗时接近离线模式。

**显存占用：**

| 状态 | 主进程 (Talker+API) | 子进程 (Decoder) | 合计 |
|------|-------------------|-----------------|------|
| 闲置 | 1258 MiB          | 718 MiB         | ~2.0 GB |
| 推理峰值 | 1549 MiB          | 718 MiB         | ~2.2 GB |

> 推理过程中主进程显存陡增约 300 MiB（加载模型权重并执行推理），推理结束后释放回落至基线。**部署建议：预留 3GB+ 显存以确保稳定运行。**

## 功能

### 多音色合成

服务内置多种预设音色，可通过 `/v1/audio/voices` 查看可用列表。

```bash
# 查看可用音色
curl http://localhost:8091/v1/audio/voices

# 使用指定音色合成语音
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-TTS","input":"你好世界","voice":"Hutao","response_format":"wav"}' \
  --output speech.wav
```

### 声音克隆

提供参考音频和对应文本即可克隆音色。支持本地文件路径、HTTP URL、Base64 Data URL。

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-TTS","input":"要合成的文本","ref_audio":"file:///path/to/reference.wav","ref_text":"参考音频的原文","response_format":"wav"}' \
  --output clone.wav
```

克隆音色会被缓存（`voice_cache_dir`），后续相同 ref_audio + ref_text 命中缓存时跳过编码器提取，显著降低延迟。

### 流式合成

适合低延迟播放场景，支持 WAV 和 PCM 格式。

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-TTS","input":"你好世界","voice":"Hutao","response_format":"pcm","stream":true}' \
  --output speech.pcm
```

### WebSocket 流式合成

适用于实时对话等需要持续交互的场景，支持分句流式调度。

```python
import asyncio, json, websockets

async def tts():
    async with websockets.connect("ws://localhost:8091/v1/audio/speech/stream") as ws:
        # 普通合成
        await ws.send(json.dumps({"type": "session.config", "voice": "Hutao", "language": "zh"}))
        
        # 或声音克隆
        # await ws.send(json.dumps({"type": "session.config", "language": "zh",
        #     "ref_audio": "data:audio/wav;base64,...", "ref_text": "参考文本"}))
        
        await ws.send(json.dumps({"type": "input.text", "text": "你好世界。"}))
        await ws.send(json.dumps({"type": "input.done"}))
        with open("output.pcm", "wb") as f:
            async for msg in ws:
                if isinstance(msg, bytes):
                    f.write(msg)
                elif json.loads(msg).get("type") == "session.done":
                    break

asyncio.run(tts())
```

> **性能优化**：克隆模式下，分句间复用音色锚点（`reset(clear_voice=False)`），避免每句重复 `set_voice` 开销（约 1~2s），分句 TTFB 接近普通合成水平。

### API Key 鉴权

在配置文件中设置 `api_key` 后，所有请求需携带鉴权头：

```bash
curl -H "Authorization: Bearer sk-你的密钥" http://localhost:8091/v1/audio/voices
```

健康检查 `/health` 不受鉴权限制。

## 配置

配置文件为项目目录下的 `config.yaml`。参考 `config.yaml.example` 获取完整配置项。

也支持通过环境变量配置（前缀 `QWEN3_TTS_`）：

```bash
# Linux
QWEN3_TTS_MODEL_DIR="/path/to/model" QWEN3_TTS_PORT=9090 bash tts_server.sh start

# Windows
set QWEN3_TTS_MODEL_DIR=D:\models\Qwen3-TTS
set QWEN3_TTS_PORT=9090
start.bat
```

关键配置项：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `QWEN3_TTS_MODEL_DIR` | `model` | 模型目录（相对路径） |
| `QWEN3_TTS_PORT` | 8091 | 监听端口 |
| `QWEN3_TTS_ONNX_PROVIDER` | CUDA | ONNX EP（CUDA / CPU） |
| `QWEN3_TTS_API_KEY` | (空) | API 鉴权密钥 |
| `QWEN3_TTS_MAX_CONCURRENT_STREAMS` | 1 | 最大并发流数 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/audio/speech` | 语音合成（支持流式和非流式） |
| WS | `/v1/audio/speech/stream` | WebSocket 流式合成 |
| GET | `/v1/audio/voices` | 可用音色列表 |
| GET | `/v1/models` | 模型信息 |
| GET | `/health` | 健康检查 |

## 输出格式

支持 `wav`、`pcm`、`mp3`、`opus`、`aac`、`flac`。

## 环境要求

- NVIDIA GPU（推荐 8GB+ 显存）
- CUDA 12.0+ / 13.0
- cuDNN 9.x（需 `onnxruntime-gpu` 时）
- Python 3.12+

## 服务端日志

推理日志格式如下，包含 LLM 推理和 ONNX Decode 的分项耗时：

```
[TTS] text='今天阳光明媚...'  steps=50  audio=3.92s  inference=1.03s(0.96s+0.07s)  RTF=0.26x  tok/s=48.5  first_chunk=0.070s  first_audio=0.140s
```

| 字段 | 说明 |
|---|---|
| `inference` | 总耗时(LLM + Decode)，括号内分项 |
| `RTF` | (LLM + Decode) / 音频时长 |
| `first_chunk` | 推理开始→首组码本生成 |
| `first_audio` | 推理开始→首音生成（= first_chunk + 首包解码） |