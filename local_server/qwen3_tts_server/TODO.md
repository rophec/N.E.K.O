# Qwen3-TTS Sidecar TODO

- [ ] 优化声音克隆的 Encoder 与缓存路径（暂不修改）
  - 当前 Codec Encoder 和 Speaker Encoder 固定使用 `CPUExecutionProvider`。
  - 只影响声音克隆的参考音频预处理与首音延迟，不影响预设音色，也不影响生成阶段的 RTF。
  - 当前 `data:` 参考音频每次解析为新的随机临时文件，而 VoiceCache 使用“临时文件路径 + ref_text”作为缓存键，导致相同参考音频跨 WebSocket 会话无法命中缓存，并重复执行 CPU Encoder。
  - 优先方案：缓存键改为“音频内容哈希 + ref_text”，补充跨连接缓存命中测试，并清理不再需要的临时音频文件。
  - 后续方案：在缓存修复后，再对 CPU、DirectML、CUDA Encoder 做首音延迟和显存占用 A/B 测试；没有数据前不切换默认 EP。
  - 验收：同一参考音频首次请求执行 Encoder，后续连接直接加载 voice anchor；预设音色路径行为不变。
