# TTS 重构落地记录

> 状态：已合并 PR 的实现梳理。  
> 口径：本文按 GitHub PR 粒度阅读 `#1818/#1824/#1830/#1842/#1848/#1850/#1851/#1852/#1853` 的标题、说明、文件列表与当前代码状态，记录 TTS 重构实际落地内容。  
> 相关蓝图：[`docs/design/tts-voice-source-unification.md`](./tts-voice-source-unification.md)

## 1. 总览

这轮 TTS 重构不是单点功能，而是一组连续 PR，把原本纠缠在 `voice_id`、`get_tts_worker()` 内联分支、native voice registry、voice clone 存储中的逻辑拆成更明确的几层：

1. **TTS provider/backend**：谁来合成语音，例如 `gptsovits`、`vllm_omni`、`mimo`、`elevenlabs`。
2. **voice source**：声音身份从哪里来，例如 `preset`、`clone`、`design`。
3. **voice ref**：在某个 provider 内部的音色标识，例如 `Milo`、某个 clone id、某个 ElevenLabs design voice id。
4. **dispatch registry**：provider 如何被选中、如何解析 worker、如何声明 UI/probe/音色目录能力。

旧世界里，`voice_id` 经常同时承担三种含义：

- 声音身份：到底是哪一个音色。
- 后端路由：例如靠 `gsv:`、`eleven:` 前缀决定走哪个 provider。
- 来源类型：预制、克隆、描述生成没有统一字段，只能靠前缀或 voice meta 猜。

新世界把这些信息拆开，核心数据模型变成：

```jsonc
{
  "source": "preset | clone | design",
  "provider": "mimo | elevenlabs | gptsovits | vllm_omni | ...",
  "ref": "provider 内部音色标识",
  "config": {}
}
```

为了兼容已有用户数据，运行时仍保留 legacy string shim：读路径可以把对象还原成旧 `voice_id` 字符串，写路径在安全时惰性迁移。

## 2. PR 时间线

| PR | 主题 | 主要作用 |
| --- | --- | --- |
| [#1818](https://github.com/Project-N-E-K-O/N.E.K.O/pull/1818) | 统一 TTS provider 注册表 + 声音来源能力模型（后端） | 建立 `utils/tts_provider_registry.py`，把 provider dispatch 从内联 if/else 收敛到注册表。 |
| [#1824](https://github.com/Project-N-E-K-O/N.E.K.O/pull/1824) | 将 `tts_client.py` 拆分为 package | 删除大单文件，改成 `main_logic/tts_client/` package，按 provider 拆 worker。 |
| [#1830](https://github.com/Project-N-E-K-O/N.E.K.O/pull/1830) | source-first 选声器 + GPT-SoVITS 迁下拉 + 结构化 voice 地基 | 前端按 provider/source 分组；新增 `utils/voice_config.py`；GPT-SoVITS 开始迁到 TTS 下拉。 |
| [#1842](https://github.com/Project-N-E-K-O/N.E.K.O/pull/1842) | `voice_id` 惰性迁移到 `{source,provider,ref}` | 写侧接入结构化对象，读侧容忍 string/dict 两形态。 |
| [#1848](https://github.com/Project-N-E-K-O/N.E.K.O/pull/1848) | MiMo preset catalog 迁到 hosted provider | MiMo 从 native registry 摘除，预设音色目录挂到 TTS provider 注册表。 |
| [#1850](https://github.com/Project-N-E-K-O/N.E.K.O/pull/1850) | GPT-SoVITS 启用收口到 `ttsModelProvider` | `ttsModelProvider == "gptsovits"` 成为单一真相，退役旧 `gptsovitsEnabled` 写路径。 |
| [#1851](https://github.com/Project-N-E-K-O/N.E.K.O/pull/1851) | ElevenLabs voice design + MiMo voiceclone | 增加 ElevenLabs design endpoint；实现 MiMo clone enrollment 与 dispatch。 |
| [#1852](https://github.com/Project-N-E-K-O/N.E.K.O/pull/1852) | GPT-SoVITS 教程文档按钮 | API 设置页选中 GPT-SoVITS 时显示教程链接，补齐 8 locale。 |
| [#1853](https://github.com/Project-N-E-K-O/N.E.K.O/pull/1853) | hosted preset ownership 元数据修复 | 保存 MiMo preset 时正确标注 `{source:preset, provider:mimo}`。 |

## 3. 当前架构落点

### 3.1 `main_logic/tts_client/`

`main_logic/tts_client.py` 已被替换为 package：

```text
main_logic/tts_client/
  __init__.py
  _infra.py
  _registry_meta.py
  _telemetry.py
  workers/
    cogtts.py
    cosyvoice.py
    dummy.py
    elevenlabs.py
    gemini.py
    gptsovits.py
    grok.py
    local_cosyvoice.py
    mimo.py
    minimax.py
    openai.py
    qwen.py
    step.py
    vllm_omni.py
```

拆分原则：

- `__init__.py` 保留 facade 和旧 import surface，维持 `from main_logic.tts_client import X` 兼容。
- `get_tts_worker()`、`_get_voice_meta()`、`_grok_voice_id_is_xai_custom()` 仍在 `__init__.py`，保证测试 monkeypatch 命中真实 dispatch 命名空间。
- `_infra.py` 放队列、重采样、分句、worker 主循环等共享设施。
- `_registry_meta.py` 放 provider meta registry。
- `workers/*.py` 放 provider worker 和各自的 `_*_is_selected` / `_*_resolve` adapter。

这次拆分设计目标是零行为变化，真正的行为变化主要来自前后相邻的注册表、voice config、MiMo/GPT-SoVITS/ElevenLabs PR。

### 3.2 `utils/tts_provider_registry.py`

注册表是新 dispatch 中心。一个 `TTSProvider` 负责声明：

- `key`：provider key，例如 `mimo`。
- `kind`：`local` 或 `hosted`。
- `priority`：dispatch 顺序，数字越小越先选中。
- `capabilities`：支持哪些 source，例如 `preset`、`clone`、`design`。
- `is_selected(ctx)`：当前配置/音色是否命中该 provider。
- `resolve(ctx)`：返回 `(worker, api_key_override, provider_key)`。
- UI/probe 元数据：默认 URL/model/voice、字段名、endpoint 是否可编辑、probe 策略。
- `preset_catalog`：provider 自带预设音色目录，例如 MiMo。

当前注册表 dispatch 优先级大致为：

| priority | provider | 说明 |
| --- | --- | --- |
| 10 | `gptsovits` | local/self-hosted，clone/reference audio 类路径。 |
| 20 | `vllm_omni` | local/self-hosted，preset voice id。 |
| 30 | `minimax` | hosted clone。 |
| 40 | `elevenlabs` | hosted clone/design。 |
| 50 | `cosyvoice` | hosted clone。 |
| 60 | `mimo` | hosted preset/clone。 |

注意：Gemini/Step/Grok/free 等 core-native preset 仍主要通过 `native_voice_registry` 和 `get_native_tts_worker()` 处理；统一注册表已经承载 local/hosted 特异 provider，但 native 还没有完全折进来。

### 3.3 `DispatchContext`

`DispatchContext` 统一了两种选中机制：

- **配置选中**：例如 `ttsModelProvider == "gptsovits"` 或 `vllm_omni`，只依赖 `core_config` / `cm`。
- **音色元数据选中**：例如用户选择了某个 clone voice，通过 `voice_meta.provider` 命中 `minimax`、`elevenlabs`、`cosyvoice`、`mimo`。

`voice_meta` 是惰性加载：配置选中的 provider 可以在任何 voice storage 查询之前短路，避免显式 provider 被 clone meta 查询副作用影响。

## 4. PR 细节

### 4.1 #1818：provider 注册表地基

核心文件：

- `utils/tts_provider_registry.py`
- `main_logic/tts_client.py`（当时还是单文件）
- `main_routers/config_router.py`
- `static/js/api_key_settings.js`
- `utils/gptsovits_config.py`
- `utils/mimo_tts_voices.py`

落地内容：

- 新增 `TTSProvider` 注册表，把 provider 的选中、resolve、UI/probe 元数据集中声明。
- `get_tts_worker()` 原先的 GPT-SoVITS/vLLM/MiniMax/ElevenLabs/CosyVoice/MiMo 特判开始折入注册表。
- 按 priority 保留旧 dispatch 顺序。
- `DispatchContext` 引入惰性 `voice_meta`。
- `api_key_settings.js` 开始从 `/api/config/api_providers` 的 `tts_providers` 元数据读取 TTS provider 信息，减少前端硬编码。
- GPT-SoVITS 放宽为允许远程 URL，不再只允许 localhost。

行为变化：

- dispatch 目标上应与旧逻辑等价。
- 有意变化是 GPT-SoVITS 远程 URL 可用，属于维护者接受的行为变化。

风险点：

- dispatch 顺序不能乱。
- `voice_meta` 不能在显式 provider 命中前被提前加载。
- `tts_dropdown_only=false` 的 provider 不能被错误从普通 LLM 下拉中隐藏，例如 MiniMax。

### 4.2 #1824：`tts_client` 拆 package

核心文件：

- 删除：`main_logic/tts_client.py`
- 新增：`main_logic/tts_client/__init__.py`
- 新增：`main_logic/tts_client/_infra.py`
- 新增：`main_logic/tts_client/_registry_meta.py`
- 新增：`main_logic/tts_client/_telemetry.py`
- 新增：`main_logic/tts_client/workers/*.py`

落地内容：

- 5235 行级别的大单文件拆成 package。
- 每个 provider worker 单独成文件。
- facade 维持旧 import 兼容，测试和生产 import 不需要大面积改。
- 注册表 bootstrap 仍由 `__init__.py` 导入 worker 后执行。

重要约束：

- `get_tts_worker()` 实际 lookup 的 `get_config_manager`、`_get_voice_meta`、`websockets` 等符号仍在 `main_logic.tts_client` 命名空间，保留 monkeypatch 行为。
- worker `__module__` 变化不应影响运行，因为生产和测试走 Thread 而非 Process pickling。

### 4.3 #1830：source-first UI 与结构化 voice 地基

核心文件：

- `static/js/api_key_settings.js`
- `static/js/character_card_manager.js`
- `templates/api_key_settings.html`
- 8 个 locale JSON
- `utils/voice_config.py`
- `utils/config_manager.py`
- `main_logic/tts_client/workers/gptsovits.py`
- `tests/unit/test_gptsovits_dropdown_routing.py`
- `tests/unit/test_voice_config.py`

落地内容：

1. API 设置页修复三类 provider 元数据问题：
   - `/api_providers` 的 `tts_providers` 加载失败时，用结构信号兜底判断 TTS-only provider。
   - `editable_endpoint` provider 保存时不被 provider profile URL 覆盖。
   - 注册表里存在但不在 `assist_api_providers` 的 provider 也能进入 TTS 下拉，例如 GPT-SoVITS。

2. GPT-SoVITS 迁到 TTS 下拉：
   - 后端 `_gptsovits_is_selected` 迁移期同时认 `ttsModelProvider == "gptsovits"` 和旧 `GPTSOVITS_ENABLED`。
   - 前端删除独立启用开关，把 GSV 字段移进 TTS 配置面板。
   - 写路径不再产生 `__gptsovits_disabled__|` 占位符，读路径仍兼容。

3. 角色音色 source-first 分组：
   - 克隆音色按 provider 分组，例如 `MiniMax · 克隆`、`ElevenLabs · 克隆`。
   - free/native/GSV 分到各自 provider/source group。
   - 选中值仍写 legacy `voice_id`，为后续惰性迁移做准备。

4. 新增 `utils/voice_config.py`：
   - `VoiceConfig`
   - `parse_legacy_voice_id`
   - `normalize_voice_id`
   - `to_legacy_voice_id`
   - 本 PR 只是地基，尚未全面接入 at-rest 写路径。

### 4.4 #1842：`voice_id` at-rest 惰性迁移

核心文件：

- `app/main_server.py`
- `config/__init__.py`
- `main_logic/core.py`
- `main_routers/characters_router.py`
- `utils/config_manager.py`
- `utils/voice_config.py`
- `tests/unit/test_voice_lazy_migration.py`

落地内容：

- 用户设置音色时，`ConfigManager.voice_id_to_storage_value()` 尝试把 legacy string 转成结构对象存储。
- 读路径通过 `read_legacy_voice_id()` 同时容忍旧 string 和新 dict。
- 不做 bulk sweep；只在用户触发设置时迁移当前条目。
- 空值仍存空串。
- 内部 preset/free/default/YUI/cleanup 等路径仍可以写 string，避免扩大行为面。

核心不变式：

```text
store -> read == 原 legacy voice key
```

也就是说，迁移不能改变 voice library 绑定 key。若 `to_legacy(vc)` 还原后不等于原串，写侧会保留 legacy string，不迁移成对象。

主要风险与防护：

- 某个读点漏用 `read_legacy_voice_id()`，可能对 dict 调 `.strip()` 崩溃。
- 顶层 legacy `voice_id` 若已经是对象，迁移到 `_reserved` 时不能被清理丢失。
- schema 需要接受 `voice_id` 为 `(str, dict)`。

### 4.5 #1848：MiMo preset catalog 从 native 摘除

核心文件：

- `utils/tts_provider_registry.py`
- `utils/mimo_tts_voices.py`
- `utils/native_voice_registry.py`
- `utils/config_manager.py`
- `main_routers/characters_router.py`
- `main_logic/tts_client/__init__.py`
- `tests/unit/test_mimo_tts_voices.py`

落地内容：

- `TTSProvider` 新增 `preset_catalog`。
- 新增 `PresetCatalog`，形状对齐 native voice catalog。
- MiMo 固定音色表变成 `MIMO_PRESET_CATALOG`，挂到 hosted `mimo` provider 上。
- 从 `native_voice_registry` 删除 MiMo。
- `/voices` 改成按当前选中 provider 决定：
  - winner 有 `preset_catalog`：返回该目录。
  - winner 没目录，例如 vLLM/GPT-SoVITS：不返回目录，也不 fallback 到 core-native，避免 UI 露出会被误路由的 native voice。
  - 没有 registry winner：走 core-native。
- `validate_voice_id` 支持 selected hosted preset。
- MiMo preset preview 显式返回不支持，同名 clone 仍可放行走 clone preview。

行为前后：

| 场景 | 改前 | 改后 |
| --- | --- | --- |
| MiMo dispatch | hosted provider | 不变 |
| MiMo `/voices` | 借 native registry 吐目录 | hosted `preset_catalog` 吐目录 |
| MiMo 是否 native | `is_native_voice` 可命中 | 不再命中 |
| vLLM/GSV 选中时 `/voices` | 可能露出 core-native | 抑制 native，避免误导 |

### 4.6 #1850：GPT-SoVITS 单一真相

核心文件：

- `utils/config_manager.py`
- `main_logic/tts_client/workers/gptsovits.py`
- `static/js/api_key_settings.js`
- `main_routers/config_router.py`
- `tests/unit/test_api_config_manager.py`
- `tests/unit/test_gptsovits_dropdown_routing.py`
- `tests/frontend/test_api_settings.py`

落地内容：

- `ttsModelProvider == "gptsovits"` 成为 GPT-SoVITS 启用单一真相。
- 旧 `gptsovitsEnabled` 不再由前端保存。
- `ConfigManager.get_core_config()` 的 snapshot 派生 `GPTSOVITS_ENABLED`，下游 13 个读点继续读该字段。
- `_gptsovits_is_selected` 删除自己 raw load `ttsModelProvider` 的逻辑，回到只读派生后的 `GPTSOVITS_ENABLED` + `tts_custom.is_custom`。
- 保存时若用户显式切到非 GSV provider，会惰性清旧 `gptsovitsEnabled`。

兼容策略：

- 显式选择 `gptsovits`：启用。
- 显式选择其他 provider：关闭，即使旧 flag 还残留。
- `ttsModelProvider` 缺失、空串、`follow_assist`、`follow_core`：视作未显式选择，回落旧 `gptsovitsEnabled`。

为什么不能用简单 OR：

```text
旧 flag OR 下拉 == gptsovits
```

因为旧 flag 可能残留在配置里。如果用户已经切到 `vllm_omni`，OR 会让 GSV 粘住，导致无法切走。

### 4.7 #1851：ElevenLabs design 与 MiMo clone

核心文件：

- `main_routers/characters_router.py`
- `main_logic/tts_client/__init__.py`
- `main_logic/tts_client/workers/mimo.py`
- `utils/voice_clone.py`
- `utils/mimo_tts_voices.py`
- `utils/config_manager.py`
- `static/js/voice_clone.js`
- `templates/voice_clone.html`
- `static/js/character_card_manager.js`
- 8 个 locale JSON
- `tests/unit/test_elevenlabs_voice_design.py`
- `tests/unit/test_mimo_voice_clone.py`

#### ElevenLabs voice design

新增能力：

- `elevenlabs` provider 的 capabilities 加 `design`。
- `POST /voice_design_preview`：文字描述调用 ElevenLabs design API，返回 preview 与试听音频。
- `POST /voice_design_create`：从 preview 创建正式 ElevenLabs voice。
- 创建后的 design voice 落成普通 ElevenLabs voice id，`voice_meta.source = "design"`。
- dispatch 复用 ElevenLabs clone 路径：只要 `voice_meta.provider == "elevenlabs"` 即可。

注意：

- design 不需要新 worker。
- 前端 source-first picker 可按 `voice_meta.source` 把 design 分组显示为“描述生成”。

#### MiMo voice clone

调研结论：

- MiMo voiceclone 没有远端注册 voice id。
- MiMo 每次合成时要把参考音频内联到请求里。

最终存储策略：

- MiMo clone 与 MiniMax 对偶：clone 身份存进 `voice_meta`。
- MiMo 的 `voice_meta` 存参考音频 base64，例如 `clone_sample_b64`。
- `get_voices_for_current_api(for_listing=True)` 会剥掉大体积 base64，避免前端列表加载背大 blob。
- dispatch/preview 需要完整 meta 时用 `for_listing=False`。

dispatch 策略：

- `_mimo_is_selected` 增加 `voice_meta.provider == "mimo"` 分支。
- `_mimo_resolve` 读取 sample，构造 data URI，绑定到 `mimo_tts_worker(clone_voice=...)`。
- worker 有 clone voice 时切到 `mimo-v2.5-tts-voiceclone` 模型。

前端变化：

- voice clone 页面新增 MiMo provider。
- MiMo 禁用直链 clone，只允许本地文件上传。
- `voice.provider.mimo`、`voice.mimoApiRequired` 等 i18n 补齐 8 locale。

### 4.8 #1852：GPT-SoVITS 教程按钮

核心文件：

- `templates/api_key_settings.html`
- `static/js/api_key_settings.js`
- `static/css/api_key_settings.css`
- `static/css/dark-mode.css`
- `static/i18n-i18next.js`
- 8 个 locale JSON

落地内容：

- 选中 GPT-SoVITS 时，在 TTS 配置区域显示“教程文档”链接。
- 中文界面指向中文文档。
- 其他语言指向通用文档。
- 监听语言切换，动态更新 href。

这属于 UI 辅助增强，不改变 TTS dispatch。

### 4.9 #1853：hosted preset ownership 修复

核心文件：

- `utils/voice_config.py`
- `utils/config_manager.py`
- `utils/tts_provider_registry.py`
- `tests/unit/test_voice_config.py`
- `tests/unit/test_mimo_tts_voices.py`

问题背景：

#1848 让 MiMo preset 通过 hosted `preset_catalog` 合法化，但保存侧 normalizer 还不认识 hosted preset。用户选 `Milo` 时可能存成：

```json
{"source": "", "provider": "", "ref": "Milo"}
```

这不会立刻破坏运行时，因为 `ref` 还在，读路径也能回到裸 key；但会丢失 ownership 元数据，未来依赖 `source/provider` 的逻辑会受影响。

修复内容：

- `normalize_voice_id()` 新增 `hosted_preset_provider` 注入点。
- `ConfigManager.normalize_voice_id_to_config()` 用 `tts_provider_registry.is_selected_preset_voice` 和 `selected_provider_key` 判断当前 selected hosted preset。
- MiMo preset 保存时能落成：

```json
{"source": "preset", "provider": "mimo", "ref": "Milo"}
```

解析顺序：

```text
vllm -> clone -> native preset -> hosted preset -> free preset -> unresolved ref
```

## 5. 重要运行路径

### 5.1 TTS worker 选择

简化后的运行路径：

```text
core.py / session startup
  -> get_tts_worker(core_api_type, has_custom_voice, voice_id)
    -> build DispatchContext(core_config, cm, voice_id, has_custom_voice, lazy voice_meta)
    -> tts_provider_registry.resolve_selected(ctx)
      -> first provider where is_selected(ctx)
      -> provider.resolve(ctx)
    -> if no registry provider, try native TTS worker
    -> if custom voice fallback, use CosyVoice/Grok/free/default paths
```

关键点：

- registry provider 优先于 native fallback。
- explicit config provider 不应触发 voice storage 查询。
- `voice_meta.provider` 决定 clone/design hosted routing。
- `provider_key` 仍会被上游用于 metadata、normalizer、api key resolution。

### 5.2 角色音色保存

简化路径：

```text
PUT /catgirl/voice_id
  -> ConfigManager.voice_id_to_storage_value(legacy_voice_id)
    -> normalize_voice_id_to_config()
      -> utils.voice_config.normalize_voice_id(...)
    -> if to_legacy(vc) == original
         store vc.to_dict()
       else
         keep original string
```

读路径：

```text
get_reserved(..., "voice_id")
  -> read_legacy_voice_id(raw)
    -> string: strip and return
    -> dict: VoiceConfig.from_any(raw) -> to_legacy_voice_id(vc)
```

这意味着迁移前后的运行时消费者仍看到 legacy `voice_id`。

### 5.3 `/voices`

简化规则：

```text
selected registry provider?
  yes:
    has preset_catalog?
      yes -> return provider preset catalog
      no  -> return no native fallback
  no:
    return core-native voices
```

原因：如果 `vllm_omni` 或 `gptsovits` 已经赢了 dispatch，但 UI 仍展示 core-native voices，用户选中后会被错误路由。

## 6. 未来改 TTS 时的检查清单

### 6.1 新增 provider

新增 provider 时优先走注册表，不要在 `get_tts_worker()` 里继续塞新 if/else。

需要检查：

- `utils/tts_provider_registry.py`
- `main_logic/tts_client/workers/<provider>.py`
- `main_logic/tts_client/__init__.py` 注册块
- 是否需要 `preset_catalog`
- 是否属于 `local` / `hosted`
- capabilities 是 `preset`、`clone`、`design` 中哪些
- `is_selected(ctx)` 是否是配置选中还是 voice meta 选中
- `resolve(ctx)` 的 api key 来源是否明确
- API 设置页是否能通过 `/api/config/api_providers` 的 `tts_providers` 元数据自动渲染

### 6.2 新增 voice source

不要再发明新前缀塞进 `voice_id`。优先扩展：

- `VoiceConfig.source`
- provider `capabilities`
- source-first picker 分组
- enrollment endpoint
- `voice_meta.source`

### 6.3 修改 `voice_id`

必须保持：

```text
store -> read == 原 voice library key
```

若迁移对象会让 `to_legacy(vc)` 改变原 key，应保留旧 string。

同时检查这些读点是否仍经 `read_legacy_voice_id()`：

- `main_logic/core.py`
- `app/main_server.py`
- `main_routers/characters_router.py`
- `utils/config_manager.py`
- 任何直接 `get_reserved(..., "voice_id")` 的新增代码

### 6.4 修改 GPT-SoVITS

当前真相：

- UI 选择：`ttsModelProvider == "gptsovits"`
- snapshot 派生：`GPTSOVITS_ENABLED`
- worker 读取：派生后的 `GPTSOVITS_ENABLED` + `tts_custom.is_custom`
- 旧 `gptsovitsEnabled` 只用于兼容未显式选择 provider 的旧配置

不要重新让前端保存 `gptsovitsEnabled`，也不要恢复独立启用开关。

### 6.5 修改 MiMo

当前分类：

- MiMo 是 hosted provider，不是 native。
- preset 走 `MIMO_PRESET_CATALOG`。
- clone 没有远端 voice id，sample base64 存在 `voice_meta`。
- listing 路径需要剥离大体积 base64。

不要把 MiMo 加回 `native_voice_registry`。

### 6.6 i18n

任何前端可见文案变更都必须同步 8 个 locale：

- `en.json`
- `ja.json`
- `ko.json`
- `zh-CN.json`
- `zh-TW.json`
- `ru.json`
- `pt.json`
- `es.json`

### 6.7 测试建议

按改动范围选择：

- provider dispatch：`tests/unit/test_*tts*`、相关 provider worker 测试。
- voice config/migration：`tests/unit/test_voice_config.py`、`tests/unit/test_voice_lazy_migration.py`。
- GPT-SoVITS 下拉：`tests/unit/test_gptsovits_dropdown_routing.py`、`tests/frontend/test_api_settings.py`。
- MiMo：`tests/unit/test_mimo_tts_voices.py`、`tests/unit/test_mimo_voice_clone.py`。
- ElevenLabs design：`tests/unit/test_elevenlabs_voice_design.py`。
- source-first picker：`tests/frontend/test_character_card_manager_regressions.py`。
- voice clone page：`tests/frontend/test_voice_clone.py`。

项目 Python 命令应使用 `uv run`。

## 7. 已知文档债

以下旧文档可能仍描述旧结构，后续有空应同步更新：

- `docs/modules/tts-client.md` 仍可能写 `main_logic/tts_client.py` 单文件。
- `docs/modules/index.md` 的 TTS Client 路径可能仍指向旧单文件。
- `docs/architecture/tts-pipeline.md` 仍偏旧 pipeline 总览，未覆盖 provider registry/source-first/structured voice。

本文先作为重构落地索引，避免继续改 TTS 时重新从 PR 历史里挖上下文。
