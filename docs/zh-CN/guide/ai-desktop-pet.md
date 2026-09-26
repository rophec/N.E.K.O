---
title: AI 桌宠、AI 桌面宠物与虚拟伴侣｜Project N.E.K.O.
titleTemplate: false
description: Project N.E.K.O. 是开源 AI 桌宠与 AI 桌面宠物，支持 Live2D、VRM、语音文字对话、长期记忆、Agent 与插件扩展。
seoSchemaType: WebPage
seoFaq:
  - question: 什么是 AI 桌宠？
    answer: AI 桌宠是在桌面界面中呈现角色，并把动画形象与对话或助手能力结合起来的软件。Project N.E.K.O. 还提供语音文字交互、按角色长期记忆、可选主动行为、Agent 与插件。
  - question: Project N.E.K.O. 是开源 AI 虚拟伴侣吗？
    answer: 是。Project N.E.K.O. 在 GitHub 以 Apache License 2.0 发布源码，用户可以检查实现、从源码运行并通过插件扩展。
  - question: N.E.K.O. 猫娘桌宠支持 Live2D 和 VRM 吗？
    answer: 支持。N.E.K.O. 支持 Live2D、VRM、MMD 与 PNGTuber 形象格式，不同格式分别提供相应的模型、动画、表情和交互能力。
  - question: N.E.K.O. 有长期记忆吗？
    answer: N.E.K.O. 按角色维护近期记忆、事实、反思和 Persona 知识。数据默认存储在本机，但部分记忆处理会使用用户配置的模型 Provider。
  - question: 这个本地 AI 助手能完全离线运行吗？
    answer: 目前不是一键即可完全离线的产品。界面和默认记忆存储在本机，部分组件可自托管，但常用模型、语音、Steam、浏览器、信息源和 Agent 功能可能仍需联网。
---

# AI 桌宠、AI 桌面宠物与虚拟伴侣

Project N.E.K.O. 是一个**开源 AI 桌宠**和虚拟伴侣：动画角色可以运行在桌宠窗口中，通过语音或文字交流，维护按角色隔离的长期记忆，并使用可选 Agent 和插件扩展。搜索“AI 桌面宠物”“猫娘桌宠”或“虚拟伴侣 AI”的用户，通常需要的不只是桌面动画，而是可见角色、对话能力和持续状态的组合。

## N.E.K.O. 为什么属于 AI 桌宠？

| 能力 | N.E.K.O. 提供的实现 | 核验页面 |
| --- | --- | --- |
| 动画角色 | Live2D、VRM、MMD 与 PNGTuber 四类形象格式 | [Avatar 与前端文档](/zh-CN/frontend/) |
| 语音和文字 | 可配置 Provider 的 Realtime 或普通请求/响应对话路径 | [架构概览](/zh-CN/architecture/) |
| 长期上下文 | 近期记忆、事实、反思、Persona 知识与显式召回 | [长期记忆架构](/zh-CN/architecture/memory-system) |
| 桌面形态 | Electron 分发中的主形象窗口，以及独立聊天和字幕窗口 | [页面与模板](/zh-CN/frontend/pages) |
| 主动能力 | 启用对应功能时，可选的屏幕上下文与内容感知交互 | [数据与隐私边界](./data-and-privacy) |
| 扩展能力 | 开源代码、插件 SDK、市场接入、API 与 Agent 入口 | [插件文档](/zh-CN/plugins/) |

传统桌宠一般只负责移动、播放动画或响应点击。AI 虚拟伴侣还需要对话循环、可控数据流、持久状态，以及把角色表情和回答连接起来的机制。N.E.K.O. 将这些层分开，便于开发者检查和配置。

## 猫娘桌宠可以选择 Live2D 或 VRM

[Live2D](/zh-CN/frontend/live2d) 适合使用 Cubism 模型资源的表现型 2D 角色。N.E.K.O. 可以把语义情绪映射到模型动作与表情，导入用户模型，并加载受支持的创意工坊资源。

[VRM](/zh-CN/frontend/vrm) 适合 3D 人形 Avatar。N.E.K.O. 可加载 `.vrm` 模型和 `.vrma` 动画，把对话情绪映射到表情，并可选地向兼容 VMC Protocol 的接收端发送动作。

MMD 与 PNGTuber 提供更多视觉形式。不同格式的能力并不完全相同，应以对应形象文档为准。

## 长期记忆不等于所有处理都在本机

N.E.K.O. 的伴侣记忆按角色隔离，包含有限工作上下文、近期对话、抽取事实、更高层反思与 Persona 知识。它不是单一向量数据库，向量 Embedding 也是可选召回路径，而不是唯一机制。

记忆文件默认保存在本机，但本地存储不代表每一步处理都不会联网。摘要、抽取、反思或当前对话可能访问配置的 Provider。选择方案前，请阅读[记忆系统](/zh-CN/architecture/memory-system)、[数据与隐私指南](./data-and-privacy)和[本地与离线指南](./local-and-offline)。

## 开源与安装入口

Project N.E.K.O. 源码已在 [GitHub](https://github.com/Project-N-E-K-O/N.E.K.O) 以 Apache License 2.0 发布。开发者可以从源码运行、检查 API 并开发插件；需要打包分发版的用户可比较 [Steam、GitHub Releases 与源码安装方式](./install-options)。

> 想体验这款 AI 桌面宠物？[前往 Steam 查看 Project N.E.K.O.](https://store.steampowered.com/app/4099310/__NEKO/?utm_source=project-neko.online&utm_medium=referral&utm_campaign=ai_desktop_pet&utm_content=category_page_zh_cn)，或[从开源仓库开始](https://github.com/Project-N-E-K-O/N.E.K.O)。

## 常见问题

### 什么是 AI 桌宠？

AI 桌宠是在桌面界面中呈现角色，并把动画形象与对话或助手能力结合起来的软件。Project N.E.K.O. 还提供语音文字交互、按角色长期记忆、可选主动行为、Agent 与插件。

### Project N.E.K.O. 是开源 AI 虚拟伴侣吗？

是。Project N.E.K.O. 在 GitHub 以 Apache License 2.0 发布源码，用户可以检查实现、从源码运行并通过插件扩展。

### N.E.K.O. 猫娘桌宠支持 Live2D 和 VRM 吗？

支持。N.E.K.O. 支持 Live2D、VRM、MMD 与 PNGTuber 形象格式，不同格式分别提供相应的模型、动画、表情和交互能力。

### N.E.K.O. 有长期记忆吗？

N.E.K.O. 按角色维护近期记忆、事实、反思和 Persona 知识。数据默认存储在本机，但部分记忆处理会使用用户配置的模型 Provider。

### 这个本地 AI 助手能完全离线运行吗？

目前不是一键即可完全离线的产品。界面和默认记忆存储在本机，部分组件可自托管，但常用模型、语音、Steam、浏览器、信息源和 Agent 功能可能仍需联网。

