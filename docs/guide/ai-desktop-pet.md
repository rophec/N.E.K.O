---
title: AI Desktop Pet and Open-Source AI Companion
titleTemplate: false
description: Meet Project N.E.K.O., an open-source AI desktop pet and companion with Live2D or VRM avatars, voice and text chat, long-term memory, agents, and plugins.
seoSchemaType: WebPage
seoFaq:
  - question: What is an AI desktop pet?
    answer: An AI desktop pet is a character that lives in a desktop interface and combines an animated avatar with conversation or assistant features. Project N.E.K.O. adds voice and text interaction, persistent per-character memory, optional proactive behavior, agents, and plugins.
  - question: Is Project N.E.K.O. an open-source AI companion?
    answer: Yes. Project N.E.K.O. publishes its source code on GitHub under the Apache License 2.0. Users can inspect the implementation, run from source, and extend it with plugins.
  - question: Does the N.E.K.O. desktop pet support Live2D and VRM?
    answer: Yes. N.E.K.O. supports Live2D and VRM avatars as well as MMD and PNGTuber formats. Each format has its own model, animation, expression, and interaction capabilities.
  - question: Does N.E.K.O. have long-term memory?
    answer: N.E.K.O. maintains per-character recent memory, extracted facts, reflections, and persona knowledge. Storage is local by default, while some memory processing can use the configured model provider.
  - question: Can the AI desktop companion run completely offline?
    answer: Not as a one-click fully offline product. The interface and default memory storage are local, and selected components can be self-hosted, but common model, voice, Steam, browser, feed, and agent features may require network access.
---

# AI Desktop Pet and Open-Source AI Companion

Project N.E.K.O. is an **open-source AI desktop pet** and companion: an animated character that can live in a desktop-pet window, talk by voice or text, maintain per-character memory, use optional agent channels, and be extended through plugins. People searching for a “Neko desktop pet” or a “desktop pet with AI” are usually looking for this combination of visible character, conversation, and persistent behavior rather than a decorative animation alone.

## What makes N.E.K.O. an AI desktop pet?

| Capability | What N.E.K.O. provides | Where to verify it |
| --- | --- | --- |
| Animated character | Live2D, VRM, MMD, and PNGTuber avatar formats | [Avatar and frontend documentation](/frontend/) |
| Voice and text | Realtime or request/response conversation paths with configurable providers | [Architecture overview](/architecture/) |
| Long-term context | Recent memory, facts, reflections, persona knowledge, and explicit recall | [Long-term memory architecture](/architecture/memory-system) |
| Desktop behavior | A main avatar window plus separate chat and subtitle windows in the Electron distribution | [Pages and templates](/frontend/pages) |
| Proactive features | Optional screen-aware and content-aware interaction when the related features are enabled | [Data and privacy boundaries](./data-and-privacy) |
| Extensibility | Open-source code, a plugin SDK, marketplace integration, APIs, and agent-facing entries | [Plugin documentation](/plugins/) |

Classic desktop pets normally animate, wander, or react to clicks. An AI desktop companion additionally needs a conversation loop, controllable data flow, durable state, and a way to connect character expressions to its responses. N.E.K.O. keeps those layers separate so developers can inspect and configure them.

## Choose Live2D or VRM for the companion

[Live2D](/frontend/live2d) is useful for expressive 2D characters built from Cubism model assets. N.E.K.O. can map semantic emotions to model motions and expressions, import user models, and load supported Workshop assets.

[VRM](/frontend/vrm) is useful for 3D humanoid avatars. N.E.K.O. loads `.vrm` models and `.vrma` animations, maps conversation emotions to expressions, and can optionally publish motion to a VMC-compatible receiver.

MMD and PNGTuber formats provide additional visual options. Feature support differs by format, so the avatar documentation is the source of truth for model-specific behavior.

## Long-term memory without pretending everything is local

N.E.K.O.'s companion memory is scoped by character. It combines a bounded working context with recent conversation, extracted facts, higher-level reflections, and persona knowledge. The system is not just a vector database, and vector embeddings are optional rather than the only retrieval path.

Memory files are local by default, but local storage does not mean every processing step stays on the device. Summarization, extraction, reflection, or active conversation can contact the providers selected in the configuration. Read the [memory system](/architecture/memory-system), [data and privacy guide](./data-and-privacy), and [local/offline guide](./local-and-offline) before choosing a setup.

## Open-source routes and installation choices

The Project N.E.K.O. source is available on [GitHub](https://github.com/Project-N-E-K-O/N.E.K.O) under the Apache License 2.0. Developers can run from source, inspect the APIs, and build plugins. Users who prefer a packaged distribution can review the [Steam, GitHub Releases, and source installation options](./install-options).

> Ready to try the AI desktop pet? [View Project N.E.K.O. on Steam](https://store.steampowered.com/app/4099310/__NEKO/?utm_source=project-neko.online&utm_medium=referral&utm_campaign=ai_desktop_pet&utm_content=category_page_en) or [start with the open-source repository](https://github.com/Project-N-E-K-O/N.E.K.O).

## Frequently asked questions

### What is an AI desktop pet?

An AI desktop pet is a character that lives in a desktop interface and combines an animated avatar with conversation or assistant features. Project N.E.K.O. adds voice and text interaction, persistent per-character memory, optional proactive behavior, agents, and plugins.

### Is Project N.E.K.O. an open-source AI companion?

Yes. Project N.E.K.O. publishes its source code on GitHub under the Apache License 2.0. Users can inspect the implementation, run from source, and extend it with plugins.

### Does the N.E.K.O. desktop pet support Live2D and VRM?

Yes. N.E.K.O. supports Live2D and VRM avatars as well as MMD and PNGTuber formats. Each format has its own model, animation, expression, and interaction capabilities.

### Does N.E.K.O. have long-term memory?

N.E.K.O. maintains per-character recent memory, extracted facts, reflections, and persona knowledge. Storage is local by default, while some memory processing can use the configured model provider.

### Can the AI desktop companion run completely offline?

Not as a one-click fully offline product. The interface and default memory storage are local, and selected components can be self-hosted, but common model, voice, Steam, browser, feed, and agent features may require network access.

