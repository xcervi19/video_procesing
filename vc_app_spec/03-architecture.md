# 03 — Architecture

## System Overview

VideoPipe is a **single-process, in-memory pipeline**. No message queue or distributed stages; the entire DAG runs in one process with a shared `PipelineContext`.

## Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                             │
│  process | subtitles | info | presets                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PipelineConfig (optional YAML/JSON)  →  create_context_from_config│
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline (core/pipeline.py)                                     │
│  - Holds nodes by name                                           │
│  - Resolves execution order (topological sort)                   │
│  - Runs nodes in order; passes PipelineContext                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ LoadVideosNode │     │ GenerateSubs  │     │ ExportNode    │
│ CropNode      │     │ RenderSubs     │     │ ...           │
│ MergeNode     │     │ CreateNeon...  │     │               │
│ ...           │     │ ...            │     │               │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                ┌─────────────────────────────┐
                │   PipelineContext           │
                │   input_files, output_path  │
                │   clips, subtitles, config   │
                │   special_words, metadata   │
                └─────────────────────────────┘
```

## Pipeline Engine

- **Pipeline**: Holds a dict of nodes by name; computes execution order via Kahn’s algorithm from node `dependencies`.
- **Node**: Abstract base; each node has `name`, `dependencies` (list of node names), `process(context) -> NodeResult`, optional `validate(context) -> bool`.
- **Execution**: For each node in order, check dependencies succeeded → validate → process → store optional output in context; on failure, stop (unless configured otherwise).

## Context as Message Bus

- **PipelineContext** is the only shared state. Nodes do not call each other directly.
- Context carries: `input_files`, `output_path`, `work_dir`, `clips` (keyed), `subtitles`, `special_words`, `config`, `export_settings`, `node_outputs`, `metadata`.
- “Main” clip: `context.get_main_clip()` / `context.set_main_clip(clip)`; used by most nodes as the current working clip.

## Plugin System

- **PluginRegistry**: Central registry for effect plugins, transition plugins, processor plugins, and node classes.
- **Discovery**: Can auto-discover classes in `videopipe.effects`, `videopipe.transitions`, `videopipe.subtitles`, `videopipe.nodes`.
- **Config-driven pipelines**: `Pipeline.from_config(config, node_registry)` builds a pipeline from a list of stage names and a registry mapping names to node classes.

## Directory Layout (Logical)

- **cli.py** — Entry point, subparsers, calls into pipeline and nodes.
- **videopipe/core** — `pipeline.py`, `node.py`, `context.py`, `config.py`.
- **videopipe/nodes** — Concrete nodes: `video_nodes`, `subtitle_nodes`, `effect_nodes`.
- **videopipe/video** — `clip.py` (load, info), `merge.py`, `export.py`.
- **videopipe/subtitles** — `whisper_stt.py`, `renderer.py`.
- **videopipe/effects** — Neon, text effects, speed (e.g. `neon.py`, `text_effects.py`, `speed.py`).
- **videopipe/transitions** — `base.py` (Transition, crossfade, wipe), `slide.py`.
- **videopipe/plugins** — `base.py` (Plugin, EffectPlugin, etc.), `registry.py`.
- **videopipe/utils** — Fonts, FFmpeg helpers.
- **examples/** — Standalone pipelines (e.g. `instagram_neon_pipeline.py`).

## Data Flow (Summary)

1. Config (file or CLI) → `PipelineConfig` → `create_context_from_config` → `PipelineContext`.
2. Pipeline built by adding nodes (or from config); each node declares dependencies.
3. `pipeline.run(context)`: topological order → for each node, `node.execute(context)` → `process(context)` reads/writes context (e.g. `set_main_clip`).
4. Export node writes `context.output_path` using `context.get_main_clip()` and export preset.
5. `context.cleanup()` closes clips and releases resources.
