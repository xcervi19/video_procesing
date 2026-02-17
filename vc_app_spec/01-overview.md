# 01 — Application Overview

## What Is VideoPipe?

**VideoPipe** is a **professional video processing pipeline** for creating Instagram-style content. It is:

- **Pipeline-based**: A DAG of processing nodes (load → process → export).
- **No GUI**: Pure data pipeline; driven by CLI and/or YAML/JSON config.
- **AI-powered**: Uses Whisper for automatic speech-to-text and word-level subtitles.
- **Extensible**: Plugin system for effects, transitions, and custom nodes.

## Goals

- Merge multiple videos with optional transitions (slide, crossfade, wipe).
- Generate subtitles from audio (Whisper) with word-level timing.
- Render subtitles with optional animation, neon effects, and spoken-word highlight.
- Support neon text overlays (timed or synced to subtitles), sound effects, crop, speed change.
- Export to professional codecs (e.g. ProRes 422 HQ) or fast preview (e.g. H.264).

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Video I/O & editing | MoviePy 2.x |
| Speech-to-text | faster-whisper (Whisper) |
| ML runtime | PyTorch 2.x |
| Config | PyYAML, JSON |
| CLI | argparse |
| External | FFmpeg (system), optionally librosa (time-stretch) |

## High-Level Flow

1. **Input**: One or more video files (paths in config or CLI).
2. **Load**: Videos loaded into pipeline context as MoviePy clips (`input_0`, `input_1`, …; single clip also set as `main`).
3. **Process**: Optional chain of nodes: crop, preview trim, merge/transitions, subtitles (generate + render), neon overlays, sound effects, speed change, etc.
4. **Export**: Main clip written to disk with chosen preset (e.g. ProRes 422 HQ or H.264 fast for preview).

## Entry Points

- **CLI**: `videopipe` (or `python -m cli`) with subcommands `process`, `subtitles`, `info`, `presets`.
- **Programmatic**: Build a `Pipeline`, add nodes, create `PipelineContext`, call `pipeline.run(context)`.
- **Example script**: `examples/instagram_neon_pipeline.py` — full pipeline from CLI args or YAML config.

## Out of Scope (Current Spec)

- Real-time preview GUI.
- Cloud or distributed execution.
- Native mobile apps.
