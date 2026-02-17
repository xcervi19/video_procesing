# 02 — Requirements

## Functional Requirements

### Input / Output

- **FR-1** Accept one or more input video files (paths from config or CLI).
- **FR-2** Support a single output file path; format inferred from extension or preset.
- **FR-3** Support YAML and JSON configuration files with validation and sensible defaults.

### Pipeline

- **FR-4** Execute a directed acyclic graph (DAG) of nodes in topological order.
- **FR-5** Support node dependencies by name; execution order resolved automatically.
- **FR-6** Pass a single shared context object through all nodes (no direct node-to-node coupling).
- **FR-7** Support dry-run mode (list nodes in execution order without running).

### Video

- **FR-8** Load video with MoviePy; optional target resolution; store clips in context by key (e.g. `input_0`, `main`).
- **FR-9** Merge multiple clips with optional transition (slide, crossfade, wipe) between consecutive clips.
- **FR-10** Support in-video transitions: cut at given time(s), apply transition between segments, rejoin.
- **FR-11** Crop by percentage (top, bottom, left, right); apply to main and optionally all input clips.
- **FR-12** Preview mode: trim to a time range (and optionally scale) for fast iteration.
- **FR-13** Change playback speed with pitch preservation (e.g. librosa-based time-stretch).

### Subtitles

- **FR-14** Generate subtitles from video audio using Whisper (word-level timestamps).
- **FR-15** Load subtitles from existing SRT file as an alternative to Whisper.
- **FR-16** Render subtitles onto video: configurable font, size, position, stroke, background.
- **FR-17** Optional word-by-word animation and spoken-word highlight (e.g. soft pill background).
- **FR-18** Support “special words” with distinct styling (e.g. neon); list configurable in config/context.

### Effects & Overlays

- **FR-19** Neon text overlays: timed (start/duration) or synced to subtitle segments/words.
- **FR-20** Neon styling: color, glow, pulse; optional futuristic effect.
- **FR-21** Optional professional animations for overlays: typewriter, pop-in, fade.
- **FR-22** Add sound effects from a folder at specified times; optional volume and fadeout.

### Export

- **FR-23** Export using presets: ProRes 422 HQ/LT, ProRes 4444, H.264, H.265, etc.
- **FR-24** In preview mode, use a fast preset and output to a distinct file (e.g. `*_preview.mp4`).

### CLI

- **FR-25** Commands: `process` (run pipeline), `subtitles` (Whisper-only to SRT), `info` (video metadata), `presets` (list export presets).
- **FR-26** `process`: accept `-i`/`-o`, `--config`, `--subtitles`, `--transition`, `--preset`, `--dry-run`, etc.

## Non-Functional Requirements

- **NFR-1** Python 3.10+.
- **NFR-2** FFmpeg available on system for probe/export; no GUI dependency.
- **NFR-3** Fonts for subtitles/overlays: project `fonts/` directory (e.g. .ttf/.otf); no mandatory system font list.
- **NFR-4** Logging: configurable level; pipeline and nodes log progress and errors.
- **NFR-5** Fail fast: pipeline stops on first node failure unless configured otherwise; context cleanup on exit.
