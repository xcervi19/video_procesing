# 09 — Setup and Run

## Prerequisites

- **Python**: 3.10 or higher.
- **FFmpeg**: Installed on system (used for probe and export).
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: Download from https://ffmpeg.org/download.html
- **Optional**: CUDA for faster Whisper; Apple Silicon uses MPS automatically with PyTorch.

## Dependencies (Python)

From project root:

```bash
pip install -r requirements.txt
```

Or install as package (editable) with optional dev deps:

```bash
pip install -e ".[dev]"
```

Core dependencies (see `requirements.txt` / `pyproject.toml`):

- moviepy >= 2.0.0
- numpy >= 1.24.0
- Pillow >= 10.0.0
- faster-whisper >= 1.0.0
- torch >= 2.0.0
- PyYAML >= 6.0
- librosa >= 0.10.0 (speed change with pitch preservation)

Dev (optional): pytest, pytest-cov, black, ruff, mypy.

## Fonts

Place font files (`.ttf` or `.otf`) in project **fonts/** directory. Subtitle and overlay nodes resolve font names (e.g. `Arial-Bold`, `Bebas Neue`) against this directory. If a font is missing, the pipeline can raise FileNotFoundError (no system fallback in strict path).

## Running the CLI

```bash
# Process one video, output ProRes
videopipe process -i video.mp4 -o output.mov

# Process with subtitles and slide transition for two clips
videopipe process -i clip1.mp4 clip2.mp4 -o merged.mov --subtitles --transition slide

# Generate SRT only
videopipe subtitles -i video.mp4 -o subs.srt --model medium

# Video info
videopipe info video.mp4

# List export presets
videopipe presets

# Use config file
videopipe process --config pipeline.yaml
```

## Running the Example Pipeline

```bash
# From project root
python examples/instagram_neon_pipeline.py -i input.mp4 -o output.mov

# With YAML config
python examples/instagram_neon_pipeline.py --config configs/my_edit.yaml

# Dry run
python examples/instagram_neon_pipeline.py --config configs/my_edit.yaml --dry-run
```

The example builds a pipeline that can include: load, preview mode, crop, in-video transitions, subtitles (if spoken_word_highlight or sync_to_titles overlays), neon text overlays, sound effects, export. Config keys: `input_files`, `output_path`, `preview_mode`, `preview`, `crop`, `in_video_transitions`, `transition_type`, `transition_duration`, `spoken_word_highlight`, `whisper_model`, `neon_settings`, `text_overlays`, `sound_effects`, `export`.

## Programmatic Usage

```python
from pathlib import Path
from videopipe.core.pipeline import Pipeline
from videopipe.core.context import PipelineContext, create_context_from_config
from videopipe.nodes import LoadVideosNode, MergeVideosNode, GenerateSubtitlesNode, RenderSubtitlesNode, ExportNode

# Minimal: load and export
context = PipelineContext(
    input_files=[Path("input.mp4")],
    output_path=Path("output.mov"),
)
pipeline = Pipeline(name="Minimal")
pipeline.add_node(LoadVideosNode())
pipeline.add_node(ExportNode(preset="prores_422_hq"))
results = pipeline.run(context)
context.cleanup()
```

## Tests

```bash
pytest tests/ -v
```

Test paths and options are in `pyproject.toml` (testpaths, addopts).

## Linting / Formatting

- **Black**: `black .` (line-length 100).
- **Ruff**: `ruff check .` (select E, W, F, I, B, C4; ignore E501, B008).
- **MyPy**: `mypy videopipe` (python 3.10, strict options; see pyproject.toml).

## Output and Logs

- Pipeline and nodes use Python `logging`; level controlled by CLI `-v`/`-q`.
- Example script may tee stdout/stderr to `.cursor/output.log` for IDE integration.
- Working directory for intermediates is a temp dir in `context.work_dir` (e.g. SRT file). Cleanup does not delete work_dir by default (can be changed for production).

## Reproducing the Application

To reproduce VideoPipe from this spec:

1. Create Python package `videopipe` with packages under `videopipe.*` (core, nodes, video, subtitles, effects, transitions, plugins, utils).
2. Implement core: Pipeline (topological sort, run loop), Node (execute, validate, process), PipelineContext, PipelineConfig (load/save/validate).
3. Implement video: clip (load_clip, get_clip_info), merge (merge_clips, MergeConfig), export (presets, VideoExporter).
4. Implement nodes per 06-nodes-catalog; wire dependencies by name.
5. Implement subtitles: WhisperTranscriber, SubtitleRenderer / AnimatedSubtitleRenderer, SubtitleStyle; font resolution from `fonts/`.
6. Implement effects and transitions per 08; register in plugin registry if using discovery.
7. Implement CLI (argparse) per 07.
8. Add example script that builds pipeline from args or YAML (e.g. instagram_neon_pipeline.py).
9. Install deps, add fonts, run tests and manual runs per 09.
