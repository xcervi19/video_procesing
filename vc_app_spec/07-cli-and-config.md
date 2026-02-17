# 07 — CLI and Configuration

## CLI Entry Point

- **Command**: `videopipe` (from `pyproject.toml` script) or `python cli.py`.
- **Global flags**: `-v/--verbose`, `-q/--quiet` (logging level).

## Subcommands

### process

Run the video processing pipeline.

| Option | Description |
|--------|-------------|
| `-i, --input` | One or more input video files |
| `-o, --output` | Output file path |
| `-c, --config` | YAML or JSON config file (can supply input_files, output_path, and full pipeline config) |
| `--preset` | Export preset (default: prores_422_hq) |
| `--subtitles` | Enable Whisper + render subtitles |
| `--no-animate` | Disable subtitle animation |
| `--whisper-model` | Whisper model (default: medium) |
| `--transition` | Type: slide, crossfade, wipe (when multiple inputs) |
| `--transition-duration` | Seconds |
| `--special-words` | Words to highlight (e.g. neon); space-separated |
| `--dry-run` | Print execution order only |

When `--config` is used, input/output can be overridden by `-i`/`-o`. Config is loaded via `PipelineConfig.from_file()`; context is created with `create_context_from_config(config.to_dict())`. Pipeline is built in code from options (see cli.py: add LoadVideosNode, conditionally Merge or ApplyTransition, optionally GenerateSubtitles + RenderSubtitles, ExportNode).

### subtitles

Generate SRT from video audio only (no full pipeline).

| Option | Description |
|--------|-------------|
| `-i, --input` | Input video (required) |
| `-o, --output` | Output SRT path (default: input with .srt) |
| `--model` | Whisper model (default: medium) |
| `--language` | Force language (optional) |
| `--print` | Print SRT to stdout |

Uses `WhisperTranscriber` and `transcribe_to_srt()`.

### info

Print video metadata (resolution, fps, duration, codecs, etc.) for one or more files. Uses `get_clip_info()` from `videopipe.video.clip`.

### presets

List available export presets (name, codec, container, pixel format, profile, quality, audio codec). Uses `get_available_presets()` from `videopipe.video.export`.

---

## Configuration File Format (YAML / JSON)

Config is merged with defaults; paths can be relative. Structure:

```yaml
# Input/Output
input_files:
  - path/to/video1.mp4
  - path/to/video2.mp4
output_path: path/to/output.mov

# Pipeline stages (for Pipeline.from_config)
pipeline_stages:
  - load_videos
  - generate_subtitles
  - apply_effects
  - apply_transitions
  - export

# Whisper
whisper_model: medium  # tiny, base, small, medium, large, large-v2, large-v3

# Export (merged into export_settings)
export_settings:
  codec: prores_ks
  profile: 3
  pix_fmt: yuv422p10le
  audio_codec: pcm_s24le
  quality: 9

# Subtitles (merged into subtitle_settings)
subtitle_settings:
  font: Arial-Bold
  font_size: 48
  color: white
  stroke_color: black
  stroke_width: 2
  position: [center, bottom]
  margin_bottom: 50
  position_y_percent: null
  max_words_per_subtitle: null

# Neon (for overlays / special words)
neon_settings:
  color: "#39FF14"
  glow_intensity: 1.5
  glow_radius: 10
  pulse_speed: 2.0
  animation_type: pulse
  font: Bebas Neue

# Spoken-word highlight (used by RenderSubtitlesNode)
spoken_word_highlight:
  enabled: true
  effect: soft_pill

# Transitions
transition_settings:
  duration: 0.5
  type: slide
  direction: left
  easing: ease_in_out

# Special words (word -> effect config)
special_words:
  word1:
    type: neon
    color: "#FF00FF"
  word2:
    type: neon
    color: "#39FF14"

# Optional: preview mode (example script)
preview_mode: false
preview:
  start_time: 0
  end_time: 30
  scale: null

# Optional: crop (example script)
crop:
  top: 0
  bottom: 10
  left: 0
  right: 0

# Optional: in-video transitions (example script)
in_video_transitions: [5.0, 10.0]
transition_type: slide
transition_duration: 0.3

# Optional: text overlays (example script)
text_overlays:
  - name: overlay1
    text: "Hello"
    start_time: 1.0
    duration: 2.0
    position: [center, center]
    sync_to_titles: false
    use_subtitle_word: false
    width_percent: 50
    animation: typewriter
  - name: word_highlight
    text: ""
    sync_to_titles: true
    use_subtitle_word: true
    animation: pop_in

# Optional: sound effects (example script)
sound_effects:
  folder: effects_sound
  sounds:
    - name: whoosh.mp3
      time: 1.0
      volume: 0.8
    - name: impact.wav
      time: 4.0
  fadeout: true
  fadeout_duration: 1.0

# Optional: export section (example script)
export:
  preset: prores_422_hq
```

Validation (PipelineConfig): input files exist, output parent exists, export profile in valid set, whisper_model in allowed list.
