# 04 — Data Models

## PipelineContext

Shared state passed to every node. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `input_files` | `list[Path]` | Input video paths |
| `output_path` | `Path \| None` | Final output file path |
| `work_dir` | `Path` | Temp directory for intermediates (e.g. SRT) |
| `clips` | `dict[str, Any]` | Video clips by key: `input_0`, `input_1`, `main` |
| `audio_clips` | `dict[str, Any]` | Audio clips if stored separately |
| `metadata` | `dict` | e.g. `VideoMetadata` per clip, `preview_mode`, `preview_start`, `preview_end` |
| `subtitles` | `list[SubtitleEntry]` | Subtitle segments with timing and optional word_timings |
| `special_words` | `dict[str, dict]` | Word (lowercase) → effect config (e.g. neon color) |
| `node_outputs` | `dict[str, Any]` | Output of each node by node name |
| `config` | `dict` | Full config (file + overrides) |
| `export_settings` | `dict` | Codec, profile, pixel format, audio codec |

Methods (conceptual):

- `add_clip(key, clip, metadata?)`, `get_clip(key)`, `get_main_clip()`, `set_main_clip(clip)`
- `add_subtitle(entry)`, `add_special_word(word, effect_config)`
- `get_temp_path(filename)`, `store_node_output(node_name, output)`, `get_node_output(node_name)`
- `cleanup()` — close clips, release resources

## VideoMetadata

Simple dataclass for clip metadata: `width`, `height`, `fps`, `duration`, `codec`, `audio_codec`, `bitrate` (optional). Can be built from a MoviePy clip via `from_clip(clip)`.

## SubtitleEntry

Single subtitle segment:

- `text: str`
- `start_time: float`, `end_time: float`
- `style: dict` (optional)
- `word_timings: list[dict]` — each: `word`, `start`, `end`, optional `confidence`

## ClipInfo (video/clip.py)

From FFprobe or MoviePy fallback: `path`, `width`, `height`, `fps`, `duration`, `codec`, `pix_fmt`, `bitrate`, `audio_codec`, `audio_sample_rate`, `audio_channels`. Used by CLI `info` command.

## PipelineConfig

Config container (from YAML/JSON or defaults):

- `input_files`, `output_path`
- `export_settings` — codec, profile, pix_fmt, audio_codec, quality
- `subtitle_settings` — font, font_size, color, stroke_*, position, margin_bottom, position_y_percent, max_words_per_subtitle, bg_*, line_spacing
- `neon_settings` — color, glow_intensity, glow_radius, pulse_speed, animation_type
- `spoken_word_highlight` — enabled, effect (e.g. soft_pill), style overrides
- `transition_settings` — duration, type, direction, easing
- `special_words` — word → effect config
- `pipeline_stages` — list of stage names for config-driven build
- `whisper_model`, `debug`

Validation: input files exist, output parent exists, valid export profile and Whisper model.

## NodeResult

Result of `node.process(context)`:

- `status`: enum — PENDING, RUNNING, COMPLETED, FAILED, SKIPPED
- `output`: optional value stored in context for downstream nodes
- `error`: optional Exception for FAILED
- `metadata`: optional dict

Factory methods: `success_result(output?, **metadata)`, `failure_result(error, **metadata)`, `skipped_result(reason)`.

## Export Presets

Preset (e.g. ExportPreset) has: `name`, `codec`, `container`, `pixel_format`, `profile`, `quality`, `video_bitrate`, `audio_codec`, `audio_bitrate`, `audio_sample_rate`, `extra_params`. Converts to FFmpeg params via `to_ffmpeg_params()`.

Named presets: `prores_422_hq`, `prores_422_lt`, `prores_4444`, `h264`, `h265`, `h264_fast` (preview), etc.

## MergeConfig

For merging clips: `transition` (Transition instance or None), `transition_duration`, `target_resolution`, `resize_method` (fit/fill/stretch), `audio_crossfade`, `audio_crossfade_duration`, `pad_color`.

## Whisper / STT Types

- **WordTiming**: `word`, `start`, `end`, `confidence`
- **TranscriptionSegment**: `text`, `start`, `end`, `words: list[WordTiming]`, `language`, `confidence`
- **TranscriptionResult**: `text`, `segments`, `language`, `duration`; method `to_srt()`.

## SubtitleStyle (renderer)

Dataclass: font, font_size, color, stroke_color, stroke_width, bg_color, bg_opacity, position, margin_bottom, margin_sides, line_spacing, position_y_percent, max_words_per_subtitle, fade_in, fade_out. `from_dict` / `to_dict` for config.

## Transition Types

- **Transition** (base): `duration`, `easing`; abstract `make_frame(clip_a, clip_b, t, progress)`; `apply(clip_a, clip_b)` returns single clip.
- **TransitionDirection**: LEFT, RIGHT, UP, DOWN, diagonals.
- Implementations: CrossfadeTransition, WipeTransition, SlideTransition, QuickSlideTransition (see 08).
