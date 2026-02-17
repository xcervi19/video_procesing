# 08 — Subtitles and Effects

## Speech-to-Text (Whisper)

- **Module**: `videopipe.subtitles.whisper_stt`
- **Implementation**: Prefer **faster-whisper**; optional fallback to openai-whisper.
- **WhisperTranscriber**: Constructor takes `model` (tiny, base, small, medium, large, large-v2, large-v3) and optional `language`.
- **transcribe(path, word_timestamps=True)** → TranscriptionResult (segments with words, start/end, confidence).
- **transcribe_to_srt(path, output_path?)** → SRT string and optionally writes file.
- **TranscriptionResult**: `text`, `segments` (list of TranscriptionSegment), `language`, `duration`; `to_srt()` for SRT string.
- **TranscriptionSegment**: `text`, `start`, `end`, `words` (list of WordTiming).
- **WordTiming**: `word`, `start`, `end`, `confidence`.

Segments are converted to **SubtitleEntry** in GenerateSubtitlesNode and stored in `context.subtitles` with `word_timings` for each segment.

---

## Subtitle Rendering

- **Module**: `videopipe.subtitles.renderer`
- **SubtitleStyle**: From config (`subtitle_settings`); font (path resolved from project `fonts/`), font_size, color, stroke_color, stroke_width, position, margin_bottom, position_y_percent, max_words_per_subtitle, bg_color, bg_opacity, line_spacing, fade_in, fade_out.
- **SubtitleRenderer**: Basic rendering (no word animation). Renders segments as text clips, positions by style (e.g. bottom, margin_bottom or position_y_percent), composites onto video.
- **AnimatedSubtitleRenderer**: Word-by-word or segment-level animation; supports **special_words** (neon/styling per word) and **spoken_word_highlight** (e.g. soft_pill background on current word). Uses word_timings from segments.
- **Font resolution**: RenderSubtitlesNode resolves font name to path under project `fonts/` (e.g. Arial-Bold → file matching in fonts/). No system fallback in strict mode; FileNotFoundError if missing.
- **Segments**: Renderer expects list of TranscriptionSegment-like objects (text, start, end, words). RenderSubtitlesNode converts context.subtitles (SubtitleEntry) to these using word_timings.

Spoken-word highlight config: `context.config["spoken_word_highlight"]` with `enabled`, `effect` (e.g. soft_pill), and style overrides. If enabled, animation is used and current word gets highlight (e.g. pill background).

---

## Neon Effects

- **Module**: `videopipe.effects.neon`
- **NeonConfig**: color (hex), glow_intensity, glow_radius, pulse, pulse_speed, optional font, font_size.
- **NeonGlowEffect**: Basic neon glow text clip.
- **FuturisticTextEffect**: Neon with secondary color / futuristic look.
- Used by: CreateNeonTextOverlay (standalone overlays) and AnimatedSubtitleRenderer for special_words (inline neon on words).

CreateNeonTextOverlay can create multiple segments: either fixed (start_time, duration, text) or **sync_to_titles** (timing from context.subtitles). With `use_subtitle_word=True`, one overlay per word (or filtered to special_words if configured). Supports **width_percent** (text width as % of frame width; font size computed to fit). **Animation**: typewriter, pop_in, fade (from ProfessionalTextAnimation / text_effects.py).

---

## Text Effects (Animation)

- **Module**: `videopipe.effects.text_effects`
- **ProfessionalTextAnimation** / **ProfessionalAnimationConfig**: typewriter (chars_per_second), pop_in (scale), fade; entrance/exit duration and easing.
- **AnimationTiming**: Used for timing of character/word reveals.
- Used by CreateNeonTextOverlay and potentially by subtitle renderer for word reveals.

---

## Transitions

- **Module**: `videopipe.transitions.base`, `videopipe.transitions.slide`
- **Transition** (abstract): `duration`, `easing`; `make_frame(clip_a, clip_b, t, progress)` returns frame; `apply(clip_a, clip_b)` builds transition clip and concatenates (clip_a minus overlap + transition + clip_b minus overlap).
- **Easing**: linear, ease_in_out, ease_out_expo, ease_in_expo, ease_in_out_expo (dict EASING_FUNCTIONS).
- **CrossfadeTransition**: Blend clip_a out, clip_b in by progress.
- **WipeTransition**: Direction (LEFT, RIGHT, UP, DOWN), softness; mask reveals clip_b.
- **SlideTransition** (slide.py): One clip slides in (e.g. from right), other slides out; optional motion_blur, perspective. **SlideConfig**: direction, duration, easing, motion_blur, motion_blur_samples, perspective, overlap, shadow.
- **QuickSlideTransition**: Lighter-weight slide variant.
- **create_transition(type, **kwargs)**: Factory; type in crossfade, wipe, slide, quick_slide.

---

## Speed Change

- **Module**: `videopipe.effects.speed`
- **change_speed_preserve_pitch(clip, factor)**: Time-stretch with pitch preservation (e.g. librosa). factor > 1 = faster, < 1 = slower. Returns new clip; audio and video duration scaled.

---

## Plugins (Base)

- **Module**: `videopipe.plugins.base`
- **Plugin** (abstract): `metadata`, `config`; `apply(clip, context, **kwargs) -> VideoClip`.
- **EffectPlugin**, **TransitionPlugin**, **ProcessorPlugin**, **SubtitlePlugin**, **TextEffectPlugin**: Specializations for registry and discovery.
- **PluginRegistry** (registry.py): register_effect, register_transition, register_processor, register_node; get_*; discover_plugins(package_name); list_plugins().
