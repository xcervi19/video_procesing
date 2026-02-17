# 06 — Nodes Catalog

All nodes extend `Node`; they live in `videopipe.nodes` (video_nodes, subtitle_nodes, effect_nodes). Dependencies are by **node name** (string).

---

## Video Nodes (video_nodes.py)

| Node | Name | Dependencies | Description |
|------|------|--------------|-------------|
| LoadVideosNode | `load_videos` | — | Load `context.input_files` into `context.clips` (`input_0`, …); set main if single. Optional `target_resolution`. |
| CropNode | `crop` | `load_videos` | Crop by percentage (top, bottom, left, right). Updates main and optionally all input clips. |
| PreviewModeNode | `preview_mode` | `load_videos` | If enabled: subclip(start_time, end_time), optional scale; set main; set metadata preview_mode/start/end. |
| InVideoTransitionNode | `in_video_transition` | `load_videos` | Cut at cut_times; apply transition between segments; rejoin; set main. |
| MergeVideosNode | `merge_videos` | `load_videos` | Concatenate all input clips with optional transition; set result as main. |
| ApplyTransitionNode | `apply_transitions` | `load_videos` | Same as merge but explicitly transition-focused (slide/crossfade/wipe between consecutive clips). |
| ExportNode | `export` | `load_videos` | Write `context.get_main_clip()` to `context.output_path` with given preset. Preview mode → use preview preset and `*_preview.mp4` path. |

---

## Subtitle Nodes (subtitle_nodes.py)

| Node | Name | Dependencies | Description |
|------|------|--------------|-------------|
| GenerateSubtitlesNode | `generate_subtitles` | `load_videos` | Whisper transcribe first input file; fill `context.subtitles` (SubtitleEntry with word_timings); save SRT to work_dir. |
| RenderSubtitlesNode | `render_subtitles` | `generate_subtitles` | Render context.subtitles onto main clip. Uses SubtitleRenderer or AnimatedSubtitleRenderer; respects special_words, spoken_word_highlight, style from config. Font from project `fonts/`. |
| LoadSRTSubtitlesNode | `load_srt_subtitles` | `load_videos` | Parse SRT (from path or same stem as first input); fill context.subtitles. No word_timings unless SRT encodes them. |

---

## Effect Nodes (effect_nodes.py)

| Node | Name | Dependencies | Description |
|------|------|--------------|-------------|
| ApplyEffectsNode | `apply_effects` | `load_videos` | Placeholder: applies list of effect names from config; currently no-op beyond storing main clip. |
| ApplyNeonEffectNode | `apply_neon_effect` | `render_subtitles` | Configures neon for target/special words in context (color, glow); actual neon rendering is in subtitle/overlay renderer. |
| ApplySpokenWordHighlightNode | `apply_spoken_word_highlight` | `generate_subtitles` | Sets context.config["spoken_word_highlight"] (effect e.g. soft_pill, enabled); used by RenderSubtitlesNode. |
| CreateNeonTextOverlay | (configurable name) | `load_videos` or `generate_subtitles` | Adds neon text overlay(s). Fixed (start_time, duration, text) or sync_to_titles (segment/word timing). Optional width_percent, animation (typewriter, pop_in, fade). |
| AddSoundEffectNode | `add_sound_effects` | `load_videos` | Load sounds from folder; add at given times to main clip audio; optional volume/fadeout per sound. |
| ChangeSpeedNode | `change_speed` | `load_videos` | Change playback speed with pitch preservation (e.g. librosa); set new clip as main. |

---

## Dependency Chains (Examples)

- **Minimal export**: `load_videos` → `export`.
- **Merge two clips**: `load_videos` → `ApplyTransitionNode` or `MergeVideosNode` → `export`.
- **Subtitles**: `load_videos` → `generate_subtitles` → `render_subtitles` → … → `export`.
- **Instagram-style**: `load_videos` → optional `PreviewModeNode` → optional `CropNode` → optional `InVideoTransitionNode` → optional `GenerateSubtitlesNode` → `RenderSubtitlesNode` → multiple `CreateNeonTextOverlay` → optional `AddSoundEffectNode` → `ExportNode`. Dependencies are set so each step depends on the previous (by name).

---

## Node Names and Registry

- Node **name** is the identifier used in dependencies and in pipeline’s internal dict. Default is class name with first letter lowercased and CamelCase to snake_case (e.g. `CreateNeonTextOverlay` → `create_neon_overlay` or a custom `name=`).
- For config-driven pipelines, `pipeline_stages` lists these names; registry maps name → node class. Custom names (e.g. for multiple overlays) must be unique in the pipeline.
