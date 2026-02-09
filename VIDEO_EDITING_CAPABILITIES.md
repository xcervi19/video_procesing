# Implemented Video Editing Capabilities

This file lists the video editing features that are already implemented.

## Clip handling
- Load single or multiple video files (with audio)
- Optional input resizing to a target resolution
- Crop by percentage from any edge
- Merge multiple clips into one sequence
- Preview mode to trim to a time range (optional downscale)

## Transitions
- Slide transitions (directional, with motion blur and shadow options)
- Quick slide transition variant
- Crossfade transition
- Directional wipe transition
- Transitions between clips or inside a single clip at cut points

## Timing and speed
- Change playback speed with pitch-preserved audio

## Subtitles and text
- Auto-generate subtitles with Whisper (word-level timing)
- Import existing SRT subtitles
- Render subtitles on video (basic or animated)
- Word-by-word subtitle animation (kinetic typography)
- Spoken-word highlight background (karaoke-style)
- Neon text overlays at specific times and positions
- Neon text overlay animations: typewriter, pop-in, fade

## Audio
- Add sound effects at specific times (volume control, fadeout)
- Composite sound effects with existing audio

## Export
- Preset exports for ProRes 422 HQ/LT, ProRes 4444, H.264, H.265
- Instagram Reels export preset
