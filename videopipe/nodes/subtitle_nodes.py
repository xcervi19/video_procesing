"""
Subtitle processing pipeline nodes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from videopipe.core.node import Node, NodeResult
from videopipe.core.context import PipelineContext, SubtitleEntry
from videopipe.subtitles.whisper_stt import WhisperTranscriber
from videopipe.subtitles.renderer import (
    SubtitleRenderer,
    AnimatedSubtitleRenderer,
    SubtitleStyle,
)
from videopipe.utils.fonts import get_font_path_for_config, find_font_file

logger = logging.getLogger(__name__)

# Fallback font paths when the configured font cannot be loaded (e.g. Arial-Bold on macOS)
SUBTITLE_FONT_FALLBACKS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
]


class GenerateSubtitlesNode(Node):
    """
    Generate subtitles from video audio using Whisper.
    
    Transcribes the audio track and stores subtitle entries
    in the pipeline context with word-level timing.
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        whisper_model: str = "medium",
        language: Optional[str] = None,
    ):
        super().__init__(
            name="generate_subtitles",
            config=config,
            dependencies=["load_videos"],
        )
        self.whisper_model = whisper_model
        self.language = language
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            # Get the first input file for transcription
            if not context.input_files:
                return NodeResult.failure_result(ValueError("No input files"))
            
            input_path = context.input_files[0]
            
            logger.info(f"Generating subtitles with Whisper ({self.whisper_model})")
            logger.info(f"Input: {input_path}")
            
            # Initialize transcriber
            transcriber = WhisperTranscriber(
                model=self.whisper_model,
                language=self.language,
            )
            
            # Transcribe
            result = transcriber.transcribe(
                input_path,
                word_timestamps=True,
            )
            
            logger.info(f"Detected language: {result.language}")
            logger.info(f"Transcribed {len(result.segments)} segments")
            
            # Convert to SubtitleEntry objects
            for segment in result.segments:
                word_timings = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "confidence": w.confidence,
                    }
                    for w in segment.words
                ]
                
                entry = SubtitleEntry(
                    text=segment.text,
                    start_time=segment.start,
                    end_time=segment.end,
                    word_timings=word_timings,
                )
                context.add_subtitle(entry)
            
            # Save SRT file in work directory
            srt_path = context.get_temp_path("subtitles.srt")
            srt_content = result.to_srt()
            srt_path.write_text(srt_content, encoding="utf-8")
            logger.info(f"Saved SRT to: {srt_path}")
            
            return NodeResult.success_result(
                output=result,
                segments=len(result.segments),
                language=result.language,
                srt_path=str(srt_path),
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


class RenderSubtitlesNode(Node):
    """
    Render subtitles onto the video with optional animation.
    
    Supports:
    - Basic subtitle rendering
    - Word-by-word animation
    - Special effects for marked words (e.g., neon glow)
    - Spoken-word highlight effects (background emphasis)
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        animated: bool = True,
        style: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            name="render_subtitles",
            config=config,
            dependencies=["generate_subtitles"],
        )
        self.animated = animated
        self.style_config = style or {}
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip"))
            
            if not context.subtitles:
                logger.warning("No subtitles to render")
                return NodeResult.success_result(output=clip)
            
            # Build style from config and context
            style_dict = {
                **context.config.get("subtitle_settings", {}),
                **self.style_config,
            }
            # Resolve subtitle font to a path Pillow can open (avoid "cannot open resource")
            font_name = style_dict.get("font", "Arial-Bold")
            if callable(font_name):
                font_name = "Arial-Bold"
            resolved = get_font_path_for_config(font_name, auto_download=False)
            if callable(resolved):
                resolved = font_name
            if resolved != font_name and Path(resolved).exists():
                style_dict["font"] = str(resolved)
            else:
                found = find_font_file(font_name) or find_font_file("Arial") or find_font_file("Helvetica")
                if found:
                    style_dict["font"] = str(found)
                else:
                    for fallback in SUBTITLE_FONT_FALLBACKS:
                        if Path(fallback).exists():
                            style_dict["font"] = fallback
                            logger.info(f"Subtitle font '{font_name}' not found, using fallback: {fallback}")
                            break
            # Ensure font is never a callable (would cause 'function' object has no attribute 'copy' in TextClip)
            if callable(style_dict.get("font")):
                style_dict["font"] = next(
                    (f for f in SUBTITLE_FONT_FALLBACKS if Path(f).exists()),
                    "/System/Library/Fonts/Helvetica.ttc",
                )
            style = SubtitleStyle.from_dict(style_dict)
            word_highlight = context.config.get("spoken_word_highlight") or None
            if not isinstance(word_highlight, dict):
                word_highlight = None
            highlight_enabled = bool(word_highlight) and word_highlight.get("enabled", True)
            use_animated = self.animated or highlight_enabled
            
            logger.info(f"Rendering {len(context.subtitles)} subtitle segments")
            logger.info(f"Animation: {'enabled' if use_animated else 'disabled'}")
            if highlight_enabled:
                effect_name = word_highlight.get("effect", "soft_pill") if word_highlight else "soft_pill"
                logger.info(f"Spoken-word highlight: {effect_name}")
            
            show_only_special = (
                (context.config.get("subtitles") or {}).get("show_only_special_words", False)
            )

            # Choose renderer: animated (effects) vs basic (plain text only)
            if use_animated:
                renderer = AnimatedSubtitleRenderer(
                    style=style,
                    special_words=context.special_words,
                    word_highlight=word_highlight,
                    show_only_special_words=show_only_special,
                )
            else:
                renderer = SubtitleRenderer(style=style)

            # Convert SubtitleEntry to TranscriptionSegment-like objects
            from videopipe.subtitles.whisper_stt import TranscriptionSegment, WordTiming
            
            segments = []
            for entry in context.subtitles:
                words = [
                    WordTiming(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        confidence=w.get("confidence", 1.0),
                    )
                    for w in entry.word_timings
                ]
                
                segment = TranscriptionSegment(
                    text=entry.text,
                    start=entry.start_time,
                    end=entry.end_time,
                    words=words,
                )
                segments.append(segment)
            
            # Render: AnimatedSubtitleRenderer uses animate=True for word highlight
            result_clip = renderer.render_subtitles(
                clip, segments, style=style, animate=use_animated
            )

            context.set_main_clip(result_clip)
            
            return NodeResult.success_result(
                output=result_clip,
                subtitles_rendered=len(segments),
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


class LoadSRTSubtitlesNode(Node):
    """
    Load subtitles from an existing SRT file.
    
    Alternative to generating subtitles with Whisper when
    subtitles already exist.
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        srt_path: Optional[Path | str] = None,
    ):
        super().__init__(
            name="load_srt_subtitles",
            config=config,
            dependencies=["load_videos"],
        )
        self.srt_path = Path(srt_path) if srt_path else None
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            srt_path = self.srt_path
            
            if srt_path is None:
                # Try to find SRT with same name as first input
                if context.input_files:
                    srt_path = context.input_files[0].with_suffix(".srt")
            
            if srt_path is None or not srt_path.exists():
                return NodeResult.failure_result(
                    FileNotFoundError(f"SRT file not found: {srt_path}")
                )
            
            logger.info(f"Loading subtitles from: {srt_path}")
            
            # Parse SRT file
            content = srt_path.read_text(encoding="utf-8")
            entries = self._parse_srt(content)
            
            for entry in entries:
                context.add_subtitle(entry)
            
            logger.info(f"Loaded {len(entries)} subtitle entries")
            
            return NodeResult.success_result(
                output=entries,
                entries_loaded=len(entries),
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)
    
    def _parse_srt(self, content: str) -> list[SubtitleEntry]:
        """Parse SRT content into SubtitleEntry objects."""
        import re
        
        entries = []
        
        # Split into blocks
        blocks = re.split(r'\n\n+', content.strip())
        
        time_pattern = re.compile(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})'
        )
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # Second line should be timing
            time_match = time_pattern.match(lines[1])
            if not time_match:
                continue
            
            # Parse times
            start_time = (
                int(time_match.group(1)) * 3600 +
                int(time_match.group(2)) * 60 +
                int(time_match.group(3)) +
                int(time_match.group(4)) / 1000
            )
            
            end_time = (
                int(time_match.group(5)) * 3600 +
                int(time_match.group(6)) * 60 +
                int(time_match.group(7)) +
                int(time_match.group(8)) / 1000
            )
            
            # Remaining lines are text
            text = '\n'.join(lines[2:])
            
            entries.append(SubtitleEntry(
                text=text,
                start_time=start_time,
                end_time=end_time,
            ))
        
        return entries
