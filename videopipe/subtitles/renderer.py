"""
Subtitle rendering with animations and special effects.

Provides:
- Basic subtitle rendering
- Word-by-word animation (kinetic typography)
- Special word highlighting (neon effects, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from moviepy import (
    VideoClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

if TYPE_CHECKING:
    from videopipe.core.context import PipelineContext, SubtitleEntry
    from videopipe.subtitles.whisper_stt import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


@dataclass
class SubtitleStyle:
    """Styling options for subtitles."""
    font: str = "Arial-Bold"
    font_size: int = 48
    color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 2
    bg_color: Optional[str] = None
    bg_opacity: float = 0.6
    position: tuple[str, str] = ("center", "bottom")
    margin_bottom: int = 50
    margin_sides: int = 40
    
    # Animation settings
    fade_in: float = 0.1
    fade_out: float = 0.1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "font": self.font,
            "font_size": self.font_size,
            "color": self.color,
            "stroke_color": self.stroke_color,
            "stroke_width": self.stroke_width,
            "bg_color": self.bg_color,
            "bg_opacity": self.bg_opacity,
            "position": self.position,
            "margin_bottom": self.margin_bottom,
            "margin_sides": self.margin_sides,
            "fade_in": self.fade_in,
            "fade_out": self.fade_out,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleStyle:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SubtitleRenderer:
    """
    Basic subtitle renderer.
    
    Renders subtitles as text overlays on video clips with
    configurable styling and positioning.
    """
    
    def __init__(self, style: Optional[SubtitleStyle] = None):
        self.style = style or SubtitleStyle()
    
    def create_subtitle_clip(
        self,
        text: str,
        duration: float,
        video_size: tuple[int, int],
        style: Optional[SubtitleStyle] = None,
    ) -> TextClip:
        """
        Create a single subtitle text clip.
        
        Args:
            text: Subtitle text
            duration: Duration in seconds
            video_size: (width, height) of the video
            style: Optional style override
            
        Returns:
            TextClip with the subtitle
        """
        style = style or self.style
        width, height = video_size
        
        # Calculate max width for text wrapping
        max_width = width - (style.margin_sides * 2)
        
        # Create text clip
        txt_clip = TextClip(
            text=text,
            font=style.font,
            font_size=style.font_size,
            color=style.color,
            stroke_color=style.stroke_color,
            stroke_width=style.stroke_width,
            method="caption",
            size=(max_width, None),
            text_align="center",
        )
        
        txt_clip = txt_clip.with_duration(duration)
        
        # Apply fade effects
        if style.fade_in > 0:
            txt_clip = txt_clip.with_effects([
                lambda clip: clip.crossfadein(style.fade_in)
            ])
        
        # Position the subtitle
        x_pos = "center"
        y_pos = height - style.margin_bottom - txt_clip.h
        
        txt_clip = txt_clip.with_position((x_pos, y_pos))
        
        return txt_clip
    
    def render_subtitles(
        self,
        video: VideoClip,
        segments: list[TranscriptionSegment],
        style: Optional[SubtitleStyle] = None,
    ) -> VideoClip:
        """
        Render subtitles onto a video clip.
        
        Args:
            video: The base video clip
            segments: Transcription segments with timing
            style: Optional style override
            
        Returns:
            Video clip with subtitles overlaid
        """
        style = style or self.style
        subtitle_clips = []
        
        for segment in segments:
            txt_clip = self.create_subtitle_clip(
                text=segment.text,
                duration=segment.duration,
                video_size=(video.w, video.h),
                style=style,
            )
            txt_clip = txt_clip.with_start(segment.start)
            subtitle_clips.append(txt_clip)
        
        # Composite all subtitle clips onto video
        return CompositeVideoClip([video] + subtitle_clips)


class AnimatedSubtitleRenderer(SubtitleRenderer):
    """
    Advanced subtitle renderer with word-by-word animations.
    
    Supports:
    - Kinetic typography (word-by-word reveal)
    - Special word highlighting
    - Custom animation effects per word
    """
    
    def __init__(
        self,
        style: Optional[SubtitleStyle] = None,
        special_words: Optional[dict[str, dict[str, Any]]] = None,
    ):
        super().__init__(style)
        self.special_words = special_words or {}
    
    def add_special_word(self, word: str, effect_config: dict[str, Any]):
        """Mark a word for special effect treatment."""
        self.special_words[word.lower()] = effect_config
    
    def _is_special_word(self, word: str) -> bool:
        """Check if a word should have special effects."""
        return word.lower().strip(".,!?;:'\"") in self.special_words
    
    def _get_word_effect(self, word: str) -> Optional[dict[str, Any]]:
        """Get the effect configuration for a special word."""
        clean_word = word.lower().strip(".,!?;:'\"")
        return self.special_words.get(clean_word)
    
    def create_word_clip(
        self,
        word: str,
        duration: float,
        video_size: tuple[int, int],
        style: Optional[SubtitleStyle] = None,
        effect_config: Optional[dict[str, Any]] = None,
    ) -> TextClip:
        """
        Create a clip for a single word with optional special effects.
        
        Args:
            word: The word text
            duration: Duration in seconds
            video_size: (width, height) of the video
            style: Style override
            effect_config: Special effect configuration
            
        Returns:
            TextClip for the word
        """
        style = style or self.style
        
        # Determine styling
        color = style.color
        font_size = style.font_size
        stroke_color = style.stroke_color
        stroke_width = style.stroke_width
        
        # Apply special effects if configured
        if effect_config:
            color = effect_config.get("color", color)
            font_size = effect_config.get("font_size", font_size)
            stroke_color = effect_config.get("stroke_color", stroke_color)
            stroke_width = effect_config.get("stroke_width", stroke_width)
        
        txt_clip = TextClip(
            text=word,
            font=style.font,
            font_size=font_size,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        
        txt_clip = txt_clip.with_duration(duration)
        
        return txt_clip
    
    def render_animated_segment(
        self,
        segment: TranscriptionSegment,
        video_size: tuple[int, int],
        style: Optional[SubtitleStyle] = None,
    ) -> VideoClip:
        """
        Render a segment with word-by-word animation.
        
        Words appear one at a time as they are spoken,
        creating a kinetic typography effect.
        """
        style = style or self.style
        width, height = video_size
        
        if not segment.words:
            # Fall back to basic rendering if no word timings
            return self.create_subtitle_clip(
                segment.text, segment.duration, video_size, style
            )
        
        word_clips = []
        current_text_parts = []
        
        for i, word_timing in enumerate(segment.words):
            # Build up the text as words are spoken
            current_text_parts.append(word_timing.word)
            current_text = " ".join(current_text_parts)
            
            # Calculate duration until next word or end
            if i < len(segment.words) - 1:
                duration = segment.words[i + 1].start - word_timing.start
            else:
                duration = segment.end - word_timing.start
            
            # Skip very short durations
            if duration < 0.01:
                continue
            
            # Check for special word effects
            effect_config = self._get_word_effect(word_timing.word)
            
            # Create the cumulative text clip
            max_width = width - (style.margin_sides * 2)
            
            # Build styled text with highlight for current word
            txt_clip = self._create_highlighted_text_clip(
                full_text=current_text,
                highlight_word=word_timing.word if effect_config else None,
                effect_config=effect_config,
                duration=duration,
                max_width=max_width,
                style=style,
            )
            
            # Position
            y_pos = height - style.margin_bottom - txt_clip.h
            txt_clip = txt_clip.with_position(("center", y_pos))
            txt_clip = txt_clip.with_start(word_timing.start - segment.start)
            
            word_clips.append(txt_clip)
        
        if not word_clips:
            return self.create_subtitle_clip(
                segment.text, segment.duration, video_size, style
            )
        
        # Composite all word states
        composite = CompositeVideoClip(word_clips, size=video_size)
        composite = composite.with_duration(segment.duration)
        
        return composite
    
    def _create_highlighted_text_clip(
        self,
        full_text: str,
        highlight_word: Optional[str],
        effect_config: Optional[dict[str, Any]],
        duration: float,
        max_width: int,
        style: SubtitleStyle,
    ) -> TextClip:
        """
        Create a text clip with optional word highlighting.
        
        For simple cases, creates a basic TextClip.
        For highlighted words, the styling is applied to make them stand out.
        """
        # For now, create basic text clip
        # TODO: Implement proper word-level highlighting with compositing
        
        color = style.color
        
        # If the last word is highlighted, use highlight color for emphasis
        if highlight_word and effect_config:
            # We'll use a simple approach: style the whole text
            # A more advanced approach would composite multiple TextClips
            pass
        
        txt_clip = TextClip(
            text=full_text,
            font=style.font,
            font_size=style.font_size,
            color=color,
            stroke_color=style.stroke_color,
            stroke_width=style.stroke_width,
            method="caption",
            size=(max_width, None),
            text_align="center",
        )
        
        txt_clip = txt_clip.with_duration(duration)
        
        return txt_clip
    
    def render_subtitles(
        self,
        video: VideoClip,
        segments: list[TranscriptionSegment],
        style: Optional[SubtitleStyle] = None,
        animate: bool = True,
    ) -> VideoClip:
        """
        Render animated subtitles onto a video.
        
        Args:
            video: Base video clip
            segments: Transcription segments with word timings
            style: Style override
            animate: Whether to use word-by-word animation
            
        Returns:
            Video with animated subtitles
        """
        if not animate:
            return super().render_subtitles(video, segments, style)
        
        style = style or self.style
        subtitle_clips = []
        
        for segment in segments:
            if segment.words:
                # Use animated rendering
                sub_clip = self.render_animated_segment(
                    segment, (video.w, video.h), style
                )
            else:
                # Fall back to basic rendering
                sub_clip = self.create_subtitle_clip(
                    segment.text, segment.duration, (video.w, video.h), style
                )
            
            sub_clip = sub_clip.with_start(segment.start)
            subtitle_clips.append(sub_clip)
        
        return CompositeVideoClip([video] + subtitle_clips)
