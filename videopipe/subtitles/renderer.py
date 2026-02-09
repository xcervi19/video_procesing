"""
Subtitle rendering with animations and special effects.

Provides:
- Basic subtitle rendering
- Word-by-word animation (kinetic typography)
- Special word highlighting (neon effects, etc.)
- Spoken-word highlighting with background emphasis
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
    ImageClip,
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
    line_spacing: int = 8
    
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
            "line_spacing": self.line_spacing,
            "fade_in": self.fade_in,
            "fade_out": self.fade_out,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleStyle:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SpokenWordHighlightStyle:
    """Styling options for spoken-word highlighting."""
    bg_color: str = "#FDE68A"
    bg_opacity: float = 0.85
    text_color: str = "#111827"
    stroke_color: Optional[str] = None
    stroke_width: int = 0
    padding_x: int = 12
    padding_y: int = 4
    corner_radius: int = 8
    reveal_mode: str = "full"
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpokenWordHighlightStyle:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


SPOKEN_WORD_HIGHLIGHT_PRESETS: dict[str, dict[str, Any]] = {
    # Soft pill highlight: warm background, calm contrast, rounded edges.
    "soft_pill": {
        "bg_color": "#FDE68A",
        "bg_opacity": 0.85,
        "text_color": "#111827",
        "stroke_color": None,
        "stroke_width": 0,
        "padding_x": 12,
        "padding_y": 4,
        "corner_radius": 8,
        "reveal_mode": "full",
    },
}


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
        word_highlight: Optional[dict[str, Any]] = None,
    ):
        super().__init__(style)
        self.special_words = special_words or {}
        self.word_highlight = word_highlight or {}
    
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
    
    def _resolve_spoken_word_highlight(self) -> Optional[dict[str, Any]]:
        """Resolve spoken-word highlight configuration from presets + overrides."""
        if not self.word_highlight:
            return None
        if isinstance(self.word_highlight, dict) and self.word_highlight.get("enabled") is False:
            return None
        if not isinstance(self.word_highlight, dict):
            return None
        
        effect_name = self.word_highlight.get("effect") or self.word_highlight.get("name") or "soft_pill"
        preset = SPOKEN_WORD_HIGHLIGHT_PRESETS.get(effect_name)
        if preset is None:
            logger.warning(
                "Unknown spoken-word highlight effect '%s'; falling back to 'soft_pill'",
                effect_name,
            )
            effect_name = "soft_pill"
            preset = SPOKEN_WORD_HIGHLIGHT_PRESETS[effect_name]
        
        overrides = {
            k: v
            for k, v in self.word_highlight.items()
            if k not in ("effect", "name", "enabled")
        }
        resolved = {**preset, **overrides}
        resolved["effect"] = effect_name
        return resolved
    
    @staticmethod
    def _load_pil_font(font: str, font_size: int):
        from PIL import ImageFont
        
        try:
            return ImageFont.truetype(font, font_size)
        except OSError:
            return ImageFont.load_default()
    
    def _measure_space_width(self, font: str, font_size: int) -> int:
        from PIL import Image, ImageDraw
        
        pil_font = self._load_pil_font(font, font_size)
        temp_img = Image.new("RGBA", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), " ", font=pil_font)
        width = bbox[2] - bbox[0]
        if width <= 0:
            width = int(font_size * 0.35)
        return width
    
    def _layout_word_clips(
        self,
        word_items: list[dict[str, Any]],
        max_width: int,
        space_width: int,
        line_spacing: int,
    ) -> tuple[list[dict[str, Any]], int, int]:
        lines: list[tuple[list[dict[str, Any]], int]] = []
        current_line: list[dict[str, Any]] = []
        current_width = 0
        
        for item in word_items:
            word_width = item["clip"].w
            if current_line:
                next_width = current_width + space_width + word_width
            else:
                next_width = word_width
            
            if current_line and next_width > max_width:
                lines.append((current_line, current_width))
                current_line = [item]
                current_width = word_width
            else:
                current_line.append(item)
                current_width = next_width
        
        if current_line:
            lines.append((current_line, current_width))
        
        if not lines:
            return [], 0, 0
        
        block_width = max(width for _, width in lines)
        layout_items: list[dict[str, Any]] = []
        y = 0
        
        for line_items, line_width in lines:
            line_height = max(item["clip"].h for item in line_items)
            x = (block_width - line_width) / 2
            
            for item in line_items:
                layout_items.append({**item, "x": x, "y": y, "line_height": line_height})
                x += item["clip"].w + space_width
            
            y += line_height + line_spacing
        
        block_height = y - line_spacing
        return layout_items, int(block_width), int(block_height)
    
    @staticmethod
    def _parse_color_rgba(color: Any, opacity: float) -> tuple[int, int, int, int]:
        if color is None:
            return (0, 0, 0, 0)
        
        if isinstance(color, (tuple, list)) and len(color) >= 3:
            r, g, b = color[:3]
        elif isinstance(color, str):
            from PIL import ImageColor
            
            if color.startswith("#"):
                hex_color = color.lstrip("#")
                if len(hex_color) == 3:
                    hex_color = "".join(ch * 2 for ch in hex_color)
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                else:
                    r, g, b = ImageColor.getrgb(color)
            else:
                r, g, b = ImageColor.getrgb(color)
        else:
            return (0, 0, 0, 0)
        
        alpha = int(max(0.0, min(opacity, 1.0)) * 255)
        return (int(r), int(g), int(b), alpha)
    
    def _create_highlight_background_clip(
        self,
        width: int,
        height: int,
        highlight_style: SpokenWordHighlightStyle,
    ) -> Optional[ImageClip]:
        if width <= 0 or height <= 0:
            return None
        
        from PIL import Image, ImageDraw
        
        radius = max(0, min(int(highlight_style.corner_radius), min(width, height) // 2))
        fill_color = self._parse_color_rgba(
            highlight_style.bg_color, highlight_style.bg_opacity
        )
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (0, 0, width, height),
            radius=radius,
            fill=fill_color,
        )
        return ImageClip(np.array(image))
    
    @staticmethod
    def _word_state_window(
        index: int,
        timings: list[Any],
        segment: Any,
    ) -> tuple[float, float]:
        start = timings[index].start - segment.start
        if index < len(timings) - 1:
            end = timings[index + 1].start - segment.start
        else:
            end = segment.end - segment.start
        
        if end <= start:
            fallback = timings[index].end - timings[index].start
            end = start + max(0.05, fallback)
        
        return start, end
    
    def _render_spoken_word_highlight_segment(
        self,
        segment: TranscriptionSegment,
        video_size: tuple[int, int],
        style: SubtitleStyle,
        highlight_config: dict[str, Any],
    ) -> VideoClip:
        if not segment.words:
            return self.create_subtitle_clip(
                segment.text, segment.duration, video_size, style
            )
        
        highlight_style = SpokenWordHighlightStyle.from_dict(highlight_config)
        max_width = max(1, int(video_size[0] - (style.margin_sides * 2)))
        line_spacing = max(0, int(style.line_spacing))
        space_width = self._measure_space_width(style.font, style.font_size)
        
        word_items: list[dict[str, Any]] = []
        for word_timing in segment.words:
            display_word = word_timing.word.strip()
            if not display_word:
                continue
            
            word_clip = TextClip(
                text=display_word,
                font=style.font,
                font_size=style.font_size,
                color=style.color,
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width,
            )
            word_items.append({
                "word": display_word,
                "clip": word_clip,
                "timing": word_timing,
            })
        
        if not word_items:
            return self.create_subtitle_clip(
                segment.text, segment.duration, video_size, style
            )
        
        layout_items, block_width, block_height = self._layout_word_clips(
            word_items,
            max_width=max_width,
            space_width=space_width,
            line_spacing=line_spacing,
        )
        
        if not layout_items:
            return self.create_subtitle_clip(
                segment.text, segment.duration, video_size, style
            )
        
        block_x = (video_size[0] - block_width) / 2
        block_y = video_size[1] - style.margin_bottom - block_height
        
        reveal_mode = (highlight_style.reveal_mode or "full").lower()
        highlight_text_color = highlight_style.text_color or style.color
        highlight_stroke_color = (
            highlight_style.stroke_color
            if highlight_style.stroke_color is not None
            else style.stroke_color
        )
        highlight_stroke_width = highlight_style.stroke_width
        padding_x = max(0, int(highlight_style.padding_x))
        padding_y = max(0, int(highlight_style.padding_y))
        
        base_clips: list[VideoClip] = []
        background_clips: list[VideoClip] = []
        highlight_text_clips: list[VideoClip] = []
        timings = [item["timing"] for item in layout_items]
        
        for index, item in enumerate(layout_items):
            word_timing = item["timing"]
            word_start, word_end = self._word_state_window(index, timings, segment)
            highlight_duration = word_end - word_start
            if highlight_duration < 0.01:
                continue
            
            if reveal_mode == "progressive":
                base_start = word_timing.start - segment.start
                base_duration = max(0.01, segment.end - word_timing.start)
            else:
                base_start = 0.0
                base_duration = segment.duration
            
            base_clip = item["clip"].with_start(base_start).with_duration(base_duration)
            base_clip = base_clip.with_position(
                (block_x + item["x"], block_y + item["y"])
            )
            base_clips.append(base_clip)
            
            bg_width = int(item["clip"].w + (padding_x * 2))
            bg_height = int(item["clip"].h + (padding_y * 2))
            bg_clip = self._create_highlight_background_clip(
                bg_width, bg_height, highlight_style
            )
            if bg_clip is not None:
                bg_clip = bg_clip.with_start(word_start).with_duration(highlight_duration)
                bg_clip = bg_clip.with_position(
                    (block_x + item["x"] - padding_x, block_y + item["y"] - padding_y)
                )
                background_clips.append(bg_clip)
            
            highlight_clip = TextClip(
                text=item["word"],
                font=style.font,
                font_size=style.font_size,
                color=highlight_text_color,
                stroke_color=highlight_stroke_color,
                stroke_width=highlight_stroke_width,
            )
            highlight_clip = highlight_clip.with_start(word_start).with_duration(highlight_duration)
            highlight_clip = highlight_clip.with_position(
                (block_x + item["x"], block_y + item["y"])
            )
            highlight_text_clips.append(highlight_clip)
        
        composite = CompositeVideoClip(
            background_clips + base_clips + highlight_text_clips,
            size=video_size,
        ).with_duration(segment.duration)
        
        if style.fade_in > 0:
            composite = composite.with_effects([
                lambda clip: clip.crossfadein(style.fade_in)
            ])
        
        return composite
    
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
        
        highlight_config = self._resolve_spoken_word_highlight()
        if highlight_config:
            return self._render_spoken_word_highlight_segment(
                segment, video_size, style, highlight_config
            )
        
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
