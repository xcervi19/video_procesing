"""
Text animation effects for dynamic subtitles and titles.

Provides various text animation styles including:
- Pop-in effects
- Typewriter effects
- Scale/zoom effects
- Bounce effects
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from moviepy import (
    VideoClip,
    TextClip,
    CompositeVideoClip,
    ColorClip,
)

if TYPE_CHECKING:
    from videopipe.core.context import PipelineContext

logger = logging.getLogger(__name__)


# ==================== Easing Functions ====================

def ease_in_out_cubic(t: float) -> float:
    """Smooth ease in/out using cubic interpolation."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_bounce(t: float) -> float:
    """Bouncy ease out effect."""
    n1 = 7.5625
    d1 = 2.75
    
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


def ease_out_elastic(t: float) -> float:
    """Elastic overshoot effect."""
    c4 = (2 * math.pi) / 3
    
    if t == 0:
        return 0
    elif t == 1:
        return 1
    else:
        return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_out_back(t: float) -> float:
    """Overshoot then settle effect."""
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


EASING_FUNCTIONS = {
    "linear": lambda t: t,
    "ease_in_out": ease_in_out_cubic,
    "ease_out_bounce": ease_out_bounce,
    "ease_out_elastic": ease_out_elastic,
    "ease_out_back": ease_out_back,
}


# ==================== Base Classes ====================

@dataclass
class TextAnimationConfig:
    """Configuration for text animations."""
    duration: float = 0.3  # Animation duration
    delay: float = 0.0  # Delay before animation starts
    easing: str = "ease_out_back"
    hold_duration: Optional[float] = None  # Duration to hold after animation
    
    def get_easing_func(self) -> Callable[[float], float]:
        return EASING_FUNCTIONS.get(self.easing, EASING_FUNCTIONS["linear"])


class TextEffect(ABC):
    """Base class for text animation effects."""
    
    def __init__(self, config: Optional[TextAnimationConfig] = None):
        self.config = config or TextAnimationConfig()
    
    @abstractmethod
    def apply_to_clip(
        self,
        clip: TextClip,
        start_time: float,
        **kwargs
    ) -> VideoClip:
        """
        Apply the animation effect to a text clip.
        
        Args:
            clip: The text clip to animate
            start_time: When the animation should start
            **kwargs: Additional effect-specific parameters
            
        Returns:
            Animated video clip
        """
        pass


class PopInEffect(TextEffect):
    """
    Text pops in with scale and optional bounce.
    
    The text starts small/invisible and scales up to full size
    with an optional overshoot effect.
    """
    
    def __init__(
        self,
        config: Optional[TextAnimationConfig] = None,
        start_scale: float = 0.0,
        end_scale: float = 1.0,
    ):
        super().__init__(config)
        self.start_scale = start_scale
        self.end_scale = end_scale
    
    def apply_to_clip(
        self,
        clip: TextClip,
        start_time: float = 0,
        **kwargs
    ) -> VideoClip:
        """Apply pop-in animation to text clip."""
        anim_duration = self.config.duration
        easing = self.config.get_easing_func()
        start_scale = self.start_scale
        end_scale = self.end_scale
        
        original_w, original_h = clip.w, clip.h
        
        def make_frame(t):
            # Calculate animation progress
            if t < start_time:
                progress = 0
            elif t < start_time + anim_duration:
                progress = (t - start_time) / anim_duration
                progress = easing(progress)
            else:
                progress = 1
            
            # Calculate current scale
            scale = start_scale + (end_scale - start_scale) * progress
            
            if scale <= 0:
                # Return transparent frame
                return np.zeros((original_h, original_w, 4), dtype=np.uint8)
            
            # Get the frame and resize
            frame = clip.get_frame(t)
            
            if scale != 1.0:
                from PIL import Image
                img = Image.fromarray(frame)
                new_size = (int(original_w * scale), int(original_h * scale))
                if new_size[0] > 0 and new_size[1] > 0:
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Center in original dimensions
                    result = np.zeros((original_h, original_w, frame.shape[2]), dtype=np.uint8)
                    x_offset = (original_w - new_size[0]) // 2
                    y_offset = (original_h - new_size[1]) // 2
                    
                    # Clip to bounds
                    src_x1 = max(0, -x_offset)
                    src_y1 = max(0, -y_offset)
                    dst_x1 = max(0, x_offset)
                    dst_y1 = max(0, y_offset)
                    
                    copy_w = min(new_size[0] - src_x1, original_w - dst_x1)
                    copy_h = min(new_size[1] - src_y1, original_h - dst_y1)
                    
                    if copy_w > 0 and copy_h > 0:
                        img_array = np.array(img)
                        result[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
                            img_array[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
                    
                    return result
            
            return frame
        
        return VideoClip(make_frame, duration=clip.duration).with_fps(clip.fps or 30)


class TypewriterEffect(TextEffect):
    """
    Text appears character by character like typing.
    """
    
    def __init__(
        self,
        config: Optional[TextAnimationConfig] = None,
        chars_per_second: float = 20,
        cursor: str = "|",
        show_cursor: bool = True,
    ):
        super().__init__(config)
        self.chars_per_second = chars_per_second
        self.cursor = cursor
        self.show_cursor = show_cursor
    
    def apply_to_text(
        self,
        text: str,
        duration: float,
        text_style: dict[str, Any],
        video_size: tuple[int, int],
    ) -> VideoClip:
        """
        Create typewriter animation for text.
        
        Args:
            text: The full text to type
            duration: Total duration
            text_style: TextClip styling parameters
            video_size: (width, height) for positioning
            
        Returns:
            Animated clip showing typing effect
        """
        chars = list(text)
        total_chars = len(chars)
        type_duration = total_chars / self.chars_per_second
        
        def make_frame(t):
            # Calculate how many characters to show
            if t < type_duration:
                progress = t / type_duration
                num_chars = int(progress * total_chars)
            else:
                num_chars = total_chars
            
            current_text = "".join(chars[:num_chars])
            
            # Add cursor
            if self.show_cursor and t < type_duration:
                # Blinking cursor
                if int(t * 4) % 2 == 0:
                    current_text += self.cursor
            
            # Create text clip for this frame
            txt = TextClip(
                text=current_text or " ",
                **text_style
            )
            
            return txt.get_frame(0)
        
        # Create the animated clip
        sample_clip = TextClip(text=text, **text_style)
        
        return VideoClip(
            make_frame,
            duration=duration
        ).with_fps(30)
    
    def apply_to_clip(
        self,
        clip: TextClip,
        start_time: float = 0,
        **kwargs
    ) -> VideoClip:
        """Apply typewriter effect (requires text from kwargs)."""
        text = kwargs.get("text", "")
        if not text:
            return clip
        
        text_style = kwargs.get("text_style", {})
        video_size = kwargs.get("video_size", (1920, 1080))
        
        return self.apply_to_text(
            text=text,
            duration=clip.duration,
            text_style=text_style,
            video_size=video_size,
        )


class ScaleEffect(TextEffect):
    """
    Animated scale/zoom effect for text.
    """
    
    def __init__(
        self,
        config: Optional[TextAnimationConfig] = None,
        scale_from: float = 1.0,
        scale_to: float = 1.2,
        pulse: bool = False,
        pulse_speed: float = 2.0,
    ):
        super().__init__(config)
        self.scale_from = scale_from
        self.scale_to = scale_to
        self.pulse = pulse
        self.pulse_speed = pulse_speed
    
    def apply_to_clip(
        self,
        clip: TextClip,
        start_time: float = 0,
        **kwargs
    ) -> VideoClip:
        """Apply scale animation effect."""
        
        def resize_func(t):
            if self.pulse:
                # Continuous pulsing
                phase = (t * self.pulse_speed * 2 * math.pi)
                scale_range = (self.scale_to - self.scale_from) / 2
                scale = self.scale_from + scale_range + scale_range * math.sin(phase)
            else:
                # One-time scale animation
                anim_duration = self.config.duration
                easing = self.config.get_easing_func()
                
                if t < start_time:
                    progress = 0
                elif t < start_time + anim_duration:
                    progress = (t - start_time) / anim_duration
                    progress = easing(progress)
                else:
                    progress = 1
                
                scale = self.scale_from + (self.scale_to - self.scale_from) * progress
            
            return scale
        
        return clip.resized(resize_func)


# ==================== Text Animator ====================

class TextAnimator:
    """
    Combines multiple text effects for complex animations.
    
    Example:
        animator = TextAnimator()
        animator.add_effect(PopInEffect())
        animator.add_effect(ScaleEffect(pulse=True))
        
        animated_clip = animator.animate_text(
            text="Hello World",
            duration=3.0,
            style={"font": "Arial", "font_size": 48}
        )
    """
    
    def __init__(self):
        self.effects: list[TextEffect] = []
    
    def add_effect(self, effect: TextEffect) -> TextAnimator:
        """Add an effect to the animation chain."""
        self.effects.append(effect)
        return self
    
    def clear_effects(self) -> TextAnimator:
        """Remove all effects."""
        self.effects.clear()
        return self
    
    def animate_text(
        self,
        text: str,
        duration: float,
        style: dict[str, Any],
        video_size: tuple[int, int] = (1920, 1080),
    ) -> VideoClip:
        """
        Create animated text clip with all configured effects.
        
        Args:
            text: Text to animate
            duration: Total duration
            style: TextClip style parameters
            video_size: Size for positioning
            
        Returns:
            Fully animated text clip
        """
        # Create base text clip
        clip = TextClip(
            text=text,
            font=style.get("font", "Arial"),
            font_size=style.get("font_size", 48),
            color=style.get("color", "white"),
            stroke_color=style.get("stroke_color", "black"),
            stroke_width=style.get("stroke_width", 2),
        )
        clip = clip.with_duration(duration)
        
        # Apply each effect in sequence
        for effect in self.effects:
            clip = effect.apply_to_clip(
                clip,
                start_time=0,
                text=text,
                text_style=style,
                video_size=video_size,
            )
        
        return clip
    
    def animate_words(
        self,
        words: list[dict[str, Any]],
        style: dict[str, Any],
        video_size: tuple[int, int] = (1920, 1080),
        stagger_delay: float = 0.1,
    ) -> VideoClip:
        """
        Animate multiple words with staggered timing.
        
        Args:
            words: List of {"word": str, "start": float, "end": float}
            style: Text style
            video_size: Video dimensions
            stagger_delay: Delay between word animations
            
        Returns:
            Composite clip with all animated words
        """
        word_clips = []
        
        for i, word_data in enumerate(words):
            word = word_data["word"]
            start = word_data["start"]
            end = word_data["end"]
            duration = end - start
            
            # Create animated word clip
            word_clip = self.animate_text(
                text=word,
                duration=duration,
                style=style,
                video_size=video_size,
            )
            
            word_clip = word_clip.with_start(start)
            word_clips.append(word_clip)
        
        if not word_clips:
            return ColorClip(size=video_size, color=(0, 0, 0, 0), duration=0.1)
        
        # Calculate total duration
        total_duration = max(w["end"] for w in words)
        
        return CompositeVideoClip(word_clips, size=video_size).with_duration(total_duration)
