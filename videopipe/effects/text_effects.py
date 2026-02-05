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

# Import neon effects for combined animation+glow
from videopipe.effects.neon import NeonGlowEffect, NeonConfig

logger = logging.getLogger(__name__)

# #region agent log
import json as _json_debug_te
import time as _time_debug_te
import os as _os_debug_te
_DEBUG_LOG_PATH_TE = "/Users/karel.cervicek/Documents/projects/video_procesing/.cursor/debug.log"
_os_debug_te.makedirs(_os_debug_te.path.dirname(_DEBUG_LOG_PATH_TE), exist_ok=True)
def _dbg_te(hyp, loc, msg, data):
    try:
        with open(_DEBUG_LOG_PATH_TE, "a") as f: f.write(_json_debug_te.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": _time_debug_te.time()}) + "\n")
    except Exception as e:
        print(f"[DEBUG LOG ERROR] {e}")
# #endregion


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


def ease_in_quad(t: float) -> float:
    """Smooth ease in using quadratic."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Smooth ease out using quadratic."""
    return 1 - (1 - t) * (1 - t)


def ease_in_out_quad(t: float) -> float:
    """Smooth ease in/out using quadratic."""
    if t < 0.5:
        return 2 * t * t
    else:
        return 1 - pow(-2 * t + 2, 2) / 2


def ease_out_expo(t: float) -> float:
    """Exponential ease out - very smooth professional feel."""
    return 1 if t == 1 else 1 - pow(2, -10 * t)


EASING_FUNCTIONS = {
    "linear": lambda t: t,
    "ease_in": ease_in_quad,
    "ease_out": ease_out_quad,
    "ease_in_out": ease_in_out_cubic,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_out_expo": ease_out_expo,
    "ease_out_bounce": ease_out_bounce,
    "ease_out_elastic": ease_out_elastic,
    "ease_out_back": ease_out_back,
}


# ==================== Professional Animation Constants ====================
# Minimum durations for animations to look professional and smooth

class AnimationTiming:
    """
    Professional animation timing constants.
    Based on motion design best practices for smooth, polished animations.
    """
    # Minimum durations (seconds) - below these, animations look choppy
    MIN_FADE_DURATION = 0.15
    MIN_TYPEWRITER_CHAR_DELAY = 0.03  # 33 chars/sec max
    MIN_SCALE_DURATION = 0.2
    MIN_SLIDE_DURATION = 0.25
    
    # Recommended durations for professional look
    RECOMMENDED_FADE_IN = 0.3
    RECOMMENDED_FADE_OUT = 0.25
    RECOMMENDED_TYPEWRITER_SPEED = 15  # chars per second
    RECOMMENDED_SCALE_DURATION = 0.4
    RECOMMENDED_HOLD_BUFFER = 0.1  # Extra time after animation completes
    
    # Frame rates
    SMOOTH_FPS = 30  # Minimum for smooth animation
    PREMIUM_FPS = 60  # For extra smooth feel
    
    @classmethod
    def calculate_typewriter_duration(cls, text: str, chars_per_second: float = None) -> float:
        """Calculate minimum duration for typewriter effect to look smooth."""
        cps = chars_per_second or cls.RECOMMENDED_TYPEWRITER_SPEED
        cps = min(cps, 1.0 / cls.MIN_TYPEWRITER_CHAR_DELAY)  # Cap at max speed
        return len(text) / cps
    
    @classmethod
    def ensure_minimum_duration(cls, requested: float, animation_type: str) -> float:
        """Ensure duration meets minimum for smooth animation."""
        minimums = {
            "fade": cls.MIN_FADE_DURATION,
            "typewriter": cls.MIN_TYPEWRITER_CHAR_DELAY * 5,  # At least 5 chars worth
            "scale": cls.MIN_SCALE_DURATION,
            "slide": cls.MIN_SLIDE_DURATION,
        }
        minimum = minimums.get(animation_type, 0.1)
        return max(requested, minimum)


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


# ==================== Professional Animation System ====================

@dataclass
class ProfessionalAnimationConfig:
    """
    Configuration for professional text animations.
    
    Designed to create smooth, polished animations that feel
    high-quality and cinematic.
    """
    # Animation type
    animation_type: str = "typewriter"  # typewriter, pop_in, fade, scale, slide
    
    # Entrance animation
    entrance_duration: float = 0.4
    entrance_easing: str = "ease_out_expo"
    
    # Exit animation  
    exit_duration: float = 0.3
    exit_easing: str = "ease_in_out_quad"
    fade_out: bool = True
    
    # Typewriter specific
    chars_per_second: float = 15.0
    cursor_blink: bool = False  # Disabled for cleaner look
    cursor_char: str = ""  # No cursor for professional look
    
    # Scale specific
    scale_start: float = 0.8
    scale_end: float = 1.0
    
    # Slide specific
    slide_direction: str = "up"  # up, down, left, right
    slide_distance: int = 50  # pixels
    
    # Common settings
    fps: int = 30
    
    def __post_init__(self):
        """Ensure minimum durations for smooth animation."""
        self.entrance_duration = AnimationTiming.ensure_minimum_duration(
            self.entrance_duration, self.animation_type
        )
        self.exit_duration = AnimationTiming.ensure_minimum_duration(
            self.exit_duration, "fade"
        )


class ProfessionalTextAnimation:
    """
    State-of-the-art text animation system.
    
    Creates smooth, cinematic text animations with:
    - Professional entrance animations (typewriter, pop, slide)
    - Smooth hold period
    - Graceful exit with fade out
    - Proper easing for natural motion
    
    Example:
        anim = ProfessionalTextAnimation(ProfessionalAnimationConfig(
            animation_type="typewriter",
            entrance_duration=0.5,
        ))
        clip = anim.create_animated_text(
            text="Hello World",
            total_duration=3.0,
            font="Bebas Neue",
            font_size=72,
            color="white",
        )
    """
    
    def __init__(self, config: Optional[ProfessionalAnimationConfig] = None):
        self.config = config or ProfessionalAnimationConfig()
    
    def create_animated_text(
        self,
        text: str,
        total_duration: float,
        font: str = "Arial",
        font_size: int = 48,
        color: str = "white",
        stroke_color: Optional[str] = None,
        stroke_width: int = 0,
        bg_color: Optional[str] = None,
        neon_config: Optional[NeonConfig] = None,
    ) -> VideoClip:
        """
        Create professionally animated text clip.
        
        The animation is structured as:
        1. Entrance phase (typewriter/pop/slide)
        2. Hold phase (text fully visible)
        3. Exit phase (fade out)
        
        Args:
            text: Text to animate
            total_duration: Total clip duration
            font: Font name or path
            font_size: Size in pixels
            color: Text color
            stroke_color: Outline color (optional)
            stroke_width: Outline width
            bg_color: Background color (optional)
            neon_config: Optional NeonConfig for neon glow effect
            
        Returns:
            Animated VideoClip
        """
        # #region agent log
        _dbg_te("NEON_ANIM", "create_animated_text:entry", "create_animated_text called", {"text": text, "neon_config_is_none": neon_config is None, "animation_type": self.config.animation_type})
        # #endregion
        
        config = self.config
        
        # Calculate phase durations
        entrance_duration = self._calculate_entrance_duration(text)
        exit_duration = config.exit_duration if config.fade_out else 0
        hold_duration = max(0, total_duration - entrance_duration - exit_duration)
        
        # Ensure minimum total duration
        if total_duration < entrance_duration + exit_duration:
            # Scale down animations proportionally
            scale = total_duration / (entrance_duration + exit_duration + 0.1)
            entrance_duration *= scale
            exit_duration *= scale
            hold_duration = max(0.1, total_duration - entrance_duration - exit_duration)
        
        fps = config.fps
        
        # Get easing functions
        entrance_easing = EASING_FUNCTIONS.get(config.entrance_easing, ease_out_expo)
        exit_easing = EASING_FUNCTIONS.get(config.exit_easing, ease_in_out_quad)
        
        # Create the animation based on type
        if config.animation_type == "typewriter":
            return self._create_typewriter_animation(
                text=text,
                total_duration=total_duration,
                entrance_duration=entrance_duration,
                hold_duration=hold_duration,
                exit_duration=exit_duration,
                entrance_easing=entrance_easing,
                exit_easing=exit_easing,
                font=font,
                font_size=font_size,
                color=color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                fps=fps,
                neon_config=neon_config,
            )
        elif config.animation_type == "pop_in":
            return self._create_pop_animation(
                text=text,
                total_duration=total_duration,
                entrance_duration=entrance_duration,
                exit_duration=exit_duration,
                entrance_easing=entrance_easing,
                exit_easing=exit_easing,
                font=font,
                font_size=font_size,
                color=color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                fps=fps,
                neon_config=neon_config,
            )
        elif config.animation_type == "fade":
            return self._create_fade_animation(
                text=text,
                total_duration=total_duration,
                entrance_duration=entrance_duration,
                exit_duration=exit_duration,
                entrance_easing=entrance_easing,
                exit_easing=exit_easing,
                font=font,
                font_size=font_size,
                color=color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                fps=fps,
                neon_config=neon_config,
            )
        else:
            # Default to simple fade
            return self._create_fade_animation(
                text=text,
                total_duration=total_duration,
                entrance_duration=entrance_duration,
                exit_duration=exit_duration,
                entrance_easing=entrance_easing,
                exit_easing=exit_easing,
                font=font,
                font_size=font_size,
                color=color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                fps=fps,
                neon_config=neon_config,
            )
    
    def _calculate_entrance_duration(self, text: str) -> float:
        """Calculate entrance duration based on animation type and text."""
        config = self.config
        
        if config.animation_type == "typewriter":
            # Typewriter needs time for each character
            char_duration = AnimationTiming.calculate_typewriter_duration(
                text, config.chars_per_second
            )
            return max(config.entrance_duration, char_duration)
        else:
            return config.entrance_duration
    
    def _create_typewriter_animation(
        self,
        text: str,
        total_duration: float,
        entrance_duration: float,
        hold_duration: float,
        exit_duration: float,
        entrance_easing: Callable,
        exit_easing: Callable,
        font: str,
        font_size: int,
        color: str,
        stroke_color: Optional[str],
        stroke_width: int,
        fps: int,
        neon_config: Optional[NeonConfig] = None,
    ) -> VideoClip:
        """
        Create smooth typewriter animation with optional neon glow.
        
        Characters appear with smooth timing and optional fade-in per character.
        Exit fades out the complete text smoothly.
        """
        from PIL import Image, ImageDraw, ImageFont
        
        chars = list(text)
        num_chars = len(chars)
        
        # #region agent log
        _dbg_te("NEON_TW", "typewriter:neon_config_check", "Typewriter neon_config received", {"neon_config_is_none": neon_config is None, "text": text})
        # #endregion
        
        # If neon config provided, use neon rendering for typewriter
        if neon_config is not None:
            # #region agent log
            _dbg_te("NEON_TW", "typewriter:using_neon", "USING NEON GLOW for typewriter", {"font": font, "font_size": font_size})
            # #endregion
            neon_config.font = font
            neon_config.font_size = font_size
            neon_effect = NeonGlowEffect(config=neon_config)
            
            # Pre-render full text to get canvas size
            full_frame = neon_effect.create_neon_frame(text, 1.0)
            canvas_h, canvas_w = full_frame.shape[:2]
            
            # Cache neon frames for visible text states (optimization)
            neon_cache = {}
            
            def get_neon_frame_for_text(visible_text: str) -> np.ndarray:
                if visible_text not in neon_cache:
                    if visible_text:
                        neon_cache[visible_text] = neon_effect.create_neon_frame(visible_text, 1.0)
                    else:
                        neon_cache[visible_text] = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
                return neon_cache[visible_text]
            
            def make_neon_frame(t):
                # Calculate which characters to show
                if t < entrance_duration and num_chars > 0:
                    linear_progress = t / entrance_duration
                    eased_progress = entrance_easing(linear_progress)
                    visible_chars = int(eased_progress * num_chars)
                    char_progress = eased_progress * num_chars
                    partial_char_alpha = char_progress - visible_chars
                else:
                    visible_chars = num_chars
                    partial_char_alpha = 1.0
                
                # Calculate exit fade
                exit_start = total_duration - exit_duration
                if t >= exit_start and exit_duration > 0:
                    exit_progress = (t - exit_start) / exit_duration
                    exit_alpha = 1 - exit_easing(exit_progress)
                else:
                    exit_alpha = 1.0
                
                # Get neon frame for visible characters
                visible_text = "".join(chars[:visible_chars]) if visible_chars > 0 else ""
                frame = get_neon_frame_for_text(visible_text)
                
                # Create result with proper canvas size
                result = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
                
                if visible_text:
                    # Copy frame, it may be smaller than canvas
                    fh, fw = frame.shape[:2]
                    result[:fh, :fw] = frame.copy()
                
                # Apply exit alpha
                if exit_alpha < 1.0:
                    result[:, :, 3] = (result[:, :, 3] * exit_alpha).astype(np.uint8)
                
                # For smooth typewriter, blend in partial next character
                if visible_chars < num_chars and partial_char_alpha > 0:
                    next_text = "".join(chars[:visible_chars + 1])
                    next_frame = get_neon_frame_for_text(next_text)
                    
                    # Blend between current and next frame based on partial alpha
                    nfh, nfw = next_frame.shape[:2]
                    blend_region = np.zeros_like(result)
                    blend_region[:nfh, :nfw] = next_frame
                    
                    # Only blend the difference (the new character area)
                    diff_alpha = (blend_region[:, :, 3].astype(float) - result[:, :, 3].astype(float))
                    diff_alpha = np.clip(diff_alpha * partial_char_alpha, 0, 255)
                    
                    # Add the faded-in new character
                    for c in range(3):
                        mask = diff_alpha > 0
                        result[:, :, c][mask] = np.clip(
                            result[:, :, c][mask] + (blend_region[:, :, c][mask] * (diff_alpha[mask] / 255)),
                            0, 255
                        ).astype(np.uint8)
                    result[:, :, 3] = np.clip(result[:, :, 3].astype(float) + diff_alpha, 0, 255).astype(np.uint8)
                    
                    # Apply exit alpha to blended result
                    if exit_alpha < 1.0:
                        result[:, :, 3] = (result[:, :, 3] * exit_alpha).astype(np.uint8)
                
                return result
            
            clip = VideoClip(make_neon_frame, duration=total_duration)
            clip = clip.with_fps(fps)
            return clip
        
        # Original plain text rendering
        # Load font for measurements
        try:
            pil_font = ImageFont.truetype(font, font_size)
        except OSError:
            pil_font = ImageFont.load_default()
        
        # Measure full text
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=pil_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Add padding
        padding = max(20, font_size // 2)
        canvas_w = text_width + padding * 2
        canvas_h = text_height + padding * 2
        
        # Parse colors
        def parse_color(c):
            if c is None:
                return None
            if isinstance(c, str) and c.startswith('#'):
                c = c.lstrip('#')
                return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
            return c
        
        text_color = parse_color(color) or (255, 255, 255)
        outline_color = parse_color(stroke_color)
        
        def make_frame(t):
            # Create canvas
            img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Calculate which characters to show
            if t < entrance_duration and num_chars > 0:
                # Typewriter phase - use eased progress
                linear_progress = t / entrance_duration
                eased_progress = entrance_easing(linear_progress)
                visible_chars = int(eased_progress * num_chars)
                
                # Calculate per-character fade for smooth appearance
                char_progress = eased_progress * num_chars
                partial_char_alpha = char_progress - visible_chars
            else:
                visible_chars = num_chars
                partial_char_alpha = 1.0
            
            # Calculate exit fade
            exit_start = total_duration - exit_duration
            if t >= exit_start and exit_duration > 0:
                exit_progress = (t - exit_start) / exit_duration
                exit_alpha = 1 - exit_easing(exit_progress)
            else:
                exit_alpha = 1.0
            
            # Draw visible text
            if visible_chars > 0:
                visible_text = "".join(chars[:visible_chars])
                
                # Apply exit alpha
                final_alpha = int(255 * exit_alpha)
                
                # Draw with stroke if specified
                if outline_color and stroke_width > 0:
                    # Draw stroke
                    for dx in range(-stroke_width, stroke_width + 1):
                        for dy in range(-stroke_width, stroke_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text(
                                    (padding + dx, padding + dy),
                                    visible_text,
                                    font=pil_font,
                                    fill=(*outline_color, final_alpha)
                                )
                
                # Draw main text
                draw.text(
                    (padding, padding),
                    visible_text,
                    font=pil_font,
                    fill=(*text_color, final_alpha)
                )
                
                # Draw partial character with fade (smooth character appearance)
                if visible_chars < num_chars and partial_char_alpha > 0:
                    next_char = chars[visible_chars]
                    # Measure position of next character
                    visible_bbox = draw.textbbox((padding, padding), visible_text, font=pil_font)
                    next_x = visible_bbox[2]
                    
                    char_alpha = int(255 * partial_char_alpha * exit_alpha)
                    
                    if outline_color and stroke_width > 0:
                        for dx in range(-stroke_width, stroke_width + 1):
                            for dy in range(-stroke_width, stroke_width + 1):
                                if dx != 0 or dy != 0:
                                    draw.text(
                                        (next_x + dx, padding + dy),
                                        next_char,
                                        font=pil_font,
                                        fill=(*outline_color, char_alpha)
                                    )
                    
                    draw.text(
                        (next_x, padding),
                        next_char,
                        font=pil_font,
                        fill=(*text_color, char_alpha)
                    )
            
            return np.array(img)
        
        clip = VideoClip(make_frame, duration=total_duration)
        clip = clip.with_fps(fps)
        
        return clip
    
    def _create_pop_animation(
        self,
        text: str,
        total_duration: float,
        entrance_duration: float,
        exit_duration: float,
        entrance_easing: Callable,
        exit_easing: Callable,
        font: str,
        font_size: int,
        color: str,
        stroke_color: Optional[str],
        stroke_width: int,
        fps: int,
        neon_config: Optional[NeonConfig] = None,
    ) -> VideoClip:
        """Create pop-in animation with scale and fade, with optional neon glow."""
        from PIL import Image, ImageDraw, ImageFont
        
        config = self.config
        
        # #region agent log
        _dbg_te("NEON_POP", "pop:neon_config_check", "Pop neon_config received", {"neon_config_is_none": neon_config is None, "text": text})
        # #endregion
        
        # If neon config provided, use neon rendering with scale
        if neon_config is not None:
            # #region agent log
            _dbg_te("NEON_POP", "pop:using_neon", "USING NEON GLOW for pop", {"font": font, "font_size": font_size})
            # #endregion
            # Update neon config with current font settings
            neon_config.font = font
            neon_config.font_size = font_size
            neon_effect = NeonGlowEffect(config=neon_config)
            
            # Pre-render neon frame at full size to get base dimensions
            base_frame = neon_effect.create_neon_frame(text, 1.0)
            base_h, base_w = base_frame.shape[:2]
            
            # Add padding for scale overshoot
            padding = max(40, font_size)
            canvas_w = base_w + padding * 2
            canvas_h = base_h + padding * 2
            
            def make_neon_frame(t):
                # Calculate scale
                if t < entrance_duration:
                    progress = entrance_easing(t / entrance_duration)
                    scale = config.scale_start + (config.scale_end - config.scale_start) * progress
                    alpha = progress
                else:
                    scale = config.scale_end
                    alpha = 1.0
                
                # Exit fade
                exit_start = total_duration - exit_duration
                if t >= exit_start and exit_duration > 0:
                    exit_progress = (t - exit_start) / exit_duration
                    alpha *= (1 - exit_easing(exit_progress))
                
                if scale <= 0 or alpha <= 0:
                    return np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
                
                # Create neon frame at scaled font size
                scaled_neon_cfg = NeonConfig(
                    color=neon_config.color,
                    glow_color=neon_config.glow_color,
                    glow_intensity=neon_config.glow_intensity,
                    glow_radius=max(1, int(neon_config.glow_radius * scale)),
                    glow_layers=neon_config.glow_layers,
                    pulse=False,  # Disable pulse during scale animation
                    font=font,
                    font_size=max(8, int(font_size * scale)),
                )
                scaled_effect = NeonGlowEffect(config=scaled_neon_cfg)
                frame = scaled_effect.create_neon_frame(text, 1.0)
                
                # Center on canvas
                result = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
                fh, fw = frame.shape[:2]
                x = (canvas_w - fw) // 2
                y = (canvas_h - fh) // 2
                
                # Clamp to valid range
                x = max(0, x)
                y = max(0, y)
                end_x = min(canvas_w, x + fw)
                end_y = min(canvas_h, y + fh)
                src_end_x = end_x - x
                src_end_y = end_y - y
                
                result[y:end_y, x:end_x] = frame[:src_end_y, :src_end_x]
                
                # Apply fade alpha
                if alpha < 1.0:
                    result[:, :, 3] = (result[:, :, 3] * alpha).astype(np.uint8)
                
                return result
            
            clip = VideoClip(make_neon_frame, duration=total_duration)
            clip = clip.with_fps(fps)
            return clip
        
        # Original plain text rendering
        # Load font
        try:
            pil_font = ImageFont.truetype(font, font_size)
        except OSError:
            pil_font = ImageFont.load_default()
        
        # Measure text
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=pil_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        padding = max(40, font_size)  # Extra padding for scale overshoot
        canvas_w = text_width + padding * 2
        canvas_h = text_height + padding * 2
        
        def parse_color(c):
            if c is None:
                return None
            if isinstance(c, str) and c.startswith('#'):
                c = c.lstrip('#')
                return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
            return c
        
        text_color = parse_color(color) or (255, 255, 255)
        outline_color = parse_color(stroke_color)
        
        def make_frame(t):
            # Calculate scale
            if t < entrance_duration:
                progress = entrance_easing(t / entrance_duration)
                scale = config.scale_start + (config.scale_end - config.scale_start) * progress
                alpha = progress  # Fade in with scale
            else:
                scale = config.scale_end
                alpha = 1.0
            
            # Exit fade
            exit_start = total_duration - exit_duration
            if t >= exit_start and exit_duration > 0:
                exit_progress = (t - exit_start) / exit_duration
                alpha *= (1 - exit_easing(exit_progress))
            
            if scale <= 0 or alpha <= 0:
                return np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            
            # Create scaled text
            scaled_size = int(font_size * scale)
            try:
                scaled_font = ImageFont.truetype(font, scaled_size)
            except OSError:
                scaled_font = pil_font
            
            img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Center the scaled text
            scaled_bbox = draw.textbbox((0, 0), text, font=scaled_font)
            scaled_w = scaled_bbox[2] - scaled_bbox[0]
            scaled_h = scaled_bbox[3] - scaled_bbox[1]
            x = (canvas_w - scaled_w) // 2
            y = (canvas_h - scaled_h) // 2
            
            final_alpha = int(255 * alpha)
            
            if outline_color and stroke_width > 0:
                scaled_stroke = max(1, int(stroke_width * scale))
                for dx in range(-scaled_stroke, scaled_stroke + 1):
                    for dy in range(-scaled_stroke, scaled_stroke + 1):
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), text, font=scaled_font, 
                                      fill=(*outline_color, final_alpha))
            
            draw.text((x, y), text, font=scaled_font, fill=(*text_color, final_alpha))
            
            return np.array(img)
        
        clip = VideoClip(make_frame, duration=total_duration)
        clip = clip.with_fps(fps)
        
        return clip
    
    def _create_fade_animation(
        self,
        text: str,
        total_duration: float,
        entrance_duration: float,
        exit_duration: float,
        entrance_easing: Callable,
        exit_easing: Callable,
        font: str,
        font_size: int,
        color: str,
        stroke_color: Optional[str],
        stroke_width: int,
        fps: int,
        neon_config: Optional[NeonConfig] = None,
    ) -> VideoClip:
        """Create simple fade in/out animation with optional neon glow."""
        from PIL import Image, ImageDraw, ImageFont
        
        # If neon config provided, use neon rendering
        if neon_config is not None:
            # Update neon config with current font settings
            neon_config.font = font
            neon_config.font_size = font_size
            neon_effect = NeonGlowEffect(config=neon_config)
            
            # Pre-render neon frame at full intensity to get size
            base_frame = neon_effect.create_neon_frame(text, 1.0)
            canvas_h, canvas_w = base_frame.shape[:2]
            
            def make_neon_frame(t):
                # Entrance fade
                if t < entrance_duration:
                    alpha = entrance_easing(t / entrance_duration)
                else:
                    alpha = 1.0
                
                # Exit fade
                exit_start = total_duration - exit_duration
                if t >= exit_start and exit_duration > 0:
                    exit_progress = (t - exit_start) / exit_duration
                    alpha *= (1 - exit_easing(exit_progress))
                
                # Create neon frame with pulsing (if enabled)
                frame = neon_effect.create_neon_frame(text, 1.0)
                
                # Apply fade alpha to the frame
                if alpha < 1.0:
                    frame = frame.copy()
                    frame[:, :, 3] = (frame[:, :, 3] * alpha).astype(np.uint8)
                
                return frame
            
            clip = VideoClip(make_neon_frame, duration=total_duration)
            clip = clip.with_fps(fps)
            return clip
        
        # Original plain text rendering
        try:
            pil_font = ImageFont.truetype(font, font_size)
        except OSError:
            pil_font = ImageFont.load_default()
        
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=pil_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        padding = max(20, font_size // 2)
        canvas_w = text_width + padding * 2
        canvas_h = text_height + padding * 2
        
        def parse_color(c):
            if c is None:
                return None
            if isinstance(c, str) and c.startswith('#'):
                c = c.lstrip('#')
                return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
            return c
        
        text_color = parse_color(color) or (255, 255, 255)
        outline_color = parse_color(stroke_color)
        
        def make_frame(t):
            # Entrance fade
            if t < entrance_duration:
                alpha = entrance_easing(t / entrance_duration)
            else:
                alpha = 1.0
            
            # Exit fade
            exit_start = total_duration - exit_duration
            if t >= exit_start and exit_duration > 0:
                exit_progress = (t - exit_start) / exit_duration
                alpha *= (1 - exit_easing(exit_progress))
            
            img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            final_alpha = int(255 * alpha)
            
            if outline_color and stroke_width > 0:
                for dx in range(-stroke_width, stroke_width + 1):
                    for dy in range(-stroke_width, stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((padding + dx, padding + dy), text, 
                                      font=pil_font, fill=(*outline_color, final_alpha))
            
            draw.text((padding, padding), text, font=pil_font, fill=(*text_color, final_alpha))
            
            return np.array(img)
        
        clip = VideoClip(make_frame, duration=total_duration)
        clip = clip.with_fps(fps)
        
        return clip


# ==================== Convenience Functions ====================

def create_typewriter_text(
    text: str,
    duration: float,
    font: str = "Arial",
    font_size: int = 48,
    color: str = "white",
    chars_per_second: float = 15.0,
    fade_out: bool = True,
) -> VideoClip:
    """
    Convenience function to create typewriter animated text.
    
    Args:
        text: Text to animate
        duration: Total duration
        font: Font name or path
        font_size: Size in pixels
        color: Text color
        chars_per_second: Typing speed
        fade_out: Whether to fade out at end
        
    Returns:
        Animated VideoClip
    """
    config = ProfessionalAnimationConfig(
        animation_type="typewriter",
        chars_per_second=chars_per_second,
        fade_out=fade_out,
    )
    animator = ProfessionalTextAnimation(config)
    return animator.create_animated_text(
        text=text,
        total_duration=duration,
        font=font,
        font_size=font_size,
        color=color,
    )


def create_pop_in_text(
    text: str,
    duration: float,
    font: str = "Arial",
    font_size: int = 48,
    color: str = "white",
    entrance_duration: float = 0.4,
    fade_out: bool = True,
) -> VideoClip:
    """
    Convenience function to create pop-in animated text.
    """
    config = ProfessionalAnimationConfig(
        animation_type="pop_in",
        entrance_duration=entrance_duration,
        entrance_easing="ease_out_back",
        scale_start=0.5,
        scale_end=1.0,
        fade_out=fade_out,
    )
    animator = ProfessionalTextAnimation(config)
    return animator.create_animated_text(
        text=text,
        total_duration=duration,
        font=font,
        font_size=font_size,
        color=color,
    )
