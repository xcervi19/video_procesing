"""
Base transition classes and common transitions.

Provides the foundation for video transitions with:
- Direction enum for slide/wipe transitions
- Base transition class
- Common transitions (crossfade, wipe)
- Utility function for applying transitions
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from moviepy import VideoClip, VideoFileClip, CompositeVideoClip, concatenate_videoclips

if TYPE_CHECKING:
    from videopipe.core.context import PipelineContext

logger = logging.getLogger(__name__)


class TransitionDirection(Enum):
    """Direction for directional transitions like slides and wipes."""
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    
    # Diagonal directions
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


# ==================== Easing Functions ====================

def ease_in_out_cubic(t: float) -> float:
    """Smooth ease in/out using cubic interpolation."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_expo(t: float) -> float:
    """Exponential ease out - fast start, slow end."""
    return 1 if t == 1 else 1 - pow(2, -10 * t)


def ease_in_expo(t: float) -> float:
    """Exponential ease in - slow start, fast end."""
    return 0 if t == 0 else pow(2, 10 * t - 10)


def ease_in_out_expo(t: float) -> float:
    """Exponential ease in/out."""
    if t == 0:
        return 0
    if t == 1:
        return 1
    if t < 0.5:
        return pow(2, 20 * t - 10) / 2
    return (2 - pow(2, -20 * t + 10)) / 2


EASING_FUNCTIONS = {
    "linear": lambda t: t,
    "ease_in_out": ease_in_out_cubic,
    "ease_out_expo": ease_out_expo,
    "ease_in_expo": ease_in_expo,
    "ease_in_out_expo": ease_in_out_expo,
}


# ==================== Base Transition ====================

class Transition(ABC):
    """
    Base class for video transitions.
    
    Transitions blend two video clips together over a specified duration.
    """
    
    def __init__(
        self,
        duration: float = 0.5,
        easing: str = "ease_in_out",
    ):
        """
        Initialize transition.
        
        Args:
            duration: Duration of the transition in seconds
            easing: Easing function name
        """
        self.duration = duration
        self.easing = easing
        self._easing_func = EASING_FUNCTIONS.get(easing, EASING_FUNCTIONS["linear"])
    
    @abstractmethod
    def make_frame(
        self,
        clip_a: VideoClip,
        clip_b: VideoClip,
        t: float,
        progress: float,
    ) -> np.ndarray:
        """
        Generate a single transition frame.
        
        Args:
            clip_a: Outgoing clip
            clip_b: Incoming clip
            t: Current time within transition
            progress: Eased progress (0.0 to 1.0)
            
        Returns:
            Frame as numpy array
        """
        pass
    
    def apply(
        self,
        clip_a: VideoClip,
        clip_b: VideoClip,
    ) -> VideoClip:
        """
        Apply transition between two clips.
        
        The transition overlaps the end of clip_a with the start of clip_b.
        
        Args:
            clip_a: Outgoing clip
            clip_b: Incoming clip
            
        Returns:
            Single clip with transition applied
        """
        # Ensure clips have same size
        if (clip_a.w, clip_a.h) != (clip_b.w, clip_b.h):
            logger.warning("Clips have different sizes, resizing clip_b to match clip_a")
            clip_b = clip_b.resized((clip_a.w, clip_a.h))
        
        size = (clip_a.w, clip_a.h)
        fps = clip_a.fps or clip_b.fps or 30
        duration = self.duration
        
        def make_frame(t):
            # Calculate eased progress
            progress = self._easing_func(min(1.0, t / duration))
            
            return self.make_frame(clip_a, clip_b, t, progress)
        
        # Create transition clip
        transition_clip = VideoClip(make_frame, duration=duration)
        transition_clip = transition_clip.with_fps(fps)
        
        # Assemble: clip_a (minus overlap) + transition + clip_b (minus overlap)
        a_duration = clip_a.duration - duration
        b_start = duration
        
        if a_duration > 0:
            clip_a_part = clip_a.subclipped(0, a_duration)
        else:
            clip_a_part = None
        
        if b_start < clip_b.duration:
            clip_b_part = clip_b.subclipped(b_start)
        else:
            clip_b_part = None
        
        # Build final clip
        clips = []
        if clip_a_part:
            clips.append(clip_a_part)
        clips.append(transition_clip)
        if clip_b_part:
            clips.append(clip_b_part)
        
        return concatenate_videoclips(clips, method="compose")


# ==================== Common Transitions ====================

class CrossfadeTransition(Transition):
    """
    Simple crossfade (dissolve) transition.
    
    Clip A fades out while clip B fades in.
    """
    
    def make_frame(
        self,
        clip_a: VideoClip,
        clip_b: VideoClip,
        t: float,
        progress: float,
    ) -> np.ndarray:
        # Get frames from both clips
        # For clip_a, we're at the end
        t_a = clip_a.duration - self.duration + t
        frame_a = clip_a.get_frame(t_a)
        
        # For clip_b, we're at the start
        t_b = t
        frame_b = clip_b.get_frame(t_b)
        
        # Blend frames
        blended = (1 - progress) * frame_a + progress * frame_b
        
        return blended.astype(np.uint8)


class WipeTransition(Transition):
    """
    Wipe transition - reveals clip B with a moving edge.
    
    The wipe can move in any direction.
    """
    
    def __init__(
        self,
        duration: float = 0.5,
        easing: str = "ease_in_out",
        direction: TransitionDirection = TransitionDirection.LEFT,
        softness: int = 10,
    ):
        """
        Initialize wipe transition.
        
        Args:
            duration: Transition duration
            easing: Easing function
            direction: Direction of the wipe
            softness: Softness of the wipe edge in pixels
        """
        super().__init__(duration, easing)
        self.direction = direction
        self.softness = softness
    
    def make_frame(
        self,
        clip_a: VideoClip,
        clip_b: VideoClip,
        t: float,
        progress: float,
    ) -> np.ndarray:
        # Get frames
        t_a = clip_a.duration - self.duration + t
        frame_a = clip_a.get_frame(t_a)
        
        t_b = t
        frame_b = clip_b.get_frame(t_b)
        
        height, width = frame_a.shape[:2]
        
        # Create wipe mask based on direction
        if self.direction == TransitionDirection.LEFT:
            # Wipe from right to left (B enters from right)
            threshold = int(width * (1 - progress))
            x_coords = np.arange(width)
            mask = 1 - np.clip((x_coords - threshold + self.softness) / (self.softness * 2), 0, 1)
            mask = mask[np.newaxis, :, np.newaxis]
            
        elif self.direction == TransitionDirection.RIGHT:
            # Wipe from left to right (B enters from left)
            threshold = int(width * progress)
            x_coords = np.arange(width)
            mask = np.clip((x_coords - threshold + self.softness) / (self.softness * 2), 0, 1)
            mask = mask[np.newaxis, :, np.newaxis]
            
        elif self.direction == TransitionDirection.UP:
            # Wipe from bottom to top (B enters from bottom)
            threshold = int(height * (1 - progress))
            y_coords = np.arange(height)
            mask = 1 - np.clip((y_coords - threshold + self.softness) / (self.softness * 2), 0, 1)
            mask = mask[:, np.newaxis, np.newaxis]
            
        elif self.direction == TransitionDirection.DOWN:
            # Wipe from top to bottom (B enters from top)
            threshold = int(height * progress)
            y_coords = np.arange(height)
            mask = np.clip((y_coords - threshold + self.softness) / (self.softness * 2), 0, 1)
            mask = mask[:, np.newaxis, np.newaxis]
            
        else:
            # Default to crossfade for unsupported directions
            mask = progress
        
        # Apply mask
        blended = frame_a * mask + frame_b * (1 - mask)
        
        return blended.astype(np.uint8)


# ==================== Utility Functions ====================

def apply_transition(
    clips: list[VideoClip],
    transition: Transition,
) -> VideoClip:
    """
    Apply a transition between all consecutive clips in a list.
    
    Args:
        clips: List of video clips
        transition: Transition to apply between each pair
        
    Returns:
        Single concatenated clip with transitions
    """
    if not clips:
        raise ValueError("No clips provided")
    
    if len(clips) == 1:
        return clips[0]
    
    result = clips[0]
    
    for clip in clips[1:]:
        result = transition.apply(result, clip)
    
    return result


def create_transition(
    transition_type: str,
    **kwargs
) -> Transition:
    """
    Factory function to create transitions by name.
    
    Args:
        transition_type: Type of transition ('crossfade', 'wipe', 'slide')
        **kwargs: Transition-specific parameters
        
    Returns:
        Transition instance
    """
    transitions = {
        "crossfade": CrossfadeTransition,
        "wipe": WipeTransition,
    }
    
    # Import slide transitions
    try:
        from videopipe.transitions.slide import SlideTransition, QuickSlideTransition
        transitions["slide"] = SlideTransition
        transitions["quick_slide"] = QuickSlideTransition
    except ImportError:
        pass
    
    if transition_type not in transitions:
        available = ", ".join(transitions.keys())
        raise ValueError(f"Unknown transition type: {transition_type}. Available: {available}")
    
    return transitions[transition_type](**kwargs)
