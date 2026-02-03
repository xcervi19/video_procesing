"""
Professional slide transitions for video editing.

Provides smooth, professional-quality slide transitions with:
- Multiple directions
- Motion blur option
- Configurable easing
- 3D perspective option
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from moviepy import VideoClip, CompositeVideoClip, concatenate_videoclips

from videopipe.transitions.base import (
    Transition,
    TransitionDirection,
    EASING_FUNCTIONS,
    ease_in_out_cubic,
)

logger = logging.getLogger(__name__)


@dataclass
class SlideConfig:
    """Configuration for slide transitions."""
    direction: TransitionDirection = TransitionDirection.LEFT
    duration: float = 0.5
    easing: str = "ease_in_out_expo"
    
    # Motion blur for smoother, more professional look
    motion_blur: bool = True
    motion_blur_samples: int = 5
    
    # 3D perspective effect (subtle depth)
    perspective: bool = False
    perspective_strength: float = 0.1
    
    # Overlap - how much clip B overlaps during slide
    # 0 = clips slide adjacent, 1 = full overlap
    overlap: float = 0.0
    
    # Shadow effect between clips
    shadow: bool = True
    shadow_width: int = 30
    shadow_opacity: float = 0.3


class SlideTransition(Transition):
    """
    Professional slide transition.
    
    Creates a smooth sliding effect where one clip pushes another
    off screen. Supports various directions, motion blur, and
    optional 3D perspective effects.
    
    Example:
        transition = SlideTransition(
            direction=TransitionDirection.LEFT,
            duration=0.5,
            motion_blur=True,
        )
        result = transition.apply(clip_a, clip_b)
    """
    
    def __init__(
        self,
        direction: TransitionDirection = TransitionDirection.LEFT,
        duration: float = 0.5,
        easing: str = "ease_in_out_expo",
        motion_blur: bool = True,
        motion_blur_samples: int = 5,
        perspective: bool = False,
        shadow: bool = True,
        shadow_width: int = 30,
        shadow_opacity: float = 0.3,
    ):
        super().__init__(duration, easing)
        self.direction = direction
        self.motion_blur = motion_blur
        self.motion_blur_samples = motion_blur_samples
        self.perspective = perspective
        self.shadow = shadow
        self.shadow_width = shadow_width
        self.shadow_opacity = shadow_opacity
    
    @classmethod
    def from_config(cls, config: SlideConfig) -> SlideTransition:
        """Create transition from config object."""
        return cls(
            direction=config.direction,
            duration=config.duration,
            easing=config.easing,
            motion_blur=config.motion_blur,
            motion_blur_samples=config.motion_blur_samples,
            perspective=config.perspective,
            shadow=config.shadow,
            shadow_width=config.shadow_width,
            shadow_opacity=config.shadow_opacity,
        )
    
    def _get_direction_vector(self) -> Tuple[int, int]:
        """Get the slide direction as (dx, dy) vector."""
        vectors = {
            TransitionDirection.LEFT: (-1, 0),
            TransitionDirection.RIGHT: (1, 0),
            TransitionDirection.UP: (0, -1),
            TransitionDirection.DOWN: (0, 1),
            TransitionDirection.TOP_LEFT: (-1, -1),
            TransitionDirection.TOP_RIGHT: (1, -1),
            TransitionDirection.BOTTOM_LEFT: (-1, 1),
            TransitionDirection.BOTTOM_RIGHT: (1, 1),
        }
        return vectors.get(self.direction, (-1, 0))
    
    def _calculate_positions(
        self,
        progress: float,
        width: int,
        height: int,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Calculate positions for both clips based on progress.
        
        Returns:
            Tuple of (clip_a_position, clip_b_position)
        """
        dx, dy = self._get_direction_vector()
        
        # Clip A moves out in the direction
        # Clip B moves in from the opposite direction
        
        a_x = int(dx * progress * width)
        a_y = int(dy * progress * height)
        
        # Clip B starts off-screen and slides in
        b_x = int(-dx * (1 - progress) * width)
        b_y = int(-dy * (1 - progress) * height)
        
        return (a_x, a_y), (b_x, b_y)
    
    def _create_shadow_gradient(
        self,
        width: int,
        height: int,
        direction: TransitionDirection,
    ) -> np.ndarray:
        """Create a shadow gradient for depth effect."""
        shadow = np.zeros((height, width, 4), dtype=np.uint8)
        
        dx, dy = self._get_direction_vector()
        
        if dx != 0:
            # Horizontal shadow
            gradient = np.linspace(0, 1, self.shadow_width)
            if dx > 0:
                gradient = gradient[::-1]
            
            shadow_col = np.zeros((height, self.shadow_width, 4), dtype=np.uint8)
            for i, alpha in enumerate(gradient):
                shadow_col[:, i, 3] = int(alpha * 255 * self.shadow_opacity)
            
            if dx < 0:
                shadow[:, :self.shadow_width] = shadow_col
            else:
                shadow[:, -self.shadow_width:] = shadow_col
                
        elif dy != 0:
            # Vertical shadow
            gradient = np.linspace(0, 1, self.shadow_width)
            if dy > 0:
                gradient = gradient[::-1]
            
            shadow_row = np.zeros((self.shadow_width, width, 4), dtype=np.uint8)
            for i, alpha in enumerate(gradient):
                shadow_row[i, :, 3] = int(alpha * 255 * self.shadow_opacity)
            
            if dy < 0:
                shadow[:self.shadow_width, :] = shadow_row
            else:
                shadow[-self.shadow_width:, :] = shadow_row
        
        return shadow
    
    def _apply_motion_blur(
        self,
        frame: np.ndarray,
        velocity: Tuple[float, float],
    ) -> np.ndarray:
        """Apply motion blur based on movement velocity."""
        if not self.motion_blur or self.motion_blur_samples < 2:
            return frame
        
        vx, vy = velocity
        
        # Skip if no significant motion
        if abs(vx) < 1 and abs(vy) < 1:
            return frame
        
        # Create motion-blurred frame by averaging offset copies
        result = np.zeros_like(frame, dtype=np.float32)
        
        for i in range(self.motion_blur_samples):
            t = i / (self.motion_blur_samples - 1) - 0.5  # -0.5 to 0.5
            offset_x = int(vx * t)
            offset_y = int(vy * t)
            
            # Roll/shift the frame
            shifted = np.roll(frame, (offset_y, offset_x), axis=(0, 1))
            result += shifted.astype(np.float32)
        
        result /= self.motion_blur_samples
        return result.astype(np.uint8)
    
    def make_frame(
        self,
        clip_a: VideoClip,
        clip_b: VideoClip,
        t: float,
        progress: float,
    ) -> np.ndarray:
        """Generate a single slide transition frame."""
        height, width = clip_a.h, clip_a.w
        
        # Get frames from both clips
        t_a = clip_a.duration - self.duration + t
        frame_a = clip_a.get_frame(t_a)
        
        t_b = t
        frame_b = clip_b.get_frame(t_b)
        
        # Calculate positions
        (a_x, a_y), (b_x, b_y) = self._calculate_positions(progress, width, height)
        
        # Calculate velocities for motion blur (pixels per frame at 30fps)
        dt = 1/30
        dp = self._easing_func(min(1.0, (t + dt) / self.duration)) - progress
        velocity_scale = dp / dt * self.duration
        
        dx, dy = self._get_direction_vector()
        velocity = (dx * width * velocity_scale, dy * height * velocity_scale)
        
        # Apply motion blur to both frames
        if self.motion_blur:
            frame_a = self._apply_motion_blur(frame_a, velocity)
            frame_b = self._apply_motion_blur(frame_b, velocity)
        
        # Create output canvas
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Place clip A
        self._place_frame(result, frame_a, a_x, a_y)
        
        # Place clip B
        self._place_frame(result, frame_b, b_x, b_y)
        
        # Add shadow between clips if enabled
        if self.shadow and 0 < progress < 1:
            shadow = self._create_shadow_gradient(width, height, self.direction)
            
            # Position shadow at the edge between clips
            shadow_x = b_x if self._get_direction_vector()[0] != 0 else 0
            shadow_y = b_y if self._get_direction_vector()[1] != 0 else 0
            
            # Blend shadow
            shadow_alpha = shadow[:, :, 3:4] / 255.0
            result = (result * (1 - shadow_alpha) + 0 * shadow_alpha).astype(np.uint8)
        
        return result
    
    def _place_frame(
        self,
        canvas: np.ndarray,
        frame: np.ndarray,
        x: int,
        y: int,
    ) -> None:
        """Place a frame onto the canvas at the given position."""
        height, width = canvas.shape[:2]
        fh, fw = frame.shape[:2]
        
        # Calculate source and destination regions
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = min(fw, width - x)
        src_y2 = min(fh, height - y)
        
        dst_x1 = max(0, x)
        dst_y1 = max(0, y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2, :3]


class QuickSlideTransition(SlideTransition):
    """
    Quick, snappy slide transition optimized for Instagram-style content.
    
    Features:
    - Fast default duration (0.3s)
    - Strong easing for punchy feel
    - Optimized for vertical video
    """
    
    def __init__(
        self,
        direction: TransitionDirection = TransitionDirection.LEFT,
        duration: float = 0.3,
        **kwargs
    ):
        # Use aggressive easing for snappy feel
        kwargs.setdefault('easing', 'ease_out_expo')
        kwargs.setdefault('motion_blur', True)
        kwargs.setdefault('motion_blur_samples', 3)  # Fewer samples for speed
        kwargs.setdefault('shadow', True)
        kwargs.setdefault('shadow_width', 20)
        kwargs.setdefault('shadow_opacity', 0.4)
        
        super().__init__(
            direction=direction,
            duration=duration,
            **kwargs
        )


# ==================== Additional Slide Variants ====================

class PushTransition(SlideTransition):
    """
    Push transition where clip B pushes clip A out.
    
    Similar to slide but with the visual metaphor of
    one clip physically pushing the other.
    """
    
    def __init__(
        self,
        direction: TransitionDirection = TransitionDirection.LEFT,
        duration: float = 0.5,
        **kwargs
    ):
        kwargs.setdefault('shadow', True)
        kwargs.setdefault('shadow_width', 50)
        kwargs.setdefault('shadow_opacity', 0.5)
        
        super().__init__(
            direction=direction,
            duration=duration,
            **kwargs
        )


class CoverTransition(SlideTransition):
    """
    Cover transition where clip B slides in on top of clip A.
    
    Clip A remains stationary while clip B covers it.
    """
    
    def _calculate_positions(
        self,
        progress: float,
        width: int,
        height: int,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        dx, dy = self._get_direction_vector()
        
        # Clip A stays in place
        a_x, a_y = 0, 0
        
        # Clip B slides in from off-screen
        b_x = int(-dx * (1 - progress) * width)
        b_y = int(-dy * (1 - progress) * height)
        
        return (a_x, a_y), (b_x, b_y)


class RevealTransition(SlideTransition):
    """
    Reveal transition where clip A slides out to reveal clip B underneath.
    
    Clip B remains stationary while clip A uncovers it.
    """
    
    def _calculate_positions(
        self,
        progress: float,
        width: int,
        height: int,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        dx, dy = self._get_direction_vector()
        
        # Clip A slides out
        a_x = int(dx * progress * width)
        a_y = int(dy * progress * height)
        
        # Clip B stays in place (revealed underneath)
        b_x, b_y = 0, 0
        
        return (a_x, a_y), (b_x, b_y)
    
    def make_frame(
        self,
        clip_a: VideoClip,
        clip_b: VideoClip,
        t: float,
        progress: float,
    ) -> np.ndarray:
        """Generate frame with B underneath A."""
        height, width = clip_a.h, clip_a.w
        
        # Get frames
        t_a = clip_a.duration - self.duration + t
        frame_a = clip_a.get_frame(t_a)
        
        t_b = t
        frame_b = clip_b.get_frame(t_b)
        
        # Calculate positions
        (a_x, a_y), (b_x, b_y) = self._calculate_positions(progress, width, height)
        
        # Start with clip B as base
        result = frame_b[:, :, :3].copy()
        
        # Place clip A on top
        self._place_frame(result, frame_a, a_x, a_y)
        
        return result
