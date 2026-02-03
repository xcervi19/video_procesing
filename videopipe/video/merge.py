"""
Video merging and concatenation operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from moviepy import (
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
)

from videopipe.transitions.base import Transition, CrossfadeTransition

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for video merging."""
    # Transition between clips
    transition: Optional[Transition] = None
    transition_duration: float = 0.5
    
    # Resolution handling
    target_resolution: Optional[tuple[int, int]] = None
    resize_method: str = "fit"  # fit, fill, stretch
    
    # Audio handling
    audio_crossfade: bool = True
    audio_crossfade_duration: float = 0.3
    
    # Padding for different aspect ratios
    pad_color: tuple[int, int, int] = (0, 0, 0)


def merge_clips(
    clips: list[VideoClip],
    config: Optional[MergeConfig] = None,
) -> VideoClip:
    """
    Merge multiple video clips into a single clip.
    
    Args:
        clips: List of video clips to merge
        config: Merge configuration
        
    Returns:
        Single merged video clip
    """
    if not clips:
        raise ValueError("No clips provided for merging")
    
    if len(clips) == 1:
        return clips[0]
    
    config = config or MergeConfig()
    
    # Normalize resolutions if needed
    if config.target_resolution:
        clips = _normalize_resolutions(clips, config)
    
    # Apply transitions if configured
    if config.transition:
        return _merge_with_transitions(clips, config)
    else:
        return concatenate_videoclips(clips, method="compose")


def _normalize_resolutions(
    clips: list[VideoClip],
    config: MergeConfig,
) -> list[VideoClip]:
    """
    Normalize all clips to the same resolution.
    
    Args:
        clips: List of clips
        config: Merge config with target resolution
        
    Returns:
        List of clips with normalized resolution
    """
    target_w, target_h = config.target_resolution
    normalized = []
    
    for clip in clips:
        if clip.w == target_w and clip.h == target_h:
            normalized.append(clip)
            continue
        
        if config.resize_method == "stretch":
            # Simple resize, may distort
            resized = clip.resized((target_w, target_h))
            
        elif config.resize_method == "fill":
            # Scale to fill, crop excess
            scale_w = target_w / clip.w
            scale_h = target_h / clip.h
            scale = max(scale_w, scale_h)
            
            new_w = int(clip.w * scale)
            new_h = int(clip.h * scale)
            
            resized = clip.resized((new_w, new_h))
            
            # Center crop
            x_offset = (new_w - target_w) // 2
            y_offset = (new_h - target_h) // 2
            
            resized = resized.cropped(
                x1=x_offset,
                y1=y_offset,
                x2=x_offset + target_w,
                y2=y_offset + target_h
            )
            
        else:  # fit
            # Scale to fit, add padding
            scale_w = target_w / clip.w
            scale_h = target_h / clip.h
            scale = min(scale_w, scale_h)
            
            new_w = int(clip.w * scale)
            new_h = int(clip.h * scale)
            
            resized = clip.resized((new_w, new_h))
            
            # Add padding
            from moviepy import ColorClip
            bg = ColorClip(
                size=(target_w, target_h),
                color=config.pad_color
            ).with_duration(clip.duration)
            
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            resized = resized.with_position((x_offset, y_offset))
            resized = CompositeVideoClip([bg, resized])
        
        normalized.append(resized)
        logger.debug(f"Normalized clip from {clip.w}x{clip.h} to {target_w}x{target_h}")
    
    return normalized


def _merge_with_transitions(
    clips: list[VideoClip],
    config: MergeConfig,
) -> VideoClip:
    """
    Merge clips with transitions between them.
    
    Args:
        clips: List of clips
        config: Merge config with transition
        
    Returns:
        Merged clip with transitions
    """
    transition = config.transition
    
    result = clips[0]
    
    for i, clip in enumerate(clips[1:], 1):
        logger.debug(f"Applying transition between clip {i-1} and {i}")
        result = transition.apply(result, clip)
    
    return result


def merge_with_crossfade(
    clips: list[VideoClip],
    fade_duration: float = 0.5,
) -> VideoClip:
    """
    Simple crossfade merge between clips.
    
    Args:
        clips: List of clips
        fade_duration: Duration of crossfade in seconds
        
    Returns:
        Merged clip
    """
    config = MergeConfig(
        transition=CrossfadeTransition(duration=fade_duration),
        transition_duration=fade_duration,
    )
    
    return merge_clips(clips, config)


def stack_clips(
    clips: list[VideoClip],
    direction: str = "horizontal",
    spacing: int = 0,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> VideoClip:
    """
    Stack clips horizontally or vertically.
    
    Args:
        clips: List of clips
        direction: 'horizontal' or 'vertical'
        spacing: Pixels between clips
        bg_color: Background color
        
    Returns:
        Composite clip with stacked videos
    """
    if not clips:
        raise ValueError("No clips provided")
    
    if direction == "horizontal":
        total_width = sum(c.w for c in clips) + spacing * (len(clips) - 1)
        max_height = max(c.h for c in clips)
        
        positioned = []
        x = 0
        for clip in clips:
            y = (max_height - clip.h) // 2
            positioned.append(clip.with_position((x, y)))
            x += clip.w + spacing
        
        size = (total_width, max_height)
        
    else:  # vertical
        max_width = max(c.w for c in clips)
        total_height = sum(c.h for c in clips) + spacing * (len(clips) - 1)
        
        positioned = []
        y = 0
        for clip in clips:
            x = (max_width - clip.w) // 2
            positioned.append(clip.with_position((x, y)))
            y += clip.h + spacing
        
        size = (max_width, total_height)
    
    # Create background
    duration = max(c.duration for c in clips)
    from moviepy import ColorClip
    bg = ColorClip(size=size, color=bg_color).with_duration(duration)
    
    return CompositeVideoClip([bg] + positioned, size=size)
