"""
Video processing pipeline nodes.
"""

from __future__ import annotations

import logging
import math
import json
import time
from pathlib import Path
from typing import Any, Optional

from moviepy import CompositeVideoClip

from videopipe.core.node import Node, NodeResult
from videopipe.core.context import PipelineContext
from videopipe.video.clip import load_clip, load_clips
from videopipe.video.merge import merge_clips, MergeConfig
from videopipe.video.export import (
    VideoExporter,
    ProResExporter,
    PRORES_422_HQ,
    get_available_presets,
)
from videopipe.transitions.base import create_transition, Transition

logger = logging.getLogger(__name__)

# #region agent log
LOG_PATH = "/Users/karel.cervicek/Documents/projects/video_procesing/.cursor/debug.log"
def _debug_log(hyp, loc, msg, data):
    with open(LOG_PATH, "a") as f: f.write(json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": time.time()}) + "\n")
# #endregion


class LoadVideosNode(Node):
    """
    Load input videos into the pipeline context.
    
    Reads video files from context.input_files and stores them
    in context.clips for subsequent nodes to process.
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        target_resolution: Optional[tuple[int, int]] = None,
    ):
        super().__init__(name="load_videos", config=config)
        self.target_resolution = target_resolution
    
    def validate(self, context: PipelineContext) -> bool:
        # #region agent log
        _debug_log("E", "video_nodes.py:validate_entry", "LoadVideosNode.validate called", {"input_files_count": len(context.input_files), "input_files": [str(p) for p in context.input_files]})
        # #endregion
        
        if not context.input_files:
            logger.error("No input files specified")
            # #region agent log
            _debug_log("E", "video_nodes.py:validate_no_files", "FAILED: No input files", {})
            # #endregion
            return False
        
        for path in context.input_files:
            if not path.exists():
                logger.error(f"Input file not found: {path}")
                # #region agent log
                _debug_log("E", "video_nodes.py:validate_not_exists", "FAILED: File not found", {"path": str(path), "exists": path.exists()})
                # #endregion
                return False
        
        # #region agent log
        _debug_log("E", "video_nodes.py:validate_success", "Validation PASSED", {})
        # #endregion
        return True
    
    def process(self, context: PipelineContext) -> NodeResult:
        # #region agent log
        _debug_log("B", "video_nodes.py:process_entry", "LoadVideosNode.process called", {"input_files_count": len(context.input_files)})
        # #endregion
        
        try:
            clips = []
            
            for i, path in enumerate(context.input_files):
                logger.info(f"Loading video {i+1}/{len(context.input_files)}: {path}")
                
                # #region agent log
                _debug_log("D", "video_nodes.py:before_load_clip", "About to call load_clip", {"path": str(path), "index": i})
                # #endregion
                
                clip = load_clip(
                    path,
                    audio=True,
                    target_resolution=self.target_resolution,
                )
                
                # #region agent log
                _debug_log("D", "video_nodes.py:after_load_clip", "load_clip returned", {"clip_w": clip.w, "clip_h": clip.h, "duration": clip.duration})
                # #endregion
                
                clip_key = f"input_{i}"
                context.add_clip(clip_key, clip)
                clips.append(clip)
                
                logger.info(f"  Resolution: {clip.w}x{clip.h}, Duration: {clip.duration:.2f}s")
            
            # Set the first clip as main if single input
            if len(clips) == 1:
                context.set_main_clip(clips[0])
                # #region agent log
                _debug_log("B", "video_nodes.py:set_main_clip", "Set main clip", {"clips_count": len(clips)})
                # #endregion
            
            return NodeResult.success_result(
                output=clips,
                clips_loaded=len(clips),
            )
            
        except Exception as e:
            # #region agent log
            _debug_log("B", "video_nodes.py:process_exception", "EXCEPTION in process", {"error": str(e), "type": type(e).__name__})
            # #endregion
            return NodeResult.failure_result(e)


class CropNode(Node):
    """
    Crop video by percentage from any edge.
    
    Removes a percentage of the video from the specified edge(s).
    Useful for removing watermarks, black bars, or unwanted content.
    
    Example:
        # Remove bottom 10% of video
        CropNode(bottom=10)
        
        # Remove 5% from top and bottom
        CropNode(top=5, bottom=5)
        
        # Remove 10% from all sides
        CropNode(top=10, bottom=10, left=10, right=10)
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        top: float = 0,
        bottom: float = 0,
        left: float = 0,
        right: float = 0,
        apply_to_all: bool = True,
    ):
        """
        Initialize crop node.
        
        Args:
            config: Node configuration
            top: Percentage to crop from top (0-100)
            bottom: Percentage to crop from bottom (0-100)
            left: Percentage to crop from left (0-100)
            right: Percentage to crop from right (0-100)
            apply_to_all: If True, apply to all input clips; if False, only main clip
        """
        super().__init__(
            name="crop",
            config=config,
            dependencies=["load_videos"],
        )
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.apply_to_all = apply_to_all
    
    def validate(self, context: PipelineContext) -> bool:
        # Validate percentages are reasonable
        for name, value in [("top", self.top), ("bottom", self.bottom), 
                           ("left", self.left), ("right", self.right)]:
            if value < 0 or value >= 50:
                logger.error(f"Invalid crop {name}: {value}% (must be 0-50)")
                return False
        
        # Check that we're not cropping more than 100% total
        if self.top + self.bottom >= 100:
            logger.error(f"Top + bottom crop ({self.top + self.bottom}%) must be less than 100%")
            return False
        if self.left + self.right >= 100:
            logger.error(f"Left + right crop ({self.left + self.right}%) must be less than 100%")
            return False
        
        return True
    
    def _crop_clip(self, clip) -> Any:
        """Apply crop to a single clip."""
        original_w, original_h = clip.w, clip.h
        
        # Calculate pixel values from percentages
        crop_top = int(original_h * (self.top / 100))
        crop_bottom = int(original_h * (self.bottom / 100))
        crop_left = int(original_w * (self.left / 100))
        crop_right = int(original_w * (self.right / 100))
        
        # Calculate new boundaries
        x1 = crop_left
        y1 = crop_top
        x2 = original_w - crop_right
        y2 = original_h - crop_bottom
        
        # Apply crop
        cropped = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
        
        logger.debug(
            f"Cropped from {original_w}x{original_h} to {cropped.w}x{cropped.h} "
            f"(top={self.top}%, bottom={self.bottom}%, left={self.left}%, right={self.right}%)"
        )
        
        return cropped
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            # Check if any cropping is needed
            if self.top == 0 and self.bottom == 0 and self.left == 0 and self.right == 0:
                logger.info("No cropping specified, skipping")
                return NodeResult.success_result(output=context.get_main_clip())
            
            crops_applied = 0
            
            # Always crop the main clip (preserves preview trimming)
            main_clip = context.get_main_clip()
            if main_clip is not None:
                cropped_main = self._crop_clip(main_clip)
                context.set_main_clip(cropped_main)
                crops_applied += 1
                # #region agent log
                _debug_log("CROP", "CropNode:process:main_cropped", "Cropped main clip", {"original_duration": main_clip.duration, "cropped_duration": cropped_main.duration})
                # #endregion
            
            if self.apply_to_all:
                # Also crop stored input clips (for multi-video pipelines)
                i = 0
                while True:
                    clip = context.get_clip(f"input_{i}")
                    if clip is None:
                        break
                    
                    cropped = self._crop_clip(clip)
                    context.add_clip(f"input_{i}", cropped)
                    i += 1
                # Note: We do NOT reset main_clip here - it's already cropped above
            
            result_clip = context.get_main_clip()
            
            logger.info(
                f"Cropped {crops_applied} clip(s): "
                f"top={self.top}%, bottom={self.bottom}%, "
                f"left={self.left}%, right={self.right}% "
                f"-> {result_clip.w}x{result_clip.h}"
            )
            
            return NodeResult.success_result(
                output=result_clip,
                crops_applied=crops_applied,
                new_resolution=f"{result_clip.w}x{result_clip.h}",
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


class SplitScreenNode(Node):
    """
    Composite two videos into one: upper (e.g. 70%) and bottom (e.g. 30%).

    Alignment and framing (option A – geometric center):
    - Both videos are aligned to the same output width; each band is full width.
    - Each video is scaled so it fully fills its band (maximum possible use of the
      source to fill the area), then center-cropped so the geometric center of
      the video is kept. No letterboxing; the band is always filled.
    - Handles different lengths via duration_mode: 'shortest' (trim to shorter)
      or 'longest' (freeze shorter on last frame).
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        upper_percent: float = 70.0,
        duration_mode: str = "shortest",
        target_size: Optional[tuple[int, int]] = None,
    ):
        super().__init__(
            name="split_screen",
            config=config,
            dependencies=["load_videos"],
        )
        self.upper_percent = upper_percent
        self.duration_mode = duration_mode
        self.target_size = target_size

    def _from_config(self, key: str, default: Any) -> Any:
        if not self.config:
            return default
        ss = self.config.get("split_screen") or {}
        if key == "upper_percent":
            return ss.get("upper_percent", self.upper_percent)
        if key == "duration_mode":
            return ss.get("duration_mode", self.duration_mode)
        if key == "target_size":
            raw = ss.get("target_size")
            if raw and len(raw) == 2:
                return tuple(int(x) for x in raw)
            return self.target_size
        return default

    def _scale_fill_center_crop(
        self,
        clip,
        target_w: int,
        target_h: int,
        shift_x_percent: float = 0.0,
        shift_y_percent: float = 0.0,
    ):
        """Scale clip to fill target area completely, then center-crop to exact size.

        Works for any ratio combination (landscape/portrait source into any band).
        Uses a single uniform scale factor so aspect ratio is preserved perfectly
        and MoviePy cannot introduce letterboxing."""
        scale = max(target_w / clip.w, target_h / clip.h)
        resized = clip.resized(scale)

        x_offset = (resized.w - target_w) // 2 - int(shift_x_percent * target_w)
        y_offset = (resized.h - target_h) // 2 - int(shift_y_percent * target_h)
        x_offset = max(0, min(resized.w - target_w, x_offset))
        y_offset = max(0, min(resized.h - target_h, y_offset))

        return resized.cropped(
            x1=x_offset,
            y1=y_offset,
            x2=x_offset + target_w,
            y2=y_offset + target_h,
        )

    def validate(self, context: PipelineContext) -> bool:
        c0 = context.get_clip("input_0")
        c1 = context.get_clip("input_1")
        if c0 is None or c1 is None:
            logger.error("SplitScreenNode requires exactly two clips (input_0, input_1)")
            return False
        return True

    def process(self, context: PipelineContext) -> NodeResult:
        upper_clip = context.get_clip("input_0")
        bottom_clip = context.get_clip("input_1")
        if upper_clip is None or bottom_clip is None:
            return NodeResult.failure_result(
                ValueError("SplitScreenNode needs input_0 and input_1")
            )

        upper_pct = self._from_config("upper_percent", self.upper_percent)
        duration_mode = self._from_config("duration_mode", self.duration_mode)
        target_size = self._from_config("target_size", self.target_size)

        upper_pct = max(10, min(90, upper_pct)) / 100.0

        if target_size:
            out_w, out_h = target_size
        else:
            out_w = upper_clip.w
            out_h = int(upper_clip.h / upper_pct)

        h_upper = int(out_h * upper_pct)
        h_bottom = out_h - h_upper

        # Align durations
        d_upper = upper_clip.duration
        d_bottom = bottom_clip.duration
        if duration_mode == "shortest":
            duration = min(d_upper, d_bottom)
            if d_upper > duration:
                upper_clip = upper_clip.subclipped(0, duration)
            if d_bottom > duration:
                bottom_clip = bottom_clip.subclipped(0, duration)
        else:
            duration = max(d_upper, d_bottom)
            if d_upper < duration:
                from moviepy import ImageClip, concatenate_videoclips
                frame = upper_clip.get_frame(max(0, d_upper - 0.04))
                freeze = ImageClip(frame).with_duration(duration - d_upper).with_fps(getattr(upper_clip, "fps", None) or 30)
                upper_clip = concatenate_videoclips([upper_clip, freeze], method="compose")
            if d_bottom < duration:
                from moviepy import ImageClip, concatenate_videoclips
                frame = bottom_clip.get_frame(max(0, d_bottom - 0.04))
                freeze = ImageClip(frame).with_duration(duration - d_bottom).with_fps(getattr(bottom_clip, "fps", None) or 30)
                bottom_clip = concatenate_videoclips([bottom_clip, freeze], method="compose")

        # Read optional per-band shift from config
        ss_cfg = self.config.get("split_screen", {}) if self.config else {}
        upper_shift = ss_cfg.get("upper_shift", [0, 0])
        bottom_shift = ss_cfg.get("bottom_shift", [0, 0])

        upper_fit = self._scale_fill_center_crop(upper_clip, out_w, h_upper, upper_shift[0], upper_shift[1])
        bottom_fit = self._scale_fill_center_crop(bottom_clip, out_w, h_bottom, bottom_shift[0], bottom_shift[1])

        upper_fit = upper_fit.with_position((0, 0))
        bottom_fit = bottom_fit.with_position((0, h_upper))

        composite = CompositeVideoClip(
            [upper_fit, bottom_fit],
            size=(out_w, out_h),
        ).with_duration(duration)

        audio_source = ss_cfg.get("audio_source", "upper")
        if audio_source == "both" and upper_clip.audio and bottom_clip.audio:
            from moviepy import CompositeAudioClip
            composite = composite.with_audio(CompositeAudioClip([upper_clip.audio, bottom_clip.audio]))
        elif audio_source == "bottom" and bottom_clip.audio:
            composite = composite.with_audio(bottom_clip.audio)
        elif upper_clip.audio:
            composite = composite.with_audio(upper_clip.audio)

        context.set_main_clip(composite)
        logger.info(
            f"Split screen: {out_w}x{out_h} (upper {upper_pct*100:.0f}% / bottom {(1-upper_pct)*100:.0f}%), duration={duration:.2f}s ({duration_mode})"
        )
        return NodeResult.success_result(
            output=composite,
            resolution=f"{out_w}x{out_h}",
            duration=duration,
        )


class PreviewModeNode(Node):
    """
    Trim video to a time range for fast preview rendering.
    
    When enabled, extracts only a portion of the video for quick testing.
    Saves significant rendering time during development.
    
    Example:
        # Preview seconds 1-5 only
        PreviewModeNode(enabled=True, start_time=1.0, end_time=5.0)
        
        # Preview with lower resolution
        PreviewModeNode(enabled=True, start_time=0, end_time=10, scale=0.5)
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        enabled: bool = False,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        scale: Optional[float] = None,
    ):
        """
        Initialize preview mode node.
        
        Args:
            config: Node configuration
            enabled: Whether preview mode is active
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of video)
            scale: Scale factor (0.5 = half resolution)
        """
        super().__init__(
            name="preview_mode",
            config=config,
            dependencies=["load_videos"],
        )
        self.enabled = enabled
        self.start_time = start_time
        self.end_time = end_time
        self.scale = scale
    
    def process(self, context: PipelineContext) -> NodeResult:
        # #region agent log
        _debug_log("C", "PreviewModeNode:process:entry", "PreviewModeNode.process called", {"enabled": self.enabled, "start_time": self.start_time, "end_time": self.end_time})
        # #endregion
        
        if not self.enabled:
            logger.info("Preview mode disabled, using full video")
            # #region agent log
            _debug_log("C", "PreviewModeNode:process:disabled", "Preview mode is DISABLED in node", {})
            # #endregion
            return NodeResult.success_result(
                output=context.get_main_clip(),
                preview_enabled=False,
            )
        
        try:
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip for preview"))
            
            original_duration = clip.duration
            
            # #region agent log
            _debug_log("C", "PreviewModeNode:process:applying", "Applying preview subclip", {"original_duration": original_duration, "start_time": self.start_time, "end_time": self.end_time})
            # #endregion
            
            # Apply time range
            start = max(0, self.start_time)
            end = self.end_time if self.end_time is not None else clip.duration
            end = min(end, clip.duration)
            # #region agent log
            _debug_log("C", "PreviewModeNode:computed_range", "Computed trim range", {"start": start, "end": end, "clip_duration": clip.duration, "end_time_param": self.end_time, "trim_is_full_video": abs(end - start - clip.duration) < 0.01})
            # #endregion

            if start >= end:
                logger.warning(f"Invalid preview range: {start}-{end}s, using full video")
                return NodeResult.success_result(
                    output=clip,
                    preview_enabled=False,
                )
            
            # Subclip
            preview_clip = clip.subclipped(start, end)
            # #region agent log
            _debug_log("C", "PreviewModeNode:process:subclipped", "Subclip created", {"new_duration": preview_clip.duration, "start": start, "end": end})
            # #endregion
            
            # Scale if specified
            if self.scale and 0 < self.scale < 1:
                new_w = int(preview_clip.w * self.scale)
                new_h = int(preview_clip.h * self.scale)
                preview_clip = preview_clip.resized((new_w, new_h))
                logger.info(f"Preview scaled to {new_w}x{new_h} ({self.scale*100:.0f}%)")
            
            context.set_main_clip(preview_clip)
            # #region agent log
            _debug_log("C", "PreviewModeNode:after_set", "Main clip set to preview subclip", {"preview_clip_duration": preview_clip.duration, "expected_duration": end - start})
            # #endregion

            # Store preview info in context for ExportNode
            context.metadata["preview_mode"] = True
            context.metadata["preview_start"] = start
            context.metadata["preview_end"] = end
            
            logger.info(
                f"Preview mode: {start:.1f}s - {end:.1f}s "
                f"(extracting {end-start:.1f}s from {original_duration:.1f}s total)"
            )
            
            return NodeResult.success_result(
                output=preview_clip,
                preview_enabled=True,
                start_time=start,
                end_time=end,
                preview_duration=end - start,
                original_duration=original_duration,
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


class InVideoTransitionNode(Node):
    """
    Apply a transition within a single video at a specified cut point.
    
    Splits the video at the given time, applies a transition between
    the two parts, and rejoins them.
    
    Example:
        # Add slide transition at 5 second mark
        InVideoTransitionNode(cut_time=5.0, transition_type="slide")
        
        # Multiple cuts with transitions
        InVideoTransitionNode(cut_times=[3.0, 6.0, 9.0], transition_type="slide")
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        cut_time: Optional[float] = None,
        cut_times: Optional[list[float]] = None,
        transition_type: str = "slide",
        transition_duration: float = 0.3,
        transition_params: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize in-video transition node.
        
        Args:
            config: Node configuration
            cut_time: Single time point to cut and add transition (seconds)
            cut_times: Multiple time points for cuts (seconds)
            transition_type: Type of transition (slide, crossfade, wipe)
            transition_duration: Duration of transition in seconds
            transition_params: Additional transition parameters
        """
        super().__init__(
            name="in_video_transition",
            config=config,
            dependencies=["load_videos"],
        )
        # Support both single time and multiple times
        if cut_times:
            self.cut_times = sorted(cut_times)
        elif cut_time is not None:
            self.cut_times = [cut_time]
        else:
            self.cut_times = []
        
        self.transition_type = transition_type
        self.transition_duration = transition_duration
        self.transition_params = transition_params or {}
    
    def validate(self, context: PipelineContext) -> bool:
        if not self.cut_times:
            logger.warning("No cut times specified for in-video transition")
            return True  # Still valid, just won't do anything
        
        clip = context.get_main_clip()
        if clip is None:
            logger.error("No main clip for in-video transition")
            return False
        
        # Validate cut times are within video duration
        for cut_time in self.cut_times:
            if cut_time <= 0 or cut_time >= clip.duration:
                logger.error(
                    f"Cut time {cut_time}s is outside video duration (0-{clip.duration:.2f}s)"
                )
                return False
        
        return True
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip"))
            
            if not self.cut_times:
                logger.info("No cut times specified, skipping in-video transition")
                return NodeResult.success_result(output=clip)
            
            # Create transition
            transition = create_transition(
                self.transition_type,
                duration=self.transition_duration,
                **self.transition_params,
            )
            
            # Split video at cut points and apply transitions
            # Add 0 and duration to make segments easier to compute
            all_times = [0] + self.cut_times + [clip.duration]
            
            # Create segments
            segments = []
            for i in range(len(all_times) - 1):
                start = all_times[i]
                end = all_times[i + 1]
                segment = clip.subclipped(start, end)
                segments.append(segment)
                logger.debug(f"Created segment {i+1}: {start:.2f}s - {end:.2f}s")
            
            # Apply transitions between segments
            result = segments[0]
            for i, segment in enumerate(segments[1:], 1):
                logger.info(
                    f"Applying {self.transition_type} transition at cut {i}/{len(self.cut_times)}"
                )
                result = transition.apply(result, segment)
            
            context.set_main_clip(result)
            
            logger.info(
                f"Applied {len(self.cut_times)} in-video transition(s), "
                f"new duration: {result.duration:.2f}s"
            )
            
            return NodeResult.success_result(
                output=result,
                cuts_made=len(self.cut_times),
                new_duration=result.duration,
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


class MergeVideosNode(Node):
    """
    Merge multiple video clips into one.
    
    Uses clips from context.clips and stores the merged result
    back as the main clip.
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        transition_type: Optional[str] = None,
        transition_duration: float = 0.5,
    ):
        super().__init__(
            name="merge_videos",
            config=config,
            dependencies=["load_videos"],
        )
        self.transition_type = transition_type
        self.transition_duration = transition_duration
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            # Get all input clips
            clips = []
            i = 0
            while True:
                clip = context.get_clip(f"input_{i}")
                if clip is None:
                    break
                clips.append(clip)
                i += 1
            
            if len(clips) < 2:
                logger.info("Less than 2 clips, no merge needed")
                if clips:
                    context.set_main_clip(clips[0])
                return NodeResult.success_result(output=clips[0] if clips else None)
            
            # Create transition if specified
            transition = None
            if self.transition_type:
                transition = create_transition(
                    self.transition_type,
                    duration=self.transition_duration,
                )
            
            # Merge clips
            merge_config = MergeConfig(
                transition=transition,
                transition_duration=self.transition_duration,
            )
            
            merged = merge_clips(clips, merge_config)
            context.set_main_clip(merged)
            
            logger.info(f"Merged {len(clips)} clips, total duration: {merged.duration:.2f}s")
            
            return NodeResult.success_result(
                output=merged,
                clips_merged=len(clips),
                total_duration=merged.duration,
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


class ApplyTransitionNode(Node):
    """
    Apply transitions between clips.
    
    Designed to work with multiple clips, applying transitions
    between each consecutive pair.
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        transition_type: str = "slide",
        transition_duration: float = 0.5,
        transition_params: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            name="apply_transitions",
            config=config,
            dependencies=["load_videos"],
        )
        self.transition_type = transition_type
        self.transition_duration = transition_duration
        self.transition_params = transition_params or {}
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            # Get clips
            clips = []
            i = 0
            while True:
                clip = context.get_clip(f"input_{i}")
                if clip is None:
                    break
                clips.append(clip)
                i += 1
            
            if len(clips) < 2:
                if clips:
                    context.set_main_clip(clips[0])
                return NodeResult.success_result(output=clips[0] if clips else None)
            
            # Create transition
            transition = create_transition(
                self.transition_type,
                duration=self.transition_duration,
                **self.transition_params,
            )
            
            # Apply transitions
            result = clips[0]
            for i, clip in enumerate(clips[1:], 1):
                logger.info(f"Applying {self.transition_type} transition {i}/{len(clips)-1}")
                result = transition.apply(result, clip)
            
            context.set_main_clip(result)
            
            return NodeResult.success_result(
                output=result,
                transitions_applied=len(clips) - 1,
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


class ExportNode(Node):
    """
    Export the final video to file.
    
    Supports various export presets including ProRes 422 HQ.
    In preview mode, automatically uses fast preset and adds _preview suffix.
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        preset: str = "prores_422_hq",
        fps: Optional[float] = None,
        preview_preset: str = "h264_fast",
    ):
        super().__init__(
            name="export",
            config=config,
            dependencies=["load_videos"],  # Must run after videos are loaded
        )
        self.preset_name = preset
        self.fps = fps
        self.preview_preset = preview_preset
    
    def validate(self, context: PipelineContext) -> bool:
        if context.output_path is None:
            logger.error("No output path specified")
            return False
        
        clip = context.get_main_clip()
        if clip is None:
            logger.error("No clip to export")
            return False
        
        return True
    
    def _get_preview_output_path(self, original_path: Path) -> Path:
        """Generate preview output path with _preview suffix and .mp4 extension."""
        stem = original_path.stem
        parent = original_path.parent
        return parent / f"{stem}_preview.mp4"
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            clip = context.get_main_clip()
            output_path = context.output_path

            # Check if preview mode is active
            is_preview = context.metadata.get("preview_mode", False)
            # #region agent log
            _debug_log("D", "ExportNode:process:entry", "ExportNode received clip", {"clip_duration": clip.duration if clip else None, "is_preview": is_preview})
            # #endregion

            # Get presets
            presets = get_available_presets()
            
            if is_preview:
                # Use preview settings: fast preset, modified filename
                output_path = self._get_preview_output_path(output_path)
                preset = presets.get(self.preview_preset)
                
                if preset is None:
                    # Fallback to h264 or create a fast preset
                    preset = presets.get("h264")
                    if preset is None:
                        logger.warning(f"Preview preset '{self.preview_preset}' not found, using default")
                        preset = PRORES_422_HQ
                
                preview_start = context.metadata.get("preview_start", 0)
                preview_end = context.metadata.get("preview_end", clip.duration)
                logger.info(f"PREVIEW MODE: Exporting {preview_start:.1f}s - {preview_end:.1f}s")
            else:
                # Use full quality settings
                preset = presets.get(self.preset_name)
                
                if preset is None:
                    logger.warning(f"Unknown preset '{self.preset_name}', using ProRes 422 HQ")
                    preset = PRORES_422_HQ
            
            # Create exporter
            exporter = VideoExporter(preset=preset)
            
            # Determine FPS
            fps = self.fps or clip.fps or 30
            
            logger.info(f"Exporting to: {output_path}")
            logger.info(f"Preset: {preset.name}")
            logger.info(f"Resolution: {clip.w}x{clip.h}")
            logger.info(f"Duration: {clip.duration:.2f}s @ {fps}fps")
            
            # Export
            result_path = exporter.export(clip, output_path, fps=fps)
            
            return NodeResult.success_result(
                output=result_path,
                output_path=str(result_path),
                preset=preset.name,
                duration=clip.duration,
                is_preview=is_preview,
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)
