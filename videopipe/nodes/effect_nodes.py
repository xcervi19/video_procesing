"""
Effect application pipeline nodes.
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from videopipe.core.node import Node, NodeResult
from videopipe.core.context import PipelineContext
from videopipe.effects.neon import NeonGlowEffect, FuturisticTextEffect, NeonConfig
from videopipe.effects.text_effects import (
    TextAnimator, 
    PopInEffect, 
    ScaleEffect,
    ProfessionalTextAnimation,
    ProfessionalAnimationConfig,
    AnimationTiming,
)
from videopipe.effects.speed import change_speed_preserve_pitch

logger = logging.getLogger(__name__)

# #region agent log
LOG_PATH = "/Users/karel.cervicek/Documents/projects/video_procesing/.cursor/debug.log"
def _debug_log(hyp, loc, msg, data):
    with open(LOG_PATH, "a") as f: f.write(json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": time.time()}) + "\n")
# #endregion


class ApplyEffectsNode(Node):
    """
    Apply visual effects to the video.
    
    This is a general-purpose effects node that can apply
    various effects based on configuration.
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        effects: Optional[list[str]] = None,
    ):
        super().__init__(
            name="apply_effects",
            config=config,
            dependencies=["load_videos"],
        )
        self.effects_list = effects or []
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip"))
            
            effects_applied = []
            
            for effect_name in self.effects_list:
                logger.info(f"Applying effect: {effect_name}")
                
                # Effect application logic would go here
                # For now, this is a placeholder for extensibility
                effects_applied.append(effect_name)
            
            context.set_main_clip(clip)
            
            return NodeResult.success_result(
                output=clip,
                effects_applied=effects_applied,
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


class ApplyNeonEffectNode(Node):
    """
    Apply neon glow effect to specific words in subtitles.
    
    Works with the subtitle system to highlight special words
    with futuristic neon styling.
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        neon_config: Optional[dict[str, Any]] = None,
        target_words: Optional[list[str]] = None,
    ):
        super().__init__(
            name="apply_neon_effect",
            config=config,
            dependencies=["render_subtitles"],
        )
        self.neon_config = neon_config or {}
        self.target_words = target_words or []
    
    def process(self, context: PipelineContext) -> NodeResult:
        try:
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip"))
            
            # Get neon configuration from context or node config
            neon_settings = {
                **context.config.get("neon_settings", {}),
                **self.neon_config,
            }
            
            # Create neon config
            neon_cfg = NeonConfig(
                color=neon_settings.get("color", "#39FF14"),
                glow_intensity=neon_settings.get("glow_intensity", 1.5),
                glow_radius=neon_settings.get("glow_radius", 10),
                pulse=neon_settings.get("pulse", True),
                pulse_speed=neon_settings.get("pulse_speed", 2.0),
            )
            
            # Register special words in context if not already done
            words = self.target_words or list(context.special_words.keys())
            
            for word in words:
                if word.lower() not in context.special_words:
                    context.add_special_word(word, {
                        "type": "neon",
                        "color": neon_cfg.color,
                        "glow_intensity": neon_cfg.glow_intensity,
                        "pulse": neon_cfg.pulse,
                    })
            
            logger.info(f"Configured neon effect for {len(words)} words")
            logger.info(f"Neon color: {neon_cfg.color}")
            
            # Note: The actual neon rendering happens in the subtitle renderer
            # This node configures the effect parameters
            
            return NodeResult.success_result(
                output=clip,
                neon_words=words,
                neon_color=neon_cfg.color,
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


# #region agent log
import json as _json_debug
import time as _time_debug
import os as _os_debug
_DEBUG_LOG_PATH = "/Users/karel.cervicek/Documents/projects/video_procesing/.cursor/debug.log"
_os_debug.makedirs(_os_debug.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
print("[DEBUG] effect_nodes.py module loading...")
def _dbg(hyp, loc, msg, data):
    try:
        with open(_DEBUG_LOG_PATH, "a") as f: f.write(_json_debug.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": _time_debug.time()}) + "\n")
        print(f"[DEBUG] Logged: {hyp} - {loc}")
    except Exception as e:
        print(f"[DEBUG LOG ERROR] {e}")
_dbg("MODULE", "effect_nodes:load", "Module loaded", {})
print("[DEBUG] effect_nodes.py module loaded successfully")
# #endregion

def _calculate_font_size_for_width(
    text: str,
    target_width: int,
    font: str = "Arial-Bold",
    base_font_size: int = 100,
) -> int:
    """
    Calculate the font size needed to make text fit a target width.
    
    Args:
        text: The text to measure
        target_width: Desired width in pixels
        font: Font name
        base_font_size: Starting font size for measurement
        
    Returns:
        Calculated font size
    """
    # #region agent log
    _dbg("F", "effect_nodes:_calculate_font_size_for_width:entry", "Font size calc started", {"text": text, "target_width": target_width, "font": font})
    # #endregion
    
    from PIL import Image, ImageDraw, ImageFont
    from pathlib import Path
    
    # Get project fonts directory
    project_fonts_dir = Path(__file__).parent.parent.parent / "fonts"
    
    # Try to load the font
    pil_font = None
    font_path_used = None
    
    # First, check if it's already a valid path
    if Path(font).exists():
        try:
            pil_font = ImageFont.truetype(font, base_font_size)
            font_path_used = font
        except OSError:
            pass
    
    # Try direct font name
    if pil_font is None:
        try:
            pil_font = ImageFont.truetype(font, base_font_size)
            font_path_used = font
            # #region agent log
            _dbg("F", "effect_nodes:_calculate_font_size_for_width:font_loaded", "Font loaded successfully", {"font": font})
            # #endregion
        except OSError:
            pass
    
    # Try project fonts directory (where auto-downloaded fonts go)
    if pil_font is None and project_fonts_dir.exists():
        # #region agent log
        _dbg("F", "effect_nodes:_calculate_font_size_for_width:searching_project_fonts", "Searching project fonts dir", {"dir": str(project_fonts_dir)})
        # #endregion
        
        # Search for matching font files
        font_normalized = font.lower().replace(" ", "").replace("-", "")
        for font_file in project_fonts_dir.glob("*.ttf"):
            file_normalized = font_file.stem.lower().replace(" ", "").replace("-", "")
            if font_normalized in file_normalized or file_normalized in font_normalized:
                try:
                    pil_font = ImageFont.truetype(str(font_file), base_font_size)
                    font_path_used = str(font_file)
                    # #region agent log
                    _dbg("F", "effect_nodes:_calculate_font_size_for_width:found_in_project", "Font found in project fonts", {"path": str(font_file)})
                    # #endregion
                    break
                except OSError:
                    continue
        
        # Also try .otf files
        if pil_font is None:
            for font_file in project_fonts_dir.glob("*.otf"):
                file_normalized = font_file.stem.lower().replace(" ", "").replace("-", "")
                if font_normalized in file_normalized or file_normalized in font_normalized:
                    try:
                        pil_font = ImageFont.truetype(str(font_file), base_font_size)
                        font_path_used = str(font_file)
                        # #region agent log
                        _dbg("F", "effect_nodes:_calculate_font_size_for_width:found_in_project_otf", "Font found in project fonts (otf)", {"path": str(font_file)})
                        # #endregion
                        break
                    except OSError:
                        continue
    
    # Try system font paths
    if pil_font is None:
        # #region agent log
        _dbg("F", "effect_nodes:_calculate_font_size_for_width:trying_system", "Trying system font paths", {"font": font})
        # #endregion
        common_paths = [
            f"/System/Library/Fonts/{font}.ttf",
            f"/System/Library/Fonts/Supplemental/{font}.ttf",
            f"/usr/share/fonts/truetype/{font.lower()}.ttf",
            # Try without spaces
            f"/System/Library/Fonts/{font.replace(' ', '')}.ttf",
            f"/System/Library/Fonts/Supplemental/{font.replace(' ', '')}.ttf",
        ]
        for path in common_paths:
            try:
                pil_font = ImageFont.truetype(path, base_font_size)
                font_path_used = path
                # #region agent log
                _dbg("F", "effect_nodes:_calculate_font_size_for_width:path_worked", "Font loaded from system path", {"path": path})
                # #endregion
                break
            except OSError:
                continue
    
    # Fallback to default
    if pil_font is None:
        # #region agent log
        _dbg("F", "effect_nodes:_calculate_font_size_for_width:default_font", "Using default font - width_percent WON'T WORK", {"searched_project_dir": str(project_fonts_dir), "exists": project_fonts_dir.exists()})
        # #endregion
        pil_font = ImageFont.load_default()
        # Default font doesn't scale well, return base size
        return base_font_size
    
    # Measure text width at base font size
    temp_img = Image.new('RGBA', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=pil_font)
    measured_width = bbox[2] - bbox[0]
    
    if measured_width == 0:
        return base_font_size
    
    # Calculate scaling ratio
    ratio = target_width / measured_width
    
    # Calculate new font size
    new_font_size = int(base_font_size * ratio)
    
    # Clamp to reasonable range
    new_font_size = max(12, min(500, new_font_size))
    
    # #region agent log
    _dbg("F", "effect_nodes:_calculate_font_size_for_width:result", "Font size calculated", {"measured_width": measured_width, "ratio": ratio, "new_font_size": new_font_size})
    # #endregion
    
    return new_font_size


class CreateNeonTextOverlay(Node):
    """
    Create standalone neon text overlays at specific times.
    
    Supports professional animations including:
    - typewriter: Characters appear one by one with smooth fade
    - pop_in: Text scales up with bounce effect
    - fade: Simple fade in/out
    - none: No animation, instant appear/disappear
    
    Example:
        # Typewriter animation with 50% width
        CreateNeonTextOverlay(
            name="text1",
            text="378",
            start_time=1.0,
            duration=2.0,
            width_percent=50,
            animation="typewriter",
        )
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        text: str = "",
        start_time: float = 0.0,
        duration: float = 3.0,
        position: tuple[str, int] = ("center", "center"),
        neon_config: Optional[dict[str, Any]] = None,
        futuristic: bool = True,
        width_percent: Optional[float] = None,
        animation: Optional[str] = None,
        animation_config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize neon text overlay node.
        
        Args:
            config: Node configuration
            name: Unique node name (required if using multiple overlays)
            text: Text to display
            start_time: When the text should appear (seconds)
            duration: How long the text should be visible (seconds)
            position: Position tuple (x, y) - can be "center", pixel values, etc.
            neon_config: Neon effect configuration
            futuristic: Use futuristic effect (True) or basic neon (False)
            width_percent: Text width as percentage of video width (e.g., 50 = 50%)
                          If None, uses font_size from neon_config
            animation: Animation type: "typewriter", "pop_in", "fade", or None
            animation_config: Optional animation settings:
                - chars_per_second: For typewriter (default: 15)
                - entrance_duration: Animation entrance time (default: 0.4)
                - exit_duration: Fade out time (default: 0.3)
                - fade_out: Whether to fade out (default: True)
        """
        node_name = name or f"neon_overlay_{text[:10]}" if text else "create_neon_overlay"
        super().__init__(
            name=node_name,
            config=config,
            dependencies=["load_videos"],
        )
        self.text = text
        self.start_time = start_time
        self.duration = duration
        self.position = position
        self.neon_config = neon_config or {}
        self.futuristic = futuristic
        self.width_percent = width_percent
        self.animation = animation
        self.animation_config = animation_config or {}
    
    def process(self, context: PipelineContext) -> NodeResult:
        # #region agent log
        _dbg("E", "CreateNeonTextOverlay:process:entry", "Processing text overlay", {"text": self.text, "width_percent": self.width_percent, "start_time": self.start_time})
        # #endregion
        try:
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip"))
            
            if not self.text:
                logger.info("No text specified for neon overlay")
                return NodeResult.success_result(output=clip)
            
            # Validate timing
            if self.start_time >= clip.duration:
                logger.warning(
                    f"Start time {self.start_time}s is beyond clip duration {clip.duration}s, skipping"
                )
                return NodeResult.success_result(output=clip)
            
            # Calculate actual duration (don't extend beyond clip)
            actual_duration = min(self.duration, clip.duration - self.start_time)
            
            # Build neon config
            neon_settings = {
                **context.config.get("neon_settings", {}),
                **self.neon_config,
            }
            
            # #region agent log
            _dbg("G", "CreateNeonTextOverlay:process:neon_settings", "Neon settings built", {"neon_settings": str(neon_settings), "self_neon_config": str(self.neon_config)})
            # #endregion
            
            # Calculate font size based on width_percent if specified
            font_name = neon_settings.get("font", "Arial-Bold")
            
            # #region agent log
            _dbg("E", "CreateNeonTextOverlay:process:width_percent_check", "Checking width_percent", {"width_percent": self.width_percent, "is_none": self.width_percent is None, "clip_w": clip.w})
            # #endregion
            
            if self.width_percent is not None:
                # Calculate target width in pixels
                target_width = int(clip.w * (self.width_percent / 100))
                
                # Calculate font size to achieve this width
                calculated_font_size = _calculate_font_size_for_width(
                    text=self.text,
                    target_width=target_width,
                    font=font_name,
                )
                
                # #region agent log
                _dbg("G", "CreateNeonTextOverlay:process:font_size_calculated", "Font size from width_percent", {"calculated_font_size": calculated_font_size, "target_width": target_width})
                # #endregion
                
                logger.info(
                    f"Text '{self.text}': width_percent={self.width_percent}% -> "
                    f"target_width={target_width}px -> font_size={calculated_font_size}"
                )
            else:
                calculated_font_size = neon_settings.get("font_size", 64)
                # #region agent log
                _dbg("G", "CreateNeonTextOverlay:process:font_size_default", "Using default font size (width_percent is None)", {"calculated_font_size": calculated_font_size})
                # #endregion
            
            # Scale glow_radius proportionally to font size for visible effect
            # Use config value as base, but scale up for large fonts
            base_glow_radius = neon_settings.get("glow_radius", 15)
            # For font sizes > 100, scale glow radius proportionally (about 8-10% of font size)
            scaled_glow_radius = max(base_glow_radius, int(calculated_font_size * 0.08))
            
            neon_cfg = NeonConfig(
                color=neon_settings.get("color", "#39FF14"),
                glow_intensity=neon_settings.get("glow_intensity", 1.5),
                glow_radius=scaled_glow_radius,
                pulse=neon_settings.get("pulse", True),
                pulse_speed=neon_settings.get("pulse_speed", 2.0),
                font=font_name,
                font_size=calculated_font_size,
            )
            
            # #region agent log
            _dbg("GLOW", "CreateNeonTextOverlay:glow_scaling", "Glow radius scaled", {"base": base_glow_radius, "scaled": scaled_glow_radius, "font_size": calculated_font_size})
            # #endregion
            
            # Check if animation is requested
            animation_type = self.animation or self.neon_config.get("animation")
            
            # #region agent log
            _dbg("A", "CreateNeonTextOverlay:branch_check", "Checking animation vs neon branch", {"animation_type": animation_type, "futuristic": self.futuristic, "has_animation": bool(animation_type and animation_type != "none")})
            # #endregion
            
            if animation_type and animation_type != "none":
                # Use professional animation system
                # #region agent log
                _dbg("B", "CreateNeonTextOverlay:animation_branch", "TOOK ANIMATION BRANCH with neon_config", {"type": animation_type, "neon_cfg_color": neon_cfg.color, "neon_cfg_glow_intensity": neon_cfg.glow_intensity})
                # #endregion
                
                # Build animation config
                anim_settings = {
                    **self.animation_config,
                    **self.neon_config.get("animation_config", {}),
                }
                
                anim_config = ProfessionalAnimationConfig(
                    animation_type=animation_type,
                    chars_per_second=anim_settings.get("chars_per_second", 15.0),
                    entrance_duration=anim_settings.get("entrance_duration", 0.4),
                    exit_duration=anim_settings.get("exit_duration", 0.3),
                    entrance_easing=anim_settings.get("entrance_easing", "ease_out_expo"),
                    exit_easing=anim_settings.get("exit_easing", "ease_in_out_quad"),
                    fade_out=anim_settings.get("fade_out", True),
                    scale_start=anim_settings.get("scale_start", 0.8),
                    scale_end=anim_settings.get("scale_end", 1.0),
                )
                
                # Create animated text WITH neon glow effect
                animator = ProfessionalTextAnimation(anim_config)
                text_clip = animator.create_animated_text(
                    text=self.text,
                    total_duration=actual_duration,
                    font=font_name,
                    font_size=calculated_font_size,
                    color=neon_settings.get("color", "#39FF14"),
                    stroke_color=neon_settings.get("stroke_color"),
                    stroke_width=neon_settings.get("stroke_width", 0),
                    neon_config=neon_cfg,  # Pass neon config for glow effect!
                )
                
                logger.info(
                    f"Created animated text: '{self.text}' with {animation_type} animation + neon glow"
                )
            else:
                # Use original neon effect (no entrance/exit animation)
                # #region agent log
                _dbg("C", "CreateNeonTextOverlay:neon_branch", "TOOK NEON BRANCH - animation SKIPPED", {"futuristic": self.futuristic})
                # #endregion
                if self.futuristic:
                    effect = FuturisticTextEffect(
                        config=neon_cfg,
                        secondary_color=neon_settings.get("secondary_color", "#00FFFF"),
                    )
                    text_clip = effect.create_futuristic_text(
                        self.text,
                        duration=actual_duration,
                    )
                else:
                    effect = NeonGlowEffect(config=neon_cfg)
                    text_clip = effect.create_neon_text(
                        self.text,
                        duration=actual_duration,
                    )
            
            # Set start time and position
            text_clip = text_clip.with_start(self.start_time)
            text_clip = text_clip.with_position(self.position)
            
            # Composite onto main clip
            from moviepy import CompositeVideoClip
            result = CompositeVideoClip([clip, text_clip], size=(clip.w, clip.h))
            result = result.with_duration(clip.duration)
            
            context.set_main_clip(result)
            
            logger.info(
                f"Created neon overlay: '{self.text}' at {self.start_time}s for {actual_duration}s "
                f"(font_size={calculated_font_size}, animation={animation_type or 'none'})"
            )
            
            return NodeResult.success_result(
                output=result,
                text=self.text,
                start_time=self.start_time,
                duration=actual_duration,
                font_size=calculated_font_size,
                animation=animation_type,
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


def _resolve_sound_name(sound_config: dict[str, Any]) -> str:
    """Resolve a sound filename from config entries."""
    return sound_config.get("name") or sound_config.get("sound") or ""


def _apply_sound_effects_to_clip(
    clip: Any,
    effects_folder: Path,
    sounds: list[dict[str, Any]],
    fadeout: bool,
    fadeout_duration: float,
    default_volume: float,
) -> tuple[Any, int]:
    """Apply sound effects to a clip, trimming to clip duration."""
    from moviepy import AudioFileClip, CompositeAudioClip
    from moviepy.audio.fx import AudioFadeOut

    audio_clips = []
    if clip.audio is not None:
        audio_clips.append(clip.audio)

    sounds_added = 0
    for sound_config in sounds:
        sound_name = _resolve_sound_name(sound_config)
        if not sound_name:
            logger.warning("Sound entry missing name; skipping")
            continue

        sound_time = float(sound_config.get("time", 0))
        if sound_time < 0:
            logger.warning(f"Sound time {sound_time}s is negative; skipping '{sound_name}'")
            continue
        if sound_time >= clip.duration:
            logger.warning(
                f"Sound time {sound_time}s exceeds clip duration {clip.duration:.2f}s; "
                f"skipping '{sound_name}'"
            )
            continue

        sound_volume = float(sound_config.get("volume", default_volume))
        sound_fadeout = sound_config.get("fadeout", fadeout)
        sound_fadeout_duration = float(
            sound_config.get("fadeout_duration", fadeout_duration)
        )

        sound_path = effects_folder / sound_name
        if not sound_path.exists():
            logger.warning(f"Sound file not found, skipping: {sound_path}")
            continue

        sound_clip = AudioFileClip(str(sound_path))
        remaining_duration = max(0.0, clip.duration - sound_time)
        if remaining_duration <= 0:
            continue

        if sound_clip.duration > remaining_duration:
            sound_clip = sound_clip.subclipped(0, remaining_duration)

        if sound_volume != 1.0:
            sound_clip = sound_clip.with_volume_scaled(sound_volume)

        if sound_fadeout and sound_fadeout_duration > 0:
            fade_duration = min(sound_fadeout_duration, sound_clip.duration)
            sound_clip = sound_clip.with_effects([AudioFadeOut(fade_duration)])

        sound_clip = sound_clip.with_start(sound_time)
        audio_clips.append(sound_clip)
        sounds_added += 1

        logger.debug(
            f"Added sound '{sound_name}' at {sound_time}s "
            f"(volume={sound_volume}, fadeout={sound_fadeout})"
        )

    if sounds_added > 0:
        composite_audio = CompositeAudioClip(audio_clips)
        result = clip.with_audio(composite_audio)
    else:
        result = clip

    return result, sounds_added


def _detect_silence_segments(
    audio_clip: Any,
    start_time: float,
    end_time: float,
    analysis_fps: int,
    silence_threshold: float,
    min_silence_duration: float,
) -> list[tuple[float, float]]:
    """Detect silent segments in an audio clip."""
    if end_time <= start_time or analysis_fps <= 0:
        return []

    step = 1.0 / analysis_fps
    times = np.arange(start_time, end_time, step)
    if times.size == 0:
        return []

    frames = np.asarray(audio_clip.get_frame(times))
    if frames.ndim == 1:
        frames = frames.reshape(-1, 1)
    elif frames.shape[0] != len(times) and frames.shape[1] == len(times):
        frames = frames.T

    rms = np.sqrt(np.mean(frames**2, axis=1))
    silent_mask = rms <= silence_threshold

    min_samples = max(1, int(min_silence_duration * analysis_fps))
    segments = []
    start_idx = None

    for idx, is_silent in enumerate(silent_mask):
        if is_silent and start_idx is None:
            start_idx = idx
        elif not is_silent and start_idx is not None:
            end_idx = idx - 1
            if end_idx - start_idx + 1 >= min_samples:
                segments.append((start_idx, end_idx))
            start_idx = None

    if start_idx is not None:
        end_idx = len(silent_mask) - 1
        if end_idx - start_idx + 1 >= min_samples:
            segments.append((start_idx, end_idx))

    return [
        (start_time + start / analysis_fps, start_time + (end + 1) / analysis_fps)
        for start, end in segments
    ]


def _select_intro_times(
    silence_segments: list[tuple[float, float]],
    insert_at: str,
    insert_offset: float,
    max_insertions: int,
) -> list[float]:
    """Select intro times from silent segments."""
    insert_at_normalized = (insert_at or "start").lower()
    times: list[float] = []

    for start, end in silence_segments:
        if insert_at_normalized == "center":
            candidate = (start + end) / 2
        elif insert_at_normalized == "end":
            candidate = max(start, end - insert_offset)
        else:
            candidate = min(end, start + insert_offset)

        times.append(candidate)
        if max_insertions and len(times) >= max_insertions:
            break

    return times


class AddSoundEffectNode(Node):
    """
    Add sound effects from a folder at specific times.
    
    Loads sound effects from a specified folder and adds them
    to the video at the given times, with optional volume fadeout.
    
    Example:
        AddSoundEffectNode(
            effects_folder=Path("effects_sound"),
            sounds=[
                {"name": "whoosh.mp3", "time": 1.0},
                {"name": "impact.wav", "time": 4.0, "volume": 0.8},
            ],
            fadeout=True,
            fadeout_duration=1.0,
        )
    """
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        effects_folder: Optional[Path | str] = None,
        sounds: Optional[list[dict[str, Any]]] = None,
        fadeout: bool = True,
        fadeout_duration: float = 1.0,
        default_volume: float = 1.0,
    ):
        """
        Initialize sound effect node.
        
        Args:
            config: Node configuration
            effects_folder: Path to folder containing sound effect files
            sounds: List of sound effect entries:
                - name: Filename of the sound effect
                - time: When to play the sound (seconds)
                - volume: Optional volume multiplier (0.0-1.0)
                - fadeout: Optional override for fadeout (True/False)
            fadeout: Default fadeout setting for all sounds
            fadeout_duration: Duration of volume fadeout in seconds
            default_volume: Default volume for sounds (0.0-1.0)
        """
        super().__init__(
            name="add_sound_effects",
            config=config,
            dependencies=["load_videos"],
        )
        self.effects_folder = Path(effects_folder) if effects_folder else None
        self.sounds = sounds or []
        self.fadeout = fadeout
        self.fadeout_duration = fadeout_duration
        self.default_volume = default_volume
    
    def validate(self, context: PipelineContext) -> bool:
        # #region agent log
        _debug_log("S1", "effect_nodes.py:validate_entry", "AddSoundEffectNode.validate called", {"sounds": self.sounds, "effects_folder": str(self.effects_folder) if self.effects_folder else None})
        # #endregion
        
        if not self.sounds:
            logger.warning("No sounds specified for sound effects node")
            return True
        
        if self.effects_folder is None:
            logger.error("No effects_folder specified")
            # #region agent log
            _debug_log("S1", "effect_nodes.py:validate_no_folder", "FAILED: No effects_folder", {})
            # #endregion
            return False
        
        if not self.effects_folder.exists():
            logger.error(f"Effects folder not found: {self.effects_folder}")
            # #region agent log
            _debug_log("S1", "effect_nodes.py:validate_folder_not_exists", "FAILED: Folder not found", {"folder": str(self.effects_folder)})
            # #endregion
            return False
        
        # Validate sound files exist
        for sound in self.sounds:
            sound_name = _resolve_sound_name(sound)
            if not sound_name:
                logger.error("Sound entry missing name in configuration")
                return False

            sound_path = self.effects_folder / sound_name
            if not sound_path.exists():
                logger.error(f"Sound file not found: {sound_path}")
                # #region agent log
                _debug_log("S1", "effect_nodes.py:validate_sound_not_found", "FAILED: Sound file not found", {"sound_path": str(sound_path)})
                # #endregion
                return False
        
        # #region agent log
        _debug_log("S1", "effect_nodes.py:validate_success", "Validation PASSED", {})
        # #endregion
        return True
    
    def process(self, context: PipelineContext) -> NodeResult:
        # #region agent log
        _debug_log("S2", "effect_nodes.py:process_entry", "AddSoundEffectNode.process called", {"sounds_count": len(self.sounds)})
        # #endregion
        
        try:
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip"))
            
            if not self.sounds:
                logger.info("No sounds specified, skipping")
                return NodeResult.success_result(output=clip)
            
            # #region agent log
            _debug_log("S2", "effect_nodes.py:apply_sound_effects", "Applying sound effects", {"effects_folder": str(self.effects_folder)})
            # #endregion

            result, sounds_added = _apply_sound_effects_to_clip(
                clip=clip,
                effects_folder=self.effects_folder,
                sounds=self.sounds,
                fadeout=self.fadeout,
                fadeout_duration=self.fadeout_duration,
                default_volume=self.default_volume,
            )

            if sounds_added > 0:
                context.set_main_clip(result)
                logger.info(f"Added {sounds_added} sound effect(s)")
            
            # #region agent log
            _debug_log("S2", "effect_nodes.py:process_success", "AddSoundEffectNode SUCCESS", {"sounds_added": sounds_added})
            # #endregion
            
            return NodeResult.success_result(
                output=result,
                sounds_added=sounds_added,
            )
            
        except Exception as e:
            # #region agent log
            _debug_log("S2", "effect_nodes.py:process_exception", "EXCEPTION in AddSoundEffectNode", {"error": str(e), "type": type(e).__name__})
            # #endregion
            return NodeResult.failure_result(e)


class AutomaticIntroBackgroundSoundNode(Node):
    """
    Automatically insert a background intro sound after detecting silence.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        effects_folder: Optional[Path | str] = None,
        sound_name: Optional[str] = None,
        enabled: Optional[bool] = None,
        fadeout: Optional[bool] = None,
        fadeout_duration: Optional[float] = None,
        default_volume: Optional[float] = None,
        min_silence_duration: Optional[float] = None,
        silence_threshold: Optional[float] = None,
        analysis_fps: Optional[int] = None,
        max_insertions: Optional[int] = None,
        insert_at: Optional[str] = None,
        insert_offset: Optional[float] = None,
        search_start: Optional[float] = None,
        search_end: Optional[float] = None,
    ):
        super().__init__(
            name="automatic_intro_background_sound",
            config=config,
            dependencies=["load_videos"],
        )
        self.effects_folder = Path(effects_folder) if effects_folder else None
        self.sound_name = sound_name
        self.enabled = enabled
        self.fadeout = fadeout
        self.fadeout_duration = fadeout_duration
        self.default_volume = default_volume
        self.min_silence_duration = min_silence_duration
        self.silence_threshold = silence_threshold
        self.analysis_fps = analysis_fps
        self.max_insertions = max_insertions
        self.insert_at = insert_at
        self.insert_offset = insert_offset
        self.search_start = search_start
        self.search_end = search_end

    def process(self, context: PipelineContext) -> NodeResult:
        try:
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip"))

            auto_config = context.config.get("automatic_intro_background_sound", {})
            enabled = (
                self.enabled
                if self.enabled is not None
                else auto_config.get("enabled", True)
            )
            if not enabled:
                logger.info("Automatic intro background sound disabled, skipping")
                return NodeResult.success_result(output=clip)

            effects_folder = self.effects_folder or Path(
                auto_config.get("folder", "effects_sound")
            )
            sound_name = (
                self.sound_name
                or auto_config.get("sound")
                or auto_config.get("name")
            )

            if not sound_name:
                logger.warning("No sound name specified for automatic intro background sound")
                return NodeResult.success_result(output=clip)

            if not effects_folder.exists():
                logger.warning(f"Effects folder not found: {effects_folder}")
                return NodeResult.success_result(output=clip)

            fadeout = (
                self.fadeout
                if self.fadeout is not None
                else auto_config.get("fadeout", True)
            )
            fadeout_duration = (
                self.fadeout_duration
                if self.fadeout_duration is not None
                else float(auto_config.get("fadeout_duration", 1.0))
            )
            default_volume = (
                self.default_volume
                if self.default_volume is not None
                else float(auto_config.get("volume", 1.0))
            )
            min_silence_duration = (
                self.min_silence_duration
                if self.min_silence_duration is not None
                else float(auto_config.get("min_silence_duration", 1.0))
            )
            silence_threshold = (
                self.silence_threshold
                if self.silence_threshold is not None
                else float(auto_config.get("silence_threshold", 0.001))
            )
            analysis_fps = (
                self.analysis_fps
                if self.analysis_fps is not None
                else int(auto_config.get("analysis_fps", 100))
            )
            max_insertions = (
                self.max_insertions
                if self.max_insertions is not None
                else int(auto_config.get("max_insertions", 1))
            )
            insert_at = self.insert_at or auto_config.get("insert_at", "start")
            insert_offset = (
                self.insert_offset
                if self.insert_offset is not None
                else float(auto_config.get("insert_offset", 0.0))
            )
            search_start = (
                self.search_start
                if self.search_start is not None
                else float(auto_config.get("search_start", 0.0))
            )
            search_end = (
                self.search_end
                if self.search_end is not None
                else auto_config.get("search_end")
            )

            search_start = max(0.0, search_start)
            search_end = clip.duration if search_end is None else float(search_end)
            search_end = min(clip.duration, search_end)

            if search_end <= search_start:
                logger.warning(
                    "Automatic intro background sound search window is empty, skipping"
                )
                return NodeResult.success_result(output=clip)

            if clip.audio is None:
                silence_segments = [(search_start, search_end)]
            else:
                silence_segments = _detect_silence_segments(
                    audio_clip=clip.audio,
                    start_time=search_start,
                    end_time=search_end,
                    analysis_fps=analysis_fps,
                    silence_threshold=silence_threshold,
                    min_silence_duration=min_silence_duration,
                )

            if not silence_segments:
                logger.info(
                    "No qualifying silence segments found for automatic intro background sound"
                )
                return NodeResult.success_result(output=clip)

            intro_times = _select_intro_times(
                silence_segments=silence_segments,
                insert_at=insert_at,
                insert_offset=insert_offset,
                max_insertions=max_insertions,
            )

            sounds = [
                {
                    "name": sound_name,
                    "time": intro_time,
                    "volume": default_volume,
                    "fadeout": fadeout,
                    "fadeout_duration": fadeout_duration,
                }
                for intro_time in intro_times
            ]

            result, sounds_added = _apply_sound_effects_to_clip(
                clip=clip,
                effects_folder=effects_folder,
                sounds=sounds,
                fadeout=fadeout,
                fadeout_duration=fadeout_duration,
                default_volume=default_volume,
            )

            if sounds_added > 0:
                context.set_main_clip(result)
                logger.info(
                    f"Automatic intro background sound inserted {sounds_added} time(s)"
                )

            return NodeResult.success_result(
                output=result,
                sounds_added=sounds_added,
                intro_times=intro_times,
            )
        except Exception as e:
            return NodeResult.failure_result(e)


class ChangeSpeedNode(Node):
    """
    Change video playback speed while preserving audio pitch.

    Uses high-quality time-stretching (phase vocoder via librosa) so that
    speech and music keep their original pitch at the new speed.

    Example:
        ChangeSpeedNode(speed_factor=1.5)   # 50% faster
        ChangeSpeedNode(speed_factor=0.5)   # Half speed (slow motion)
        # From config:
        ChangeSpeedNode()  # uses context.config["speed_factor"]
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        speed_factor: Optional[float] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize speed change node.

        Args:
            config: Node configuration
            speed_factor: Speed multiplier (e.g. 2.0 = 2x faster, 0.5 = half speed).
                          If None, reads from config["speed_factor"].
            name: Optional unique node name (default: "change_speed")
        """
        super().__init__(
            name=name or "change_speed",
            config=config,
            dependencies=["load_videos"],
        )
        self.speed_factor = speed_factor

    def process(self, context: PipelineContext) -> NodeResult:
        try:
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip"))

            factor = self.speed_factor
            if factor is None:
                factor = context.config.get("speed_factor")
            if factor is None:
                logger.warning("No speed_factor set; leaving clip unchanged")
                return NodeResult.success_result(output=clip)

            if factor <= 0:
                return NodeResult.failure_result(
                    ValueError(f"speed_factor must be positive, got {factor}")
                )

            result = change_speed_preserve_pitch(clip, factor)
            context.set_main_clip(result)

            logger.info(
                f"Speed changed by factor {factor}: "
                f"duration {clip.duration:.2f}s -> {result.duration:.2f}s (pitch preserved)"
            )

            return NodeResult.success_result(
                output=result,
                speed_factor=factor,
                original_duration=clip.duration,
                new_duration=result.duration,
            )

        except Exception as e:
            return NodeResult.failure_result(e)
