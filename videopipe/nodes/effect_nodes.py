"""
Effect application pipeline nodes.
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Any, Optional

from videopipe.core.node import Node, NodeResult
from videopipe.core.context import PipelineContext
from videopipe.effects.neon import NeonGlowEffect, FuturisticTextEffect, NeonConfig
from videopipe.effects.text_effects import TextAnimator, PopInEffect, ScaleEffect

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
_DEBUG_LOG_PATH = "/Users/karel.cervicek/Documents/projects/video_procesing/.cursor/debug.log"
def _dbg(hyp, loc, msg, data):
    with open(_DEBUG_LOG_PATH, "a") as f: f.write(_json_debug.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": _time_debug.time()}) + "\n")
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
    
    Useful for titles, lower thirds, or call-to-action text
    with the neon glow effect.
    
    Example:
        # Text at 1 second, sized to 50% of video width
        CreateNeonTextOverlay(
            name="text1",
            text="378",
            start_time=1.0,
            duration=2.0,
            width_percent=50,  # Text will be 50% of video width
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
            
            neon_cfg = NeonConfig(
                color=neon_settings.get("color", "#39FF14"),
                glow_intensity=neon_settings.get("glow_intensity", 1.5),
                glow_radius=neon_settings.get("glow_radius", 15),
                pulse=neon_settings.get("pulse", True),
                pulse_speed=neon_settings.get("pulse_speed", 2.0),
                font=font_name,
                font_size=calculated_font_size,
            )
            
            # Create neon text clip
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
                f"(font_size={calculated_font_size})"
            )
            
            return NodeResult.success_result(
                output=result,
                text=self.text,
                start_time=self.start_time,
                duration=actual_duration,
                font_size=calculated_font_size,
            )
            
        except Exception as e:
            return NodeResult.failure_result(e)


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
            sound_path = self.effects_folder / sound.get("name", "")
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
            from moviepy import AudioFileClip, CompositeAudioClip
            from moviepy.audio.fx import AudioFadeOut
            
            clip = context.get_main_clip()
            if clip is None:
                return NodeResult.failure_result(ValueError("No main clip"))
            
            if not self.sounds:
                logger.info("No sounds specified, skipping")
                return NodeResult.success_result(output=clip)
            
            # Get the original audio (if any)
            audio_clips = []
            if clip.audio is not None:
                audio_clips.append(clip.audio)
            
            sounds_added = 0
            
            for sound_config in self.sounds:
                sound_name = sound_config.get("name", "")
                sound_time = sound_config.get("time", 0)
                sound_volume = sound_config.get("volume", self.default_volume)
                sound_fadeout = sound_config.get("fadeout", self.fadeout)
                
                sound_path = self.effects_folder / sound_name
                
                # #region agent log
                _debug_log("S2", "effect_nodes.py:loading_sound", "Loading sound", {"sound_path": str(sound_path), "exists": sound_path.exists()})
                # #endregion
                
                if not sound_path.exists():
                    logger.warning(f"Sound file not found, skipping: {sound_path}")
                    continue
                
                # Load sound effect
                # #region agent log
                _debug_log("S2", "effect_nodes.py:before_audio_load", "About to load AudioFileClip", {"path": str(sound_path)})
                # #endregion
                
                sound_clip = AudioFileClip(str(sound_path))
                
                # #region agent log
                _debug_log("S2", "effect_nodes.py:after_audio_load", "AudioFileClip loaded", {"duration": sound_clip.duration})
                # #endregion
                
                # Apply volume
                if sound_volume != 1.0:
                    sound_clip = sound_clip.with_volume_scaled(sound_volume)
                
                # Apply fadeout
                if sound_fadeout and self.fadeout_duration > 0:
                    fade_duration = min(self.fadeout_duration, sound_clip.duration)
                    sound_clip = sound_clip.with_effects([AudioFadeOut(fade_duration)])
                
                # Set start time
                sound_clip = sound_clip.with_start(sound_time)
                
                audio_clips.append(sound_clip)
                sounds_added += 1
                
                logger.debug(
                    f"Added sound '{sound_name}' at {sound_time}s "
                    f"(volume={sound_volume}, fadeout={sound_fadeout})"
                )
            
            if sounds_added > 0:
                # Composite all audio
                composite_audio = CompositeAudioClip(audio_clips)
                
                # Set audio on clip
                result = clip.with_audio(composite_audio)
                context.set_main_clip(result)
                
                logger.info(f"Added {sounds_added} sound effect(s)")
            else:
                result = clip
            
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
