"""
Configuration management for the video processing pipeline.

Supports YAML and JSON configuration files with validation and defaults.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# Default export settings for ProRes 422 HQ 10-bit
DEFAULT_EXPORT_SETTINGS = {
    "codec": "prores_ks",
    "profile": 3,  # ProRes 422 HQ
    "pix_fmt": "yuv422p10le",  # 10-bit 4:2:2
    "audio_codec": "pcm_s24le",
    "audio_bitrate": None,  # Use default for PCM
    "video_bitrate": None,  # ProRes uses quality-based encoding
    "quality": 9,  # qscale for prores_ks (lower = higher quality)
}

# Default subtitle settings
DEFAULT_SUBTITLE_SETTINGS = {
    "font": "Arial-Bold",
    "font_size": 48,
    "color": "white",
    "stroke_color": "black",
    "stroke_width": 2,
    "position": ("center", "bottom"),
    "margin_bottom": 50,
    "bg_color": None,
    "bg_opacity": 0.6,
}

# Default neon effect settings
DEFAULT_NEON_SETTINGS = {
    "color": "#39FF14",  # Neon green
    "glow_intensity": 1.5,
    "glow_radius": 10,
    "pulse_speed": 2.0,
    "animation_type": "pulse",
}

# Default transition settings
DEFAULT_TRANSITION_SETTINGS = {
    "duration": 0.5,
    "type": "slide",
    "direction": "left",
    "easing": "ease_in_out",
}


@dataclass
class PipelineConfig:
    """
    Configuration container for the video processing pipeline.
    
    Attributes:
        input_files: List of input video file paths
        output_path: Path for the final output video
        export_settings: Settings for video export (codec, quality, etc.)
        subtitle_settings: Default styling for subtitles
        neon_settings: Settings for neon text effects
        transition_settings: Default transition settings
        special_words: Words to highlight with special effects
        pipeline_stages: List of pipeline stages to execute
        whisper_model: Whisper model size for transcription
        debug: Enable debug mode
    """
    
    input_files: list[Path] = field(default_factory=list)
    output_path: Optional[Path] = None
    
    export_settings: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXPORT_SETTINGS.copy()
    )
    subtitle_settings: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_SUBTITLE_SETTINGS.copy()
    )
    neon_settings: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_NEON_SETTINGS.copy()
    )
    transition_settings: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_TRANSITION_SETTINGS.copy()
    )
    
    special_words: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    pipeline_stages: list[str] = field(default_factory=lambda: [
        "load_videos",
        "generate_subtitles",
        "apply_effects",
        "apply_transitions",
        "export",
    ])
    
    whisper_model: str = "medium"
    debug: bool = False
    
    @classmethod
    def from_file(cls, path: Path | str) -> PipelineConfig:
        """Load configuration from a YAML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix in ('.yaml', '.yml'):
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create configuration from a dictionary."""
        config = cls()
        
        # Input files
        if "input_files" in data:
            config.input_files = [Path(f) for f in data["input_files"]]
        
        # Output path
        if "output_path" in data:
            config.output_path = Path(data["output_path"])
        
        # Merge settings (don't replace, update defaults)
        if "export_settings" in data:
            config.export_settings.update(data["export_settings"])
        
        if "subtitle_settings" in data:
            config.subtitle_settings.update(data["subtitle_settings"])
        
        if "neon_settings" in data:
            config.neon_settings.update(data["neon_settings"])
        
        if "transition_settings" in data:
            config.transition_settings.update(data["transition_settings"])
        
        # Special words
        if "special_words" in data:
            config.special_words = data["special_words"]
        
        # Pipeline stages
        if "pipeline_stages" in data:
            config.pipeline_stages = data["pipeline_stages"]
        
        # Other settings
        if "whisper_model" in data:
            config.whisper_model = data["whisper_model"]
        
        if "debug" in data:
            config.debug = bool(data["debug"])
        
        return config
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "input_files": [str(f) for f in self.input_files],
            "output_path": str(self.output_path) if self.output_path else None,
            "export_settings": self.export_settings,
            "subtitle_settings": self.subtitle_settings,
            "neon_settings": self.neon_settings,
            "transition_settings": self.transition_settings,
            "special_words": self.special_words,
            "pipeline_stages": self.pipeline_stages,
            "whisper_model": self.whisper_model,
            "debug": self.debug,
        }
    
    def save(self, path: Path | str):
        """Save configuration to a YAML or JSON file."""
        path = Path(path)
        
        with open(path, 'w') as f:
            if path.suffix in ('.yaml', '.yml'):
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            elif path.suffix == '.json':
                json.dump(self.to_dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        logger.info(f"Configuration saved to: {path}")
    
    def validate(self) -> list[str]:
        """
        Validate the configuration and return a list of errors.
        Returns an empty list if configuration is valid.
        """
        errors = []
        
        # Check input files exist
        for input_file in self.input_files:
            if not input_file.exists():
                errors.append(f"Input file not found: {input_file}")
        
        # Check output directory exists
        if self.output_path and not self.output_path.parent.exists():
            errors.append(f"Output directory does not exist: {self.output_path.parent}")
        
        # Validate export settings
        valid_profiles = {0, 1, 2, 3}  # ProRes profiles
        if self.export_settings.get("profile") not in valid_profiles:
            errors.append(f"Invalid ProRes profile: {self.export_settings.get('profile')}")
        
        # Validate whisper model
        valid_models = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
        if self.whisper_model not in valid_models:
            errors.append(f"Invalid Whisper model: {self.whisper_model}")
        
        return errors
