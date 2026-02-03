"""
Pipeline Context - Shared state and data container for pipeline execution.

The context acts as a message bus and data store that flows through all nodes,
allowing them to share data and communicate without tight coupling.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from moviepy import VideoFileClip, AudioFileClip

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata about a video file or clip."""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    codec: str = ""
    audio_codec: str = ""
    bitrate: Optional[int] = None
    
    @classmethod
    def from_clip(cls, clip: VideoFileClip) -> VideoMetadata:
        """Extract metadata from a MoviePy clip."""
        return cls(
            width=clip.w,
            height=clip.h,
            fps=clip.fps,
            duration=clip.duration,
        )


@dataclass
class SubtitleEntry:
    """A single subtitle entry with timing and styling."""
    text: str
    start_time: float
    end_time: float
    style: dict[str, Any] = field(default_factory=dict)
    word_timings: list[dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PipelineContext:
    """
    Context object that flows through the pipeline, carrying state and data.
    
    The context provides:
    - Input/output file management
    - Video clip storage
    - Subtitle data
    - Intermediate results
    - Configuration access
    - Temporary file management
    """
    
    # Input/Output paths
    input_files: list[Path] = field(default_factory=list)
    output_path: Optional[Path] = None
    
    # Working directory for intermediate files
    work_dir: Optional[Path] = None
    
    # Video clips (keyed by identifier)
    clips: dict[str, Any] = field(default_factory=dict)
    
    # Audio clips
    audio_clips: dict[str, Any] = field(default_factory=dict)
    
    # Metadata for input videos
    metadata: dict[str, VideoMetadata] = field(default_factory=dict)
    
    # Subtitles
    subtitles: list[SubtitleEntry] = field(default_factory=list)
    
    # Special words for effects (e.g., neon effect)
    special_words: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Node outputs (keyed by node name)
    node_outputs: dict[str, Any] = field(default_factory=dict)
    
    # Pipeline configuration
    config: dict[str, Any] = field(default_factory=dict)
    
    # Export settings
    export_settings: dict[str, Any] = field(default_factory=lambda: {
        "codec": "prores_ks",
        "profile": 3,  # ProRes 422 HQ
        "pix_fmt": "yuv422p10le",  # 10-bit
        "audio_codec": "pcm_s24le",
    })
    
    def __post_init__(self):
        if self.work_dir is None:
            self.work_dir = Path(tempfile.mkdtemp(prefix="videopipe_"))
            logger.info(f"Created working directory: {self.work_dir}")
    
    def add_clip(self, key: str, clip: Any, metadata: Optional[VideoMetadata] = None):
        """Add a video clip to the context."""
        self.clips[key] = clip
        if metadata:
            self.metadata[key] = metadata
        elif hasattr(clip, 'w') and hasattr(clip, 'h'):
            self.metadata[key] = VideoMetadata.from_clip(clip)
    
    def get_clip(self, key: str) -> Optional[Any]:
        """Get a video clip from the context."""
        return self.clips.get(key)
    
    def get_main_clip(self) -> Optional[Any]:
        """Get the main/primary video clip."""
        return self.clips.get("main") or (list(self.clips.values())[0] if self.clips else None)
    
    def set_main_clip(self, clip: Any):
        """Set the main/primary video clip."""
        self.add_clip("main", clip)
    
    def add_subtitle(self, entry: SubtitleEntry):
        """Add a subtitle entry."""
        self.subtitles.append(entry)
        self.subtitles.sort(key=lambda x: x.start_time)
    
    def add_special_word(self, word: str, effect_config: dict[str, Any]):
        """Mark a word for special effects (e.g., neon glow)."""
        self.special_words[word.lower()] = effect_config
    
    def get_temp_path(self, filename: str) -> Path:
        """Get a path for a temporary file in the working directory."""
        return self.work_dir / filename
    
    def store_node_output(self, node_name: str, output: Any):
        """Store the output of a node for later use by other nodes."""
        self.node_outputs[node_name] = output
    
    def get_node_output(self, node_name: str) -> Optional[Any]:
        """Get the stored output of a previous node."""
        return self.node_outputs.get(node_name)
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        # Close all clips
        for clip in self.clips.values():
            if hasattr(clip, 'close'):
                try:
                    clip.close()
                except Exception as e:
                    logger.warning(f"Error closing clip: {e}")
        
        for clip in self.audio_clips.values():
            if hasattr(clip, 'close'):
                try:
                    clip.close()
                except Exception as e:
                    logger.warning(f"Error closing audio clip: {e}")
        
        # Optionally clean up work directory
        # (keeping for now for debugging purposes)
        logger.info(f"Working directory preserved at: {self.work_dir}")


def create_context_from_config(config: dict[str, Any]) -> PipelineContext:
    """Create a pipeline context from a configuration dictionary."""
    ctx = PipelineContext()
    
    # Set input files
    if "input_files" in config:
        ctx.input_files = [Path(f) for f in config["input_files"]]
    
    # Set output path
    if "output_path" in config:
        ctx.output_path = Path(config["output_path"])
    
    # Set special words for effects
    if "special_words" in config:
        for word, effect_config in config["special_words"].items():
            ctx.add_special_word(word, effect_config)
    
    # Override export settings
    if "export_settings" in config:
        ctx.export_settings.update(config["export_settings"])
    
    # Store full config
    ctx.config = config
    
    return ctx
