"""
Base plugin classes for the video processing pipeline.

All plugins inherit from the Plugin base class and implement
specific interfaces for their functionality type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from moviepy import VideoClip

if TYPE_CHECKING:
    from videopipe.core.context import PipelineContext


@dataclass
class PluginMetadata:
    """Metadata describing a plugin."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)


class Plugin(ABC):
    """
    Base class for all plugins.
    
    Plugins are reusable components that can be registered with the
    plugin registry and used by pipeline nodes.
    """
    
    # Override in subclasses
    metadata: PluginMetadata = PluginMetadata(name="BasePlugin")
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self._config = config or {}
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def config(self) -> dict[str, Any]:
        return self._config
    
    def configure(self, **kwargs) -> Plugin:
        """Update plugin configuration."""
        self._config.update(kwargs)
        return self
    
    @abstractmethod
    def apply(self, clip: VideoClip, context: PipelineContext, **kwargs) -> VideoClip:
        """
        Apply the plugin's effect/transformation to a clip.
        
        Args:
            clip: The input video clip
            context: Pipeline context for accessing shared state
            **kwargs: Additional parameters
            
        Returns:
            The transformed video clip
        """
        pass
    
    def validate_config(self) -> list[str]:
        """
        Validate the plugin configuration.
        Returns a list of error messages (empty if valid).
        """
        return []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class EffectPlugin(Plugin):
    """
    Base class for visual effect plugins.
    
    Effect plugins modify the visual appearance of video clips,
    such as adding text overlays, color grading, or animations.
    """
    
    @property
    def effect_type(self) -> str:
        """Type of effect (e.g., 'text', 'color', 'animation')."""
        return "generic"
    
    def get_duration(self) -> Optional[float]:
        """
        Get the duration of the effect.
        Returns None if the effect spans the entire clip.
        """
        return None
    
    def supports_keyframes(self) -> bool:
        """Whether this effect supports keyframe animation."""
        return False


class TransitionPlugin(Plugin):
    """
    Base class for video transition plugins.
    
    Transition plugins handle the visual transition between two clips,
    such as fades, wipes, slides, or complex morphs.
    """
    
    metadata = PluginMetadata(name="BaseTransition")
    
    @property
    def transition_type(self) -> str:
        """Type of transition (e.g., 'slide', 'fade', 'wipe')."""
        return "generic"
    
    @property
    def default_duration(self) -> float:
        """Default duration for this transition in seconds."""
        return 0.5
    
    @abstractmethod
    def apply_transition(
        self,
        clip_a: VideoClip,
        clip_b: VideoClip,
        duration: float,
        context: PipelineContext,
        **kwargs
    ) -> VideoClip:
        """
        Apply the transition between two clips.
        
        Args:
            clip_a: The outgoing clip
            clip_b: The incoming clip
            duration: Duration of the transition in seconds
            context: Pipeline context
            **kwargs: Additional parameters
            
        Returns:
            A single clip with the transition applied
        """
        pass
    
    def apply(self, clip: VideoClip, context: PipelineContext, **kwargs) -> VideoClip:
        """
        Apply method for compatibility with Plugin interface.
        For transitions, use apply_transition instead.
        """
        raise NotImplementedError(
            "Use apply_transition() for TransitionPlugin instead of apply()"
        )


class ProcessorPlugin(Plugin):
    """
    Base class for general processing plugins.
    
    Processor plugins handle general video processing tasks that don't
    fit neatly into effects or transitions, such as:
    - Audio processing
    - Subtitle generation
    - Video analysis
    - Format conversion
    """
    
    metadata = PluginMetadata(name="BaseProcessor")
    
    @property
    def processor_type(self) -> str:
        """Type of processor (e.g., 'audio', 'subtitle', 'analysis')."""
        return "generic"
    
    def process(self, context: PipelineContext, **kwargs) -> Any:
        """
        Execute the processing task.
        
        Unlike apply(), process() can modify the context directly
        and return arbitrary data rather than just a clip.
        
        Args:
            context: Pipeline context
            **kwargs: Additional parameters
            
        Returns:
            Processing result (type depends on processor)
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def apply(self, clip: VideoClip, context: PipelineContext, **kwargs) -> VideoClip:
        """Default implementation that processes and returns the clip unchanged."""
        self.process(context, **kwargs)
        return clip


class SubtitlePlugin(ProcessorPlugin):
    """
    Base class for subtitle-related plugins.
    
    Handles subtitle generation, styling, and rendering.
    """
    
    metadata = PluginMetadata(name="BaseSubtitle")
    
    @property
    def processor_type(self) -> str:
        return "subtitle"
    
    @abstractmethod
    def generate_subtitles(
        self,
        context: PipelineContext,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Generate subtitle entries from video/audio.
        
        Returns:
            List of subtitle dictionaries with timing and text
        """
        pass
    
    @abstractmethod
    def render_subtitles(
        self,
        clip: VideoClip,
        subtitles: list[dict[str, Any]],
        context: PipelineContext,
        **kwargs
    ) -> VideoClip:
        """
        Render subtitles onto a video clip.
        
        Args:
            clip: The video clip
            subtitles: List of subtitle entries
            context: Pipeline context
            **kwargs: Style and rendering options
            
        Returns:
            Clip with subtitles rendered
        """
        pass


class TextEffectPlugin(EffectPlugin):
    """
    Base class for text-based visual effects.
    
    Specialized for text animations, styling, and effects
    like neon glow, kinetic typography, etc.
    """
    
    metadata = PluginMetadata(name="BaseTextEffect")
    
    @property
    def effect_type(self) -> str:
        return "text"
    
    @abstractmethod
    def create_text_clip(
        self,
        text: str,
        duration: float,
        context: PipelineContext,
        **kwargs
    ) -> VideoClip:
        """
        Create an animated text clip.
        
        Args:
            text: The text to display
            duration: Duration of the clip
            context: Pipeline context
            **kwargs: Style and animation options
            
        Returns:
            Animated text clip
        """
        pass
    
    def apply(self, clip: VideoClip, context: PipelineContext, **kwargs) -> VideoClip:
        """Apply text effect as overlay on existing clip."""
        text = kwargs.get("text", "")
        if not text:
            return clip
        
        text_clip = self.create_text_clip(
            text=text,
            duration=clip.duration,
            context=context,
            **kwargs
        )
        
        from moviepy import CompositeVideoClip
        return CompositeVideoClip([clip, text_clip])
