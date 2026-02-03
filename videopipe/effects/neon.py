"""
Neon glow and futuristic text effects.

Provides eye-catching text effects with:
- Neon glow with customizable colors
- Pulsing/breathing animations
- Futuristic styling with multiple glow layers
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import VideoClip, TextClip, CompositeVideoClip

if TYPE_CHECKING:
    from videopipe.core.context import PipelineContext

logger = logging.getLogger(__name__)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


@dataclass
class NeonConfig:
    """Configuration for neon glow effects."""
    # Core color (the bright inner color)
    color: str = "#39FF14"  # Neon green
    
    # Glow settings
    glow_color: Optional[str] = None  # If None, uses main color
    glow_intensity: float = 1.5
    glow_radius: int = 15
    glow_layers: int = 3
    
    # Animation
    pulse: bool = True
    pulse_speed: float = 2.0  # Hz
    pulse_min_intensity: float = 0.7
    pulse_max_intensity: float = 1.0
    
    # Text styling
    font: str = "Arial-Bold"
    font_size: int = 64
    
    # Additional futuristic effects
    scanline: bool = False
    scanline_opacity: float = 0.1
    chromatic_aberration: bool = False
    aberration_offset: int = 2


class NeonGlowEffect:
    """
    Creates neon glow effect on text.
    
    The effect is achieved by:
    1. Creating multiple blurred copies of the text at increasing sizes
    2. Layering them with the original sharp text on top
    3. Optionally animating the glow intensity
    
    Example:
        neon = NeonGlowEffect(NeonConfig(color="#00FFFF", pulse=True))
        clip = neon.create_neon_text("HELLO", duration=3.0)
    """
    
    def __init__(self, config: Optional[NeonConfig] = None):
        self.config = config or NeonConfig()
    
    def _create_glow_layer(
        self,
        text: str,
        size: Tuple[int, int],
        blur_radius: int,
        color: Tuple[int, int, int],
        font: ImageFont.FreeTypeFont,
    ) -> np.ndarray:
        """Create a single glow layer."""
        # Create image with padding for blur
        padding = blur_radius * 2
        img = Image.new('RGBA', (size[0] + padding * 2, size[1] + padding * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw text
        bbox = draw.textbbox((padding, padding), text, font=font)
        draw.text((padding, padding), text, font=font, fill=(*color, 255))
        
        # Apply gaussian blur
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return np.array(img)
    
    def _load_font(self, font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Load a font, falling back to default if not found."""
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            # Try common font paths
            common_paths = [
                f"/System/Library/Fonts/{font_name}.ttf",
                f"/System/Library/Fonts/Supplemental/{font_name}.ttf",
                f"/usr/share/fonts/truetype/{font_name.lower()}.ttf",
                f"C:/Windows/Fonts/{font_name}.ttf",
            ]
            for path in common_paths:
                try:
                    return ImageFont.truetype(path, font_size)
                except OSError:
                    continue
            
            # Fall back to default
            logger.warning(f"Font '{font_name}' not found, using default")
            return ImageFont.load_default()
    
    def create_neon_frame(
        self,
        text: str,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Create a single frame of neon text.
        
        Args:
            text: Text to render
            intensity: Glow intensity multiplier (0.0 to 1.0+)
            
        Returns:
            RGBA numpy array
        """
        config = self.config
        
        # Parse colors
        main_color = hex_to_rgb(config.color)
        glow_color = hex_to_rgb(config.glow_color) if config.glow_color else main_color
        
        # Load font
        font = self._load_font(config.font, config.font_size)
        
        # Calculate text size
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Add padding for glow
        padding = config.glow_radius * config.glow_layers * 2
        canvas_size = (text_width + padding * 2, text_height + padding * 2)
        
        # Create layers
        layers = []
        
        # Glow layers (from largest/most blurred to smallest)
        for i in range(config.glow_layers, 0, -1):
            blur = config.glow_radius * i
            layer_intensity = intensity * config.glow_intensity * (1.0 / i)
            
            # Modulate color by intensity
            layer_color = tuple(
                int(min(255, c * layer_intensity))
                for c in glow_color
            )
            
            glow_layer = self._create_glow_layer(
                text, canvas_size, blur, layer_color, font
            )
            layers.append(glow_layer)
        
        # Sharp text layer on top
        sharp_layer = self._create_glow_layer(
            text, canvas_size, 0, main_color, font
        )
        layers.append(sharp_layer)
        
        # Composite all layers
        result = np.zeros((*canvas_size[::-1], 4), dtype=np.float32)
        
        for layer in layers:
            # Resize layer to match canvas if needed
            if layer.shape[:2] != result.shape[:2]:
                pil_layer = Image.fromarray(layer)
                pil_layer = pil_layer.resize(canvas_size, Image.Resampling.LANCZOS)
                layer = np.array(pil_layer)
            
            # Alpha composite
            alpha = layer[:, :, 3:4] / 255.0
            result[:, :, :3] = result[:, :, :3] * (1 - alpha) + layer[:, :, :3] * alpha
            result[:, :, 3:4] = np.maximum(result[:, :, 3:4], layer[:, :, 3:4])
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def create_neon_text(
        self,
        text: str,
        duration: float,
        fps: int = 30,
    ) -> VideoClip:
        """
        Create animated neon text clip.
        
        Args:
            text: Text to display
            duration: Duration in seconds
            fps: Frames per second
            
        Returns:
            VideoClip with animated neon effect
        """
        config = self.config
        
        # Pre-render a frame to get size
        sample_frame = self.create_neon_frame(text, 1.0)
        height, width = sample_frame.shape[:2]
        
        def make_frame(t):
            # Calculate intensity with optional pulsing
            if config.pulse:
                phase = t * config.pulse_speed * 2 * math.pi
                intensity_range = config.pulse_max_intensity - config.pulse_min_intensity
                intensity = config.pulse_min_intensity + (intensity_range / 2) * (1 + math.sin(phase))
            else:
                intensity = 1.0
            
            frame = self.create_neon_frame(text, intensity)
            
            # Apply additional effects
            if config.scanline:
                frame = self._apply_scanlines(frame, t)
            
            if config.chromatic_aberration:
                frame = self._apply_chromatic_aberration(frame)
            
            return frame
        
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.with_fps(fps)
        
        return clip
    
    def _apply_scanlines(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Apply CRT-style scanlines."""
        height = frame.shape[0]
        scanline_height = 2
        
        # Moving scanlines
        offset = int((t * 50) % height)
        
        result = frame.copy()
        for y in range(0, height, scanline_height * 2):
            y_pos = (y + offset) % height
            if y_pos + scanline_height <= height:
                result[y_pos:y_pos + scanline_height, :, :3] = (
                    result[y_pos:y_pos + scanline_height, :, :3] * 
                    (1 - self.config.scanline_opacity)
                )
        
        return result
    
    def _apply_chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """Apply RGB channel offset for chromatic aberration effect."""
        offset = self.config.aberration_offset
        height, width = frame.shape[:2]
        
        result = np.zeros_like(frame)
        
        # Offset red channel left
        result[:, offset:, 0] = frame[:, :-offset, 0]
        
        # Keep green centered
        result[:, :, 1] = frame[:, :, 1]
        
        # Offset blue channel right
        result[:, :-offset, 2] = frame[:, offset:, 2]
        
        # Keep alpha
        result[:, :, 3] = frame[:, :, 3]
        
        return result


class FuturisticTextEffect(NeonGlowEffect):
    """
    Enhanced neon effect with additional futuristic styling.
    
    Adds:
    - Multiple color layers
    - Glitch effects
    - Digital artifacts
    """
    
    def __init__(
        self,
        config: Optional[NeonConfig] = None,
        secondary_color: str = "#00FFFF",  # Cyan
        enable_glitch: bool = False,
        glitch_intensity: float = 0.1,
    ):
        super().__init__(config)
        self.secondary_color = secondary_color
        self.enable_glitch = enable_glitch
        self.glitch_intensity = glitch_intensity
    
    def create_futuristic_text(
        self,
        text: str,
        duration: float,
        fps: int = 30,
    ) -> VideoClip:
        """
        Create futuristic animated text with enhanced effects.
        
        Args:
            text: Text to display
            duration: Duration in seconds
            fps: Frames per second
            
        Returns:
            VideoClip with futuristic effects
        """
        # Create base neon clip
        neon_clip = self.create_neon_text(text, duration, fps)
        
        # Add secondary color glow
        secondary_config = NeonConfig(
            color=self.secondary_color,
            glow_intensity=self.config.glow_intensity * 0.5,
            glow_radius=self.config.glow_radius + 5,
            glow_layers=2,
            pulse=self.config.pulse,
            pulse_speed=self.config.pulse_speed * 1.5,  # Different phase
            font=self.config.font,
            font_size=self.config.font_size,
        )
        
        secondary_effect = NeonGlowEffect(secondary_config)
        secondary_clip = secondary_effect.create_neon_text(text, duration, fps)
        
        # Composite with blend
        def make_frame(t):
            primary = neon_clip.get_frame(t)
            secondary = secondary_clip.get_frame(t)
            
            # Ensure same size
            if primary.shape != secondary.shape:
                from PIL import Image
                sec_img = Image.fromarray(secondary)
                sec_img = sec_img.resize((primary.shape[1], primary.shape[0]))
                secondary = np.array(sec_img)
            
            # Additive blend for the glow
            result = np.clip(
                primary.astype(np.float32) + secondary.astype(np.float32) * 0.3,
                0, 255
            ).astype(np.uint8)
            
            # Apply glitch if enabled
            if self.enable_glitch:
                result = self._apply_glitch(result, t)
            
            return result
        
        return VideoClip(make_frame, duration=duration).with_fps(fps)
    
    def _apply_glitch(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Apply digital glitch effect."""
        # Random glitch timing
        glitch_seed = int(t * 10)
        np.random.seed(glitch_seed)
        
        if np.random.random() > self.glitch_intensity:
            return frame
        
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Horizontal slice displacement
        num_slices = np.random.randint(1, 5)
        for _ in range(num_slices):
            y = np.random.randint(0, height - 10)
            h = np.random.randint(2, 20)
            offset = np.random.randint(-20, 20)
            
            slice_data = result[y:y+h, :].copy()
            result[y:y+h, :] = np.roll(slice_data, offset, axis=1)
        
        return result


# ==================== Helper Function ====================

def create_neon_subtitle_style(
    base_color: str = "#39FF14",
    font: str = "Arial-Bold",
    font_size: int = 48,
) -> dict[str, Any]:
    """
    Create a subtitle style dictionary with neon effect settings.
    
    This can be used with the AnimatedSubtitleRenderer to apply
    neon effects to specific words.
    """
    return {
        "type": "neon",
        "color": base_color,
        "font": font,
        "font_size": font_size,
        "glow_intensity": 1.5,
        "glow_radius": 10,
        "pulse": True,
        "pulse_speed": 2.0,
    }
