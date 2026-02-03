"""
Video export functionality with professional codec support.

Supports:
- ProRes 422 HQ (10-bit, professional quality)
- ProRes 4444 (with alpha channel)
- H.264/H.265 for web delivery
- Custom FFmpeg parameters
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from moviepy import VideoClip, VideoFileClip

logger = logging.getLogger(__name__)


@dataclass
class ExportPreset:
    """Export preset configuration."""
    name: str
    codec: str
    container: str = "mov"
    pixel_format: str = "yuv422p10le"
    
    # Video settings
    video_bitrate: Optional[str] = None
    quality: Optional[int] = None  # CRF or qscale depending on codec
    profile: Optional[int] = None  # Codec profile
    
    # Audio settings
    audio_codec: str = "pcm_s24le"
    audio_bitrate: Optional[str] = None
    audio_sample_rate: int = 48000
    
    # Additional FFmpeg options
    extra_params: list[str] = field(default_factory=list)
    
    def to_ffmpeg_params(self) -> list[str]:
        """Convert preset to FFmpeg command parameters."""
        params = [
            "-c:v", self.codec,
            "-pix_fmt", self.pixel_format,
        ]
        
        if self.profile is not None:
            params.extend(["-profile:v", str(self.profile)])
        
        if self.quality is not None:
            if self.codec.startswith("prores"):
                params.extend(["-qscale:v", str(self.quality)])
            elif self.codec in ("libx264", "libx265"):
                params.extend(["-crf", str(self.quality)])
        
        if self.video_bitrate:
            params.extend(["-b:v", self.video_bitrate])
        
        # Audio
        params.extend([
            "-c:a", self.audio_codec,
            "-ar", str(self.audio_sample_rate),
        ])
        
        if self.audio_bitrate:
            params.extend(["-b:a", self.audio_bitrate])
        
        # Extra params
        params.extend(self.extra_params)
        
        return params


# ==================== Preset Definitions ====================

PRORES_422_HQ = ExportPreset(
    name="ProRes 422 HQ",
    codec="prores_ks",
    container="mov",
    pixel_format="yuv422p10le",
    profile=3,  # ProRes 422 HQ
    quality=9,  # Good quality
    audio_codec="pcm_s24le",
    extra_params=[
        "-vendor", "apl0",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
    ],
)

PRORES_422_LT = ExportPreset(
    name="ProRes 422 LT",
    codec="prores_ks",
    container="mov",
    pixel_format="yuv422p10le",
    profile=1,  # ProRes 422 LT
    quality=11,
    audio_codec="pcm_s24le",
    extra_params=["-vendor", "apl0"],
)

PRORES_4444 = ExportPreset(
    name="ProRes 4444",
    codec="prores_ks",
    container="mov",
    pixel_format="yuva444p10le",  # With alpha
    profile=4,  # ProRes 4444
    quality=9,
    audio_codec="pcm_s24le",
    extra_params=["-vendor", "apl0"],
)

H264_HIGH_QUALITY = ExportPreset(
    name="H.264 High Quality",
    codec="libx264",
    container="mp4",
    pixel_format="yuv420p",
    quality=18,  # CRF 18 = visually lossless
    audio_codec="aac",
    audio_bitrate="320k",
    extra_params=[
        "-preset", "slow",
        "-profile:v", "high",
        "-level", "4.2",
    ],
)

H264_WEB = ExportPreset(
    name="H.264 Web",
    codec="libx264",
    container="mp4",
    pixel_format="yuv420p",
    quality=23,
    audio_codec="aac",
    audio_bitrate="192k",
    extra_params=[
        "-preset", "medium",
        "-profile:v", "main",
        "-movflags", "+faststart",
    ],
)

H265_HIGH_QUALITY = ExportPreset(
    name="H.265 High Quality",
    codec="libx265",
    container="mp4",
    pixel_format="yuv420p10le",
    quality=20,
    audio_codec="aac",
    audio_bitrate="320k",
    extra_params=[
        "-preset", "slow",
        "-tag:v", "hvc1",  # Compatibility tag
    ],
)

# Instagram-optimized preset
INSTAGRAM_REELS = ExportPreset(
    name="Instagram Reels",
    codec="libx264",
    container="mp4",
    pixel_format="yuv420p",
    video_bitrate="8M",
    audio_codec="aac",
    audio_bitrate="256k",
    audio_sample_rate=44100,
    extra_params=[
        "-preset", "slow",
        "-profile:v", "high",
        "-level", "4.2",
        "-movflags", "+faststart",
        "-r", "30",  # 30fps
    ],
)


# ==================== Exporter Classes ====================

class VideoExporter:
    """
    Video exporter with support for various codecs and presets.
    
    Example:
        exporter = VideoExporter(preset=PRORES_422_HQ)
        exporter.export(clip, "output.mov")
    """
    
    def __init__(
        self,
        preset: Optional[ExportPreset] = None,
        **overrides
    ):
        """
        Initialize exporter.
        
        Args:
            preset: Export preset (defaults to ProRes 422 HQ)
            **overrides: Override specific preset parameters
        """
        self.preset = preset or PRORES_422_HQ
        self.overrides = overrides
    
    def export(
        self,
        clip: VideoClip,
        output_path: Path | str,
        fps: Optional[float] = None,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """
        Export a video clip to file.
        
        Args:
            clip: VideoClip to export
            output_path: Output file path
            fps: Override FPS (uses clip FPS by default)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        
        # Ensure correct extension
        expected_ext = f".{self.preset.container}"
        if output_path.suffix.lower() != expected_ext:
            output_path = output_path.with_suffix(expected_ext)
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine FPS
        export_fps = fps or clip.fps or 30
        
        logger.info(f"Exporting to: {output_path}")
        logger.info(f"Preset: {self.preset.name}")
        logger.info(f"Resolution: {clip.w}x{clip.h} @ {export_fps}fps")
        
        # Build FFmpeg parameters
        ffmpeg_params = self.preset.to_ffmpeg_params()
        
        # Apply overrides
        for key, value in self.overrides.items():
            ffmpeg_params.extend([f"-{key}", str(value)])
        
        # Use MoviePy's write_videofile with FFmpeg parameters
        clip.write_videofile(
            str(output_path),
            fps=export_fps,
            codec=self.preset.codec,
            audio_codec=self.preset.audio_codec,
            ffmpeg_params=ffmpeg_params,
            logger="bar" if not progress_callback else None,
        )
        
        logger.info(f"Export complete: {output_path}")
        return output_path
    
    def export_with_ffmpeg(
        self,
        clip: VideoClip,
        output_path: Path | str,
        fps: Optional[float] = None,
    ) -> Path:
        """
        Export using direct FFmpeg call for more control.
        
        This method exports to an intermediate format first, then
        uses FFmpeg for final encoding, giving more precise control
        over the output parameters.
        """
        output_path = Path(output_path)
        expected_ext = f".{self.preset.container}"
        if output_path.suffix.lower() != expected_ext:
            output_path = output_path.with_suffix(expected_ext)
        
        export_fps = fps or clip.fps or 30
        
        # Export to intermediate format
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        logger.info("Exporting intermediate file...")
        clip.write_videofile(
            str(tmp_path),
            fps=export_fps,
            codec="rawvideo",
            audio_codec="pcm_s16le",
        )
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp_path),
        ]
        
        # Add preset parameters
        cmd.extend(self.preset.to_ffmpeg_params())
        
        # Add output path
        cmd.append(str(output_path))
        
        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg export failed: {result.stderr}")
        
        logger.info(f"Export complete: {output_path}")
        return output_path


class ProResExporter(VideoExporter):
    """
    Specialized exporter for ProRes output.
    
    Ensures proper settings for professional ProRes output including:
    - 10-bit color depth
    - Proper color space tagging
    - Apple-compatible vendor tag
    """
    
    PROFILES = {
        "proxy": 0,
        "lt": 1,
        "standard": 2,
        "hq": 3,
        "4444": 4,
        "4444xq": 5,
    }
    
    def __init__(
        self,
        profile: str = "hq",
        alpha: bool = False,
        quality: int = 9,
    ):
        """
        Initialize ProRes exporter.
        
        Args:
            profile: ProRes profile (proxy, lt, standard, hq, 4444, 4444xq)
            alpha: Include alpha channel (only for 4444 profiles)
            quality: Quality scale (1-31, lower is better)
        """
        profile_num = self.PROFILES.get(profile.lower(), 3)
        
        if alpha and profile_num < 4:
            logger.warning("Alpha channel requires 4444 profile, upgrading")
            profile_num = 4
        
        pixel_format = "yuva444p10le" if alpha else "yuv422p10le"
        
        preset = ExportPreset(
            name=f"ProRes {profile.upper()}",
            codec="prores_ks",
            container="mov",
            pixel_format=pixel_format,
            profile=profile_num,
            quality=quality,
            audio_codec="pcm_s24le",
            extra_params=[
                "-vendor", "apl0",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-colorspace", "bt709",
            ],
        )
        
        super().__init__(preset=preset)


# ==================== Utility Functions ====================

def export_prores_hq(
    clip: VideoClip,
    output_path: Path | str,
    fps: Optional[float] = None,
) -> Path:
    """
    Quick export to ProRes 422 HQ format.
    
    Args:
        clip: VideoClip to export
        output_path: Output file path
        fps: Override FPS
        
    Returns:
        Path to exported file
    """
    exporter = ProResExporter(profile="hq")
    return exporter.export(clip, output_path, fps=fps)


def export_for_instagram(
    clip: VideoClip,
    output_path: Path | str,
    fps: float = 30,
) -> Path:
    """
    Export optimized for Instagram Reels.
    
    Args:
        clip: VideoClip to export
        output_path: Output file path
        fps: Frame rate (default 30)
        
    Returns:
        Path to exported file
    """
    exporter = VideoExporter(preset=INSTAGRAM_REELS)
    return exporter.export(clip, output_path, fps=fps)


def get_available_presets() -> dict[str, ExportPreset]:
    """Get all available export presets."""
    return {
        "prores_422_hq": PRORES_422_HQ,
        "prores_422_lt": PRORES_422_LT,
        "prores_4444": PRORES_4444,
        "h264_high": H264_HIGH_QUALITY,
        "h264_web": H264_WEB,
        "h265_high": H265_HIGH_QUALITY,
        "instagram": INSTAGRAM_REELS,
    }
