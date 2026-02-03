"""
Video clip loading and information extraction.
"""

from __future__ import annotations

import logging
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from moviepy import VideoFileClip, AudioFileClip

logger = logging.getLogger(__name__)


@dataclass
class ClipInfo:
    """Information about a video clip."""
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    pix_fmt: str
    bitrate: Optional[int]
    audio_codec: Optional[str]
    audio_sample_rate: Optional[int]
    audio_channels: Optional[int]
    
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "codec": self.codec,
            "pix_fmt": self.pix_fmt,
            "bitrate": self.bitrate,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
        }


def get_clip_info(path: Path | str) -> ClipInfo:
    """
    Get detailed information about a video file using FFprobe.
    
    Args:
        path: Path to video file
        
    Returns:
        ClipInfo with video metadata
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    
    # Use ffprobe to get detailed info
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.warning(f"FFprobe failed, falling back to MoviePy: {e}")
        return _get_clip_info_moviepy(path)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse FFprobe output: {e}")
        return _get_clip_info_moviepy(path)
    
    # Extract video stream info
    video_stream = None
    audio_stream = None
    
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream
    
    if not video_stream:
        raise ValueError(f"No video stream found in: {path}")
    
    # Parse FPS (can be "30/1" format)
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
    else:
        fps = float(fps_str)
    
    # Get format info
    format_info = data.get("format", {})
    
    return ClipInfo(
        path=path,
        width=video_stream.get("width", 0),
        height=video_stream.get("height", 0),
        fps=fps,
        duration=float(format_info.get("duration", 0)),
        codec=video_stream.get("codec_name", "unknown"),
        pix_fmt=video_stream.get("pix_fmt", "unknown"),
        bitrate=int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        audio_sample_rate=int(audio_stream.get("sample_rate", 0)) if audio_stream else None,
        audio_channels=audio_stream.get("channels") if audio_stream else None,
    )


def _get_clip_info_moviepy(path: Path) -> ClipInfo:
    """Fallback method using MoviePy to get clip info."""
    clip = VideoFileClip(str(path))
    
    info = ClipInfo(
        path=path,
        width=clip.w,
        height=clip.h,
        fps=clip.fps or 30,
        duration=clip.duration,
        codec="unknown",
        pix_fmt="unknown",
        bitrate=None,
        audio_codec="unknown" if clip.audio else None,
        audio_sample_rate=None,
        audio_channels=None,
    )
    
    clip.close()
    return info


def load_clip(
    path: Path | str,
    audio: bool = True,
    target_resolution: Optional[tuple[int, int]] = None,
) -> VideoFileClip:
    """
    Load a video clip with optional preprocessing.
    
    Args:
        path: Path to video file
        audio: Whether to load audio
        target_resolution: Optional (width, height) to resize to
        
    Returns:
        VideoFileClip ready for processing
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    
    logger.info(f"Loading clip: {path}")
    
    clip = VideoFileClip(str(path))
    
    if not audio:
        clip = clip.without_audio()
    
    if target_resolution:
        clip = clip.resized(target_resolution)
        logger.debug(f"Resized clip to: {target_resolution}")
    
    return clip


def load_clips(
    paths: list[Path | str],
    audio: bool = True,
    target_resolution: Optional[tuple[int, int]] = None,
) -> list[VideoFileClip]:
    """
    Load multiple video clips.
    
    Args:
        paths: List of paths to video files
        audio: Whether to load audio
        target_resolution: Optional resolution to resize all clips to
        
    Returns:
        List of VideoFileClip objects
    """
    clips = []
    
    for path in paths:
        try:
            clip = load_clip(path, audio=audio, target_resolution=target_resolution)
            clips.append(clip)
        except Exception as e:
            logger.error(f"Failed to load clip {path}: {e}")
            raise
    
    return clips


def extract_audio(
    video_path: Path | str,
    output_path: Optional[Path | str] = None,
    sample_rate: int = 44100,
) -> Path:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to video file
        output_path: Optional output path (defaults to same name with .wav)
        sample_rate: Audio sample rate
        
    Returns:
        Path to extracted audio file
    """
    video_path = Path(video_path)
    
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "2",  # Stereo
        str(output_path)
    ]
    
    logger.info(f"Extracting audio to: {output_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")
    
    return output_path
