"""
FFmpeg utility functions.

Provides helper functions for working with FFmpeg, including:
- Version checking
- Command execution
- Video information extraction
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """
    Check if FFmpeg is available in the system PATH.
    
    Returns:
        True if FFmpeg is available, False otherwise
    """
    return shutil.which("ffmpeg") is not None


def get_ffmpeg_version() -> Optional[str]:
    """
    Get the FFmpeg version string.
    
    Returns:
        Version string or None if FFmpeg is not available
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # First line contains version
        first_line = result.stdout.split("\n")[0]
        return first_line
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_ffmpeg(
    args: list[str],
    input_file: Optional[Path | str] = None,
    output_file: Optional[Path | str] = None,
    overwrite: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run FFmpeg with the given arguments.
    
    Args:
        args: FFmpeg arguments (excluding ffmpeg command and -i/-y flags)
        input_file: Input file path
        output_file: Output file path
        overwrite: Whether to overwrite output file
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        CompletedProcess instance
        
    Raises:
        RuntimeError: If FFmpeg command fails
    """
    cmd = ["ffmpeg"]
    
    if overwrite:
        cmd.append("-y")
    
    if input_file:
        cmd.extend(["-i", str(input_file)])
    
    cmd.extend(args)
    
    if output_file:
        cmd.append(str(output_file))
    
    logger.debug(f"Running FFmpeg: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
    )
    
    if result.returncode != 0:
        error_msg = result.stderr if capture_output else "Unknown error"
        raise RuntimeError(f"FFmpeg failed: {error_msg}")
    
    return result


@dataclass
class VideoInfo:
    """Video file information from FFprobe."""
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
    format_name: str
    
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0


def get_video_info(path: Path | str) -> VideoInfo:
    """
    Get detailed video file information using FFprobe.
    
    Args:
        path: Path to video file
        
    Returns:
        VideoInfo dataclass with file details
        
    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If FFprobe fails
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")
    
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse FFprobe output: {e}")
    
    # Find video and audio streams
    video_stream = None
    audio_stream = None
    
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream
    
    if not video_stream:
        raise RuntimeError(f"No video stream found in: {path}")
    
    # Parse FPS
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
    else:
        fps = float(fps_str)
    
    format_info = data.get("format", {})
    
    return VideoInfo(
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
        format_name=format_info.get("format_name", "unknown"),
    )


def extract_frames(
    video_path: Path | str,
    output_dir: Path | str,
    fps: Optional[float] = None,
    quality: int = 2,
) -> list[Path]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract (None = all frames)
        quality: JPEG quality (2 = high quality)
        
    Returns:
        List of paths to extracted frames
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_pattern = output_dir / "frame_%05d.jpg"
    
    args = ["-q:v", str(quality)]
    
    if fps:
        args.extend(["-vf", f"fps={fps}"])
    
    run_ffmpeg(
        args=args,
        input_file=video_path,
        output_file=output_pattern,
    )
    
    # Return list of extracted frames
    frames = sorted(output_dir.glob("frame_*.jpg"))
    logger.info(f"Extracted {len(frames)} frames to {output_dir}")
    
    return frames


def concat_videos(
    input_files: list[Path | str],
    output_file: Path | str,
    codec: str = "copy",
) -> Path:
    """
    Concatenate multiple video files.
    
    Args:
        input_files: List of input video files
        output_file: Output file path
        codec: Video codec (use 'copy' for stream copy)
        
    Returns:
        Path to output file
    """
    import tempfile
    
    output_file = Path(output_file)
    
    # Create concat file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for path in input_files:
            f.write(f"file '{Path(path).absolute()}'\n")
        concat_file = Path(f.name)
    
    try:
        args = [
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", codec,
        ]
        
        subprocess.run(
            ["ffmpeg", "-y"] + args + [str(output_file)],
            capture_output=True,
            check=True,
        )
    finally:
        concat_file.unlink()
    
    return output_file
