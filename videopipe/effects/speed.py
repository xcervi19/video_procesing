"""
Speed change effect with pitch-preserved audio.

Changes video playback speed while keeping the original pitch of the audio
using high-quality time-stretching (phase vocoder) for natural-sounding results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from moviepy import VideoClip

logger = logging.getLogger(__name__)


def change_speed_preserve_pitch(clip: "VideoClip", factor: float) -> "VideoClip":
    """
    Change video and audio speed by `factor` while preserving audio pitch.

    - factor > 1: faster (e.g. 2.0 = double speed, half duration)
    - factor < 1: slower (e.g. 0.5 = half speed, double duration)

    Video is sped up/slowed via frame timing; audio is time-stretched
    with a phase-vocoder so pitch stays the same. Uses librosa for
    high-quality, natural-sounding audio.

    Args:
        clip: MoviePy VideoClip (with or without audio).
        factor: Speed factor. Must be positive (typical range 0.25–4.0).

    Returns:
        New VideoClip with same content at new speed and pitch-preserved audio.
    """
    if factor <= 0:
        raise ValueError("Speed factor must be positive")
    if abs(factor - 1.0) < 1e-6:
        return clip

    from moviepy import vfx
    from moviepy import AudioClip

    # Speed up/slow video (this also changes audio pitch; we replace audio below)
    sped_video = clip.with_effects([vfx.speedx, factor]).without_audio()

    if clip.audio is None:
        return sped_video

    # Time-stretch audio to match new duration while preserving pitch
    try:
        import librosa
    except ImportError:
        logger.warning(
            "librosa not installed; audio will not be pitch-preserved. "
            "Install with: pip install librosa"
        )
        # Fallback: use video's built-in speedx audio (pitch will change)
        return clip.with_effects([vfx.speedx, factor])

    audio = clip.audio
    fps_audio = audio.fps
    # Get full audio as (n_samples, n_channels), float in [-1, 1]
    raw = audio.to_soundarray()
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    n_samples, n_channels = raw.shape
    original_duration = n_samples / fps_audio
    # New duration after speed change
    new_duration = original_duration / factor

    # time_stretch(rate=factor): output is factor× shorter in time, same pitch
    # rate=2 → half the samples, same pitch → matches 2× video speed
    stretched_channels = []
    for ch in range(n_channels):
        y = raw[:, ch]
        stretched = librosa.effects.time_stretch(y, rate=factor)
        stretched_channels.append(stretched)

    stretched = np.column_stack(stretched_channels)
    if stretched.ndim == 1:
        stretched = stretched.reshape(-1, 1)

    # Clip to [-1, 1] to avoid overflow
    stretched = np.clip(stretched, -1.0, 1.0).astype(np.float32)

    n_new = stretched.shape[0]
    # Build frame getter for MoviePy AudioClip: at time t (seconds), return frame
    # Mono: single float; stereo: array of shape (2,) or (n_channels,) in [-1, 1]
    def make_frame(t: float) -> np.ndarray:
        i = int(t * fps_audio)
        if i >= n_new:
            out = np.zeros(n_channels, dtype=np.float32)
        else:
            out = stretched[i, :].copy()
        return out

    # MoviePy 2: AudioClip(get_frame, duration=..., fps=...) or .with_fps(...)
    try:
        new_audio = AudioClip(make_frame, duration=new_duration, fps=fps_audio)
    except TypeError:
        new_audio = AudioClip(make_frame, duration=new_duration).with_fps(fps_audio)
    result = sped_video.with_audio(new_audio)
    return result
