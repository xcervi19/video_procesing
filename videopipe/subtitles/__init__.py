"""
Subtitle generation and rendering module.

Provides:
- Automatic speech-to-text using OpenAI Whisper
- Subtitle styling and animation
- Word-level timing for kinetic typography effects
"""

from videopipe.subtitles.whisper_stt import WhisperTranscriber
from videopipe.subtitles.renderer import SubtitleRenderer, AnimatedSubtitleRenderer

__all__ = [
    "WhisperTranscriber",
    "SubtitleRenderer",
    "AnimatedSubtitleRenderer",
]
