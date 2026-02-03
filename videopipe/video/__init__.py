"""
Video operations module.

Provides video loading, manipulation, merging, and export functionality.
"""

from videopipe.video.clip import load_clip, load_clips, ClipInfo
from videopipe.video.merge import merge_clips, MergeConfig
from videopipe.video.export import (
    VideoExporter,
    ProResExporter,
    ExportPreset,
    PRORES_422_HQ,
    PRORES_4444,
    H264_HIGH_QUALITY,
)

__all__ = [
    "load_clip",
    "load_clips", 
    "ClipInfo",
    "merge_clips",
    "MergeConfig",
    "VideoExporter",
    "ProResExporter",
    "ExportPreset",
    "PRORES_422_HQ",
    "PRORES_4444",
    "H264_HIGH_QUALITY",
]
