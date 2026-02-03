"""
Visual effects module.

Provides text effects, color effects, and visual enhancements.
"""

from videopipe.effects.text_effects import (
    TextAnimator,
    PopInEffect,
    TypewriterEffect,
    ScaleEffect,
)
from videopipe.effects.neon import NeonGlowEffect, FuturisticTextEffect

__all__ = [
    "TextAnimator",
    "PopInEffect",
    "TypewriterEffect", 
    "ScaleEffect",
    "NeonGlowEffect",
    "FuturisticTextEffect",
]
