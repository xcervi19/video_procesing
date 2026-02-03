"""
Video transition effects module.

Provides professional transitions between video clips:
- Slide transitions (multiple directions)
- Fade transitions
- Wipe transitions
- Complex composited transitions
"""

from videopipe.transitions.slide import SlideTransition, QuickSlideTransition
from videopipe.transitions.base import (
    TransitionDirection,
    CrossfadeTransition,
    WipeTransition,
    apply_transition,
)

__all__ = [
    "SlideTransition",
    "QuickSlideTransition",
    "TransitionDirection",
    "CrossfadeTransition",
    "WipeTransition",
    "apply_transition",
]
