"""
Plugin system for extensible video processing components.

Plugins allow adding new effects, transitions, and processing nodes
without modifying the core pipeline code.
"""

from videopipe.plugins.base import (
    Plugin,
    EffectPlugin,
    TransitionPlugin,
    ProcessorPlugin,
)
from videopipe.plugins.registry import PluginRegistry, get_registry

__all__ = [
    "Plugin",
    "EffectPlugin",
    "TransitionPlugin",
    "ProcessorPlugin",
    "PluginRegistry",
    "get_registry",
]
