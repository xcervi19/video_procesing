"""
Plugin Registry - Central registry for discovering and managing plugins.

The registry provides a way to:
- Register plugins by name
- Discover plugins automatically
- Access plugins from anywhere in the codebase
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Optional, Type

from videopipe.plugins.base import (
    Plugin,
    EffectPlugin,
    TransitionPlugin,
    ProcessorPlugin,
    SubtitlePlugin,
    TextEffectPlugin,
)
from videopipe.core.node import Node

logger = logging.getLogger(__name__)

# Global registry instance
_registry: Optional[PluginRegistry] = None


class PluginRegistry:
    """
    Central registry for all plugins and nodes.
    
    Provides registration, discovery, and access to:
    - Effect plugins
    - Transition plugins
    - Processor plugins
    - Pipeline nodes
    """
    
    def __init__(self):
        self._effects: dict[str, Type[EffectPlugin]] = {}
        self._transitions: dict[str, Type[TransitionPlugin]] = {}
        self._processors: dict[str, Type[ProcessorPlugin]] = {}
        self._nodes: dict[str, Type[Node]] = {}
        self._instances: dict[str, Plugin] = {}
    
    # ==================== Registration ====================
    
    def register_effect(self, name: str, plugin_class: Type[EffectPlugin]) -> None:
        """Register an effect plugin."""
        self._effects[name] = plugin_class
        logger.debug(f"Registered effect: {name}")
    
    def register_transition(self, name: str, plugin_class: Type[TransitionPlugin]) -> None:
        """Register a transition plugin."""
        self._transitions[name] = plugin_class
        logger.debug(f"Registered transition: {name}")
    
    def register_processor(self, name: str, plugin_class: Type[ProcessorPlugin]) -> None:
        """Register a processor plugin."""
        self._processors[name] = plugin_class
        logger.debug(f"Registered processor: {name}")
    
    def register_node(self, name: str, node_class: Type[Node]) -> None:
        """Register a pipeline node."""
        self._nodes[name] = node_class
        logger.debug(f"Registered node: {name}")
    
    def register(self, name: str, plugin_class: Type[Plugin]) -> None:
        """
        Register a plugin, automatically detecting its type.
        """
        if issubclass(plugin_class, TransitionPlugin):
            self.register_transition(name, plugin_class)
        elif issubclass(plugin_class, TextEffectPlugin):
            self.register_effect(name, plugin_class)
        elif issubclass(plugin_class, SubtitlePlugin):
            self.register_processor(name, plugin_class)
        elif issubclass(plugin_class, EffectPlugin):
            self.register_effect(name, plugin_class)
        elif issubclass(plugin_class, ProcessorPlugin):
            self.register_processor(name, plugin_class)
        else:
            raise TypeError(f"Unknown plugin type: {plugin_class}")
    
    # ==================== Retrieval ====================
    
    def get_effect(self, name: str, **config) -> Optional[EffectPlugin]:
        """Get an effect plugin instance."""
        if name not in self._effects:
            logger.warning(f"Effect not found: {name}")
            return None
        return self._effects[name](config=config)
    
    def get_transition(self, name: str, **config) -> Optional[TransitionPlugin]:
        """Get a transition plugin instance."""
        if name not in self._transitions:
            logger.warning(f"Transition not found: {name}")
            return None
        return self._transitions[name](config=config)
    
    def get_processor(self, name: str, **config) -> Optional[ProcessorPlugin]:
        """Get a processor plugin instance."""
        if name not in self._processors:
            logger.warning(f"Processor not found: {name}")
            return None
        return self._processors[name](config=config)
    
    def get_node_class(self, name: str) -> Optional[Type[Node]]:
        """Get a node class by name."""
        return self._nodes.get(name)
    
    def get_all_effects(self) -> dict[str, Type[EffectPlugin]]:
        """Get all registered effect plugins."""
        return self._effects.copy()
    
    def get_all_transitions(self) -> dict[str, Type[TransitionPlugin]]:
        """Get all registered transition plugins."""
        return self._transitions.copy()
    
    def get_all_processors(self) -> dict[str, Type[ProcessorPlugin]]:
        """Get all registered processor plugins."""
        return self._processors.copy()
    
    def get_all_nodes(self) -> dict[str, Type[Node]]:
        """Get all registered nodes."""
        return self._nodes.copy()
    
    # ==================== Discovery ====================
    
    def discover_plugins(self, package_name: str = "videopipe") -> int:
        """
        Automatically discover and register plugins from a package.
        
        Looks for:
        - Classes inheriting from Plugin in effects/, transitions/, subtitles/
        - Classes inheriting from Node in nodes/
        
        Returns:
            Number of plugins discovered
        """
        count = 0
        
        # Import subpackages that contain plugins
        subpackages = ["effects", "transitions", "subtitles", "nodes"]
        
        for subpkg in subpackages:
            try:
                module_name = f"{package_name}.{subpkg}"
                module = importlib.import_module(module_name)
                
                # Walk through all modules in the subpackage
                if hasattr(module, "__path__"):
                    for _, name, _ in pkgutil.iter_modules(module.__path__):
                        try:
                            submodule = importlib.import_module(f"{module_name}.{name}")
                            count += self._register_from_module(submodule)
                        except ImportError as e:
                            logger.warning(f"Failed to import {module_name}.{name}: {e}")
                            
            except ImportError as e:
                logger.debug(f"Subpackage {subpkg} not found: {e}")
        
        logger.info(f"Discovered {count} plugins")
        return count
    
    def _register_from_module(self, module: Any) -> int:
        """Register all plugins found in a module."""
        count = 0
        
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
                
            attr = getattr(module, attr_name)
            
            if not isinstance(attr, type):
                continue
            
            # Skip base classes
            if attr in (Plugin, EffectPlugin, TransitionPlugin, ProcessorPlugin, 
                       SubtitlePlugin, TextEffectPlugin, Node):
                continue
            
            try:
                if issubclass(attr, Node) and attr is not Node:
                    self.register_node(attr_name, attr)
                    count += 1
                elif issubclass(attr, Plugin) and attr not in (
                    Plugin, EffectPlugin, TransitionPlugin, ProcessorPlugin,
                    SubtitlePlugin, TextEffectPlugin
                ):
                    self.register(attr_name, attr)
                    count += 1
            except TypeError:
                pass
        
        return count
    
    def list_plugins(self) -> dict[str, list[str]]:
        """List all registered plugins by category."""
        return {
            "effects": list(self._effects.keys()),
            "transitions": list(self._transitions.keys()),
            "processors": list(self._processors.keys()),
            "nodes": list(self._nodes.keys()),
        }


def get_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def register_effect(name: str):
    """Decorator to register an effect plugin."""
    def decorator(cls: Type[EffectPlugin]) -> Type[EffectPlugin]:
        get_registry().register_effect(name, cls)
        return cls
    return decorator


def register_transition(name: str):
    """Decorator to register a transition plugin."""
    def decorator(cls: Type[TransitionPlugin]) -> Type[TransitionPlugin]:
        get_registry().register_transition(name, cls)
        return cls
    return decorator


def register_processor(name: str):
    """Decorator to register a processor plugin."""
    def decorator(cls: Type[ProcessorPlugin]) -> Type[ProcessorPlugin]:
        get_registry().register_processor(name, cls)
        return cls
    return decorator


def register_node(name: str):
    """Decorator to register a pipeline node."""
    def decorator(cls: Type[Node]) -> Type[Node]:
        get_registry().register_node(name, cls)
        return cls
    return decorator
