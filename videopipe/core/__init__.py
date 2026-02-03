"""
Core pipeline components - the heart of the video processing system.
"""

from videopipe.core.pipeline import Pipeline
from videopipe.core.node import Node, NodeResult
from videopipe.core.context import PipelineContext
from videopipe.core.config import PipelineConfig

__all__ = [
    "Pipeline",
    "Node",
    "NodeResult",
    "PipelineContext",
    "PipelineConfig",
]
