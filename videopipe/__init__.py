"""
VideoPipe - Professional Video Processing Pipeline for Instagram Content

A configurable, AI-powered video editing pipeline designed for continuous development.
No GUI - pure data pipeline architecture with extensible plugin system.
"""

__version__ = "0.1.0"
__author__ = "VideoPipe Team"

from videopipe.core.pipeline import Pipeline
from videopipe.core.context import PipelineContext
from videopipe.core.config import PipelineConfig

__all__ = [
    "Pipeline",
    "PipelineContext", 
    "PipelineConfig",
    "__version__",
]
