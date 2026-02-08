"""
Pipeline nodes - pre-built processing nodes for common operations.

Nodes are the building blocks of the pipeline, each handling
a specific processing step.
"""

from videopipe.nodes.video_nodes import (
    LoadVideosNode,
    CropNode,
    PreviewModeNode,
    InVideoTransitionNode,
    MergeVideosNode,
    ExportNode,
    ApplyTransitionNode,
)
from videopipe.nodes.subtitle_nodes import (
    GenerateSubtitlesNode,
    RenderSubtitlesNode,
)
from videopipe.nodes.effect_nodes import (
    ApplyEffectsNode,
    ApplyNeonEffectNode,
    CreateNeonTextOverlay,
    AddSoundEffectNode,
    ChangeSpeedNode,
)

__all__ = [
    "LoadVideosNode",
    "CropNode",
    "PreviewModeNode",
    "InVideoTransitionNode",
    "MergeVideosNode",
    "ExportNode",
    "ApplyTransitionNode",
    "GenerateSubtitlesNode",
    "RenderSubtitlesNode",
    "ApplyEffectsNode",
    "ApplyNeonEffectNode",
    "CreateNeonTextOverlay",
    "AddSoundEffectNode",
    "ChangeSpeedNode",
]
