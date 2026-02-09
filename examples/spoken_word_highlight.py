#!/usr/bin/env python3
"""
Spoken Word Highlight Example

This example demonstrates how to highlight the currently spoken word
with a soft pill background.

Usage:
    python examples/spoken_word_highlight.py input.mp4 output.mov
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from videopipe.core.pipeline import Pipeline
from videopipe.core.context import PipelineContext
from videopipe.nodes import (
    LoadVideosNode,
    GenerateSubtitlesNode,
    ApplySpokenWordHighlightNode,
    RenderSubtitlesNode,
    ExportNode,
)


def main():
    if len(sys.argv) < 3:
        print("Usage: python spoken_word_highlight.py input.mp4 output.mov")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    
    context = PipelineContext(
        input_files=[input_path],
        output_path=output_path,
    )
    
    pipeline = Pipeline(name="SpokenWordHighlightPipeline")
    pipeline.add_node(LoadVideosNode())
    pipeline.add_node(GenerateSubtitlesNode(whisper_model="medium"))
    pipeline.add_node(ApplySpokenWordHighlightNode(
        effect="soft_pill",
        highlight_config={
            "bg_color": "#FDE68A",
            "bg_opacity": 0.85,
            "text_color": "#111827",
            "padding_x": 12,
            "padding_y": 4,
            "corner_radius": 8,
            "reveal_mode": "full",
        },
    ))
    pipeline.add_node(RenderSubtitlesNode(animated=False))
    pipeline.add_node(ExportNode(preset="prores_422_hq"))
    
    print("\nRunning pipeline...")
    results = pipeline.run(context)
    
    failed = [name for name, result in results.items() if not result.success]
    
    if failed:
        print(f"\nPipeline failed at: {', '.join(failed)}")
        for name in failed:
            if results[name].error:
                print(f"  {name}: {results[name].error}")
        sys.exit(1)
    else:
        print(f"\nSuccess! Output saved to: {output_path}")
    
    context.cleanup()


if __name__ == "__main__":
    main()
