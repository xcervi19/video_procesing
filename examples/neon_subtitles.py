#!/usr/bin/env python3
"""
Neon Subtitles Example

This example demonstrates how to create videos with
special neon effects on specific words.

Usage:
    python examples/neon_subtitles.py input.mp4 output.mov
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
    RenderSubtitlesNode,
    ApplyNeonEffectNode,
    ExportNode,
)


def main():
    if len(sys.argv) < 3:
        print("Usage: python neon_subtitles.py input.mp4 output.mov")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    
    # Create pipeline context
    context = PipelineContext(
        input_files=[input_path],
        output_path=output_path,
    )
    
    # Mark words for neon effect
    neon_words = [
        "amazing", "incredible", "awesome", "epic",
        "wow", "insane", "crazy", "perfect",
    ]
    
    for word in neon_words:
        context.add_special_word(word, {
            "type": "neon",
            "color": "#39FF14",  # Neon green
            "glow_intensity": 1.5,
            "pulse": True,
        })
    
    print(f"\nNeon effect configured for: {', '.join(neon_words)}")
    
    # Configure neon settings in context
    context.config["neon_settings"] = {
        "color": "#39FF14",
        "glow_intensity": 1.5,
        "glow_radius": 12,
        "pulse": True,
        "pulse_speed": 2.0,
    }
    
    # Build the pipeline
    pipeline = Pipeline(name="NeonSubtitlesPipeline")
    
    pipeline.add_node(LoadVideosNode())
    pipeline.add_node(GenerateSubtitlesNode(
        whisper_model="medium",
    ))
    pipeline.add_node(ApplyNeonEffectNode(
        target_words=neon_words,
    ))
    pipeline.add_node(RenderSubtitlesNode(
        animated=True,
    ))
    pipeline.add_node(ExportNode(
        preset="prores_422_hq",
    ))
    
    # Run the pipeline
    print("\nRunning pipeline...")
    results = pipeline.run(context)
    
    # Check results
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
