#!/usr/bin/env python3
"""
Simple Pipeline Example

This example demonstrates how to use VideoPipe to process a video:
1. Load a video
2. Generate subtitles
3. Render subtitles with animation
4. Export to ProRes 422 HQ

Usage:
    python examples/simple_pipeline.py input.mp4 output.mov
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
    ExportNode,
)


def main():
    if len(sys.argv) < 3:
        print("Usage: python simple_pipeline.py input.mp4 output.mov")
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
    
    # Build the pipeline
    pipeline = Pipeline(name="SimplePipeline")
    
    # Add nodes
    pipeline.add_node(LoadVideosNode())
    pipeline.add_node(GenerateSubtitlesNode(
        whisper_model="medium",  # Use 'small' for faster processing
    ))
    pipeline.add_node(RenderSubtitlesNode(
        animated=True,  # Enable word-by-word animation
    ))
    pipeline.add_node(ExportNode(
        preset="prores_422_hq",  # ProRes 422 HQ, 10-bit
    ))
    
    # Show pipeline structure
    print("\nPipeline structure:")
    pipeline.dry_run(context)
    
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
    
    # Cleanup
    context.cleanup()


if __name__ == "__main__":
    main()
