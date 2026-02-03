#!/usr/bin/env python3
"""
Merge Videos with Transitions Example

This example shows how to merge multiple video clips
with professional slide transitions.

Usage:
    python examples/merge_with_transitions.py clip1.mp4 clip2.mp4 clip3.mp4 -o output.mov
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from videopipe.core.pipeline import Pipeline
from videopipe.core.context import PipelineContext
from videopipe.nodes import (
    LoadVideosNode,
    ApplyTransitionNode,
    ExportNode,
)


def main():
    parser = argparse.ArgumentParser(
        description="Merge videos with slide transitions"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input video files",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--transition",
        default="slide",
        choices=["slide", "crossfade", "wipe"],
        help="Transition type (default: slide)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.3,
        help="Transition duration in seconds (default: 0.3)",
    )
    parser.add_argument(
        "--direction",
        default="left",
        choices=["left", "right", "up", "down"],
        help="Slide direction (default: left)",
    )
    parser.add_argument(
        "--preset",
        default="prores_422_hq",
        help="Export preset (default: prores_422_hq)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    input_paths = []
    for path_str in args.inputs:
        path = Path(path_str)
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
        input_paths.append(path)
    
    output_path = Path(args.output)
    
    print(f"Merging {len(input_paths)} videos:")
    for path in input_paths:
        print(f"  - {path}")
    print(f"\nTransition: {args.transition} ({args.duration}s)")
    print(f"Output: {output_path}")
    
    # Create context
    context = PipelineContext(
        input_files=input_paths,
        output_path=output_path,
    )
    
    # Configure transition
    context.config["transition_settings"] = {
        "type": args.transition,
        "duration": args.duration,
        "direction": args.direction,
        "easing": "ease_out_expo",
        "motion_blur": True,
    }
    
    # Build pipeline
    pipeline = Pipeline(name="MergeWithTransitions")
    
    pipeline.add_node(LoadVideosNode())
    pipeline.add_node(ApplyTransitionNode(
        transition_type=args.transition,
        transition_duration=args.duration,
        transition_params={
            "direction": args.direction,
            "motion_blur": True,
            "shadow": True,
        },
    ))
    pipeline.add_node(ExportNode(
        preset=args.preset,
    ))
    
    # Run
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
