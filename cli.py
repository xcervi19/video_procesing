#!/usr/bin/env python3
"""
VideoPipe CLI - Command-line interface for the video processing pipeline.

Usage:
    videopipe process -i video1.mp4 video2.mp4 -o output.mov
    videopipe process --config pipeline.yaml
    videopipe subtitles -i video.mp4 -o subtitles.srt
    videopipe info video.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("videopipe")


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure logging based on verbosity settings."""
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def cmd_process(args):
    """Process videos through the pipeline."""
    from videopipe.core.config import PipelineConfig
    from videopipe.core.context import PipelineContext, create_context_from_config
    from videopipe.core.pipeline import Pipeline
    from videopipe.nodes import (
        LoadVideosNode,
        MergeVideosNode,
        ApplyTransitionNode,
        GenerateSubtitlesNode,
        RenderSubtitlesNode,
        ExportNode,
    )
    
    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = PipelineConfig.from_file(args.config)
    else:
        config = PipelineConfig()
        config.input_files = [Path(f) for f in args.input]
        config.output_path = Path(args.output) if args.output else None
        
        if args.preset:
            # Export preset overrides
            pass
    
    # Override with CLI arguments
    if args.input:
        config.input_files = [Path(f) for f in args.input]
    if args.output:
        config.output_path = Path(args.output)
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    
    # Validate config
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(error)
        sys.exit(1)
    
    # Create context from config
    context = create_context_from_config(config.to_dict())
    
    # Add special words from CLI
    if args.special_words:
        for word in args.special_words:
            context.add_special_word(word, {
                "type": "neon",
                "color": "#39FF14",  # Neon green
            })
    
    # Build pipeline
    pipeline = Pipeline(name="VideoPipeline")
    
    # Add nodes based on options
    pipeline.add_node(LoadVideosNode())
    
    if args.transition and len(config.input_files) > 1:
        pipeline.add_node(ApplyTransitionNode(
            transition_type=args.transition,
            transition_duration=args.transition_duration or 0.5,
        ))
    elif len(config.input_files) > 1:
        pipeline.add_node(MergeVideosNode())
    
    if args.subtitles:
        pipeline.add_node(GenerateSubtitlesNode(
            whisper_model=config.whisper_model,
        ))
        pipeline.add_node(RenderSubtitlesNode(
            animated=not args.no_animate,
        ))
    
    pipeline.add_node(ExportNode(
        preset=args.preset or "prores_422_hq",
    ))
    
    # Show dry run if requested
    if args.dry_run:
        pipeline.dry_run(context)
        return
    
    # Run pipeline
    logger.info("Starting video processing pipeline...")
    results = pipeline.run(context)
    
    # Report results
    failed = [name for name, result in results.items() if not result.success]
    
    if failed:
        logger.error(f"Pipeline failed at: {', '.join(failed)}")
        for name in failed:
            if results[name].error:
                logger.error(f"  {name}: {results[name].error}")
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully!")
        if context.output_path:
            logger.info(f"Output: {context.output_path}")
    
    # Cleanup
    context.cleanup()


def cmd_subtitles(args):
    """Generate subtitles from video."""
    from videopipe.subtitles.whisper_stt import WhisperTranscriber
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else input_path.with_suffix(".srt")
    
    logger.info(f"Generating subtitles from: {input_path}")
    logger.info(f"Whisper model: {args.model}")
    
    transcriber = WhisperTranscriber(
        model=args.model,
        language=args.language,
    )
    
    srt_content = transcriber.transcribe_to_srt(
        input_path,
        output_path=output_path,
    )
    
    logger.info(f"Subtitles saved to: {output_path}")
    
    if args.print:
        print("\n" + srt_content)


def cmd_info(args):
    """Display video file information."""
    from videopipe.video.clip import get_clip_info
    
    for input_path in args.input:
        path = Path(input_path)
        
        if not path.exists():
            logger.error(f"File not found: {path}")
            continue
        
        try:
            info = get_clip_info(path)
            
            print(f"\n{'='*60}")
            print(f"File: {info.path.name}")
            print(f"{'='*60}")
            print(f"Resolution: {info.width}x{info.height}")
            print(f"FPS: {info.fps:.2f}")
            print(f"Duration: {info.duration:.2f}s ({info.duration/60:.1f}min)")
            print(f"Video Codec: {info.codec}")
            print(f"Pixel Format: {info.pix_fmt}")
            if info.bitrate:
                print(f"Bitrate: {info.bitrate/1_000_000:.1f} Mbps")
            if info.audio_codec:
                print(f"Audio Codec: {info.audio_codec}")
                if info.audio_sample_rate:
                    print(f"Audio Sample Rate: {info.audio_sample_rate} Hz")
                if info.audio_channels:
                    print(f"Audio Channels: {info.audio_channels}")
            print()
            
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")


def cmd_presets(args):
    """List available export presets."""
    from videopipe.video.export import get_available_presets
    
    presets = get_available_presets()
    
    print("\nAvailable Export Presets:")
    print("=" * 60)
    
    for name, preset in presets.items():
        print(f"\n{name}:")
        print(f"  Name: {preset.name}")
        print(f"  Codec: {preset.codec}")
        print(f"  Container: {preset.container}")
        print(f"  Pixel Format: {preset.pixel_format}")
        if preset.profile is not None:
            print(f"  Profile: {preset.profile}")
        if preset.quality is not None:
            print(f"  Quality: {preset.quality}")
        print(f"  Audio: {preset.audio_codec}")


def main():
    parser = argparse.ArgumentParser(
        description="VideoPipe - Professional Video Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video with subtitles
  videopipe process -i video.mp4 -o output.mov --subtitles

  # Merge videos with slide transition  
  videopipe process -i clip1.mp4 clip2.mp4 -o merged.mov --transition slide

  # Generate subtitles only
  videopipe subtitles -i video.mp4 -o subs.srt --model medium

  # Get video info
  videopipe info video.mp4

  # Use configuration file
  videopipe process --config pipeline.yaml
        """,
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ==================== Process Command ====================
    process_parser = subparsers.add_parser(
        "process",
        help="Process videos through the pipeline",
    )
    process_parser.add_argument(
        "-i", "--input",
        nargs="+",
        help="Input video files",
    )
    process_parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )
    process_parser.add_argument(
        "-c", "--config",
        help="Configuration file (YAML or JSON)",
    )
    process_parser.add_argument(
        "--preset",
        default="prores_422_hq",
        help="Export preset (default: prores_422_hq)",
    )
    process_parser.add_argument(
        "--subtitles",
        action="store_true",
        help="Generate and render subtitles",
    )
    process_parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Disable subtitle animation",
    )
    process_parser.add_argument(
        "--whisper-model",
        default="medium",
        help="Whisper model for transcription (default: medium)",
    )
    process_parser.add_argument(
        "--transition",
        help="Transition type (slide, crossfade, wipe)",
    )
    process_parser.add_argument(
        "--transition-duration",
        type=float,
        help="Transition duration in seconds",
    )
    process_parser.add_argument(
        "--special-words",
        nargs="+",
        help="Words to highlight with neon effect",
    )
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    process_parser.set_defaults(func=cmd_process)
    
    # ==================== Subtitles Command ====================
    subs_parser = subparsers.add_parser(
        "subtitles",
        help="Generate subtitles from video",
    )
    subs_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input video file",
    )
    subs_parser.add_argument(
        "-o", "--output",
        help="Output SRT file path",
    )
    subs_parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model (tiny, base, small, medium, large)",
    )
    subs_parser.add_argument(
        "--language",
        help="Force specific language (auto-detect if not specified)",
    )
    subs_parser.add_argument(
        "--print",
        action="store_true",
        help="Print subtitles to console",
    )
    subs_parser.set_defaults(func=cmd_subtitles)
    
    # ==================== Info Command ====================
    info_parser = subparsers.add_parser(
        "info",
        help="Display video file information",
    )
    info_parser.add_argument(
        "input",
        nargs="+",
        help="Video files to analyze",
    )
    info_parser.set_defaults(func=cmd_info)
    
    # ==================== Presets Command ====================
    presets_parser = subparsers.add_parser(
        "presets",
        help="List available export presets",
    )
    presets_parser.set_defaults(func=cmd_presets)
    
    # Parse and execute
    args = parser.parse_args()
    
    setup_logging(
        verbose=getattr(args, 'verbose', False),
        quiet=getattr(args, 'quiet', False),
    )
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
