#!/usr/bin/env python3
"""
Instagram Neon Pipeline Example

This pipeline demonstrates:
1. Crop bottom by percentage
2. Multiple timed neon text overlays
3. Sound effects with fadeout
4. In-video transitions
5. ProRes 422 HQ export

Usage:
    python examples/instagram_neon_pipeline.py input.mp4 output.mov

Or with config:
    python examples/instagram_neon_pipeline.py --config configs/my_instagram_edit.yaml
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
    CropNode,
    PreviewModeNode,
    InVideoTransitionNode,
    CreateNeonTextOverlay,
    AddSoundEffectNode,
    ExportNode,
)
from videopipe.utils.fonts import get_font_path_for_config


def create_pipeline_from_args(args):
    """Create pipeline from command line arguments."""
    
    # Create context
    context = PipelineContext(
        input_files=[Path(args.input)],
        output_path=Path(args.output),
    )
    
    # Build pipeline
    pipeline = Pipeline(name="InstagramNeonPipeline")
    
    # 1. Load video
    pipeline.add_node(LoadVideosNode())
    
    # 2. Crop bottom by percentage
    if args.crop_bottom > 0:
        pipeline.add_node(CropNode(bottom=args.crop_bottom))
    
    # 3. In-video transition (optional)
    if args.transition_time:
        pipeline.add_node(InVideoTransitionNode(
            cut_time=args.transition_time,
            transition_type="slide",
            transition_duration=0.3,
        ))
    
    # 4. Add neon text overlays
    # Default texts based on your requirements:
    # - 1s: "378" then "agentic"
    # - 4s: "five !"
    # - 8s: "24 months"
    
    text_overlays = [
        {"name": "t1", "text": "378", "start_time": 1.0, "duration": 0.5},
        {"name": "t2", "text": "agentic", "start_time": 1.5, "duration": 2.0},
        {"name": "t3", "text": "five !", "start_time": 4.0, "duration": 2.0},
        {"name": "t4", "text": "24 months", "start_time": 8.0, "duration": 2.5},
    ]
    
    for overlay in text_overlays:
        pipeline.add_node(CreateNeonTextOverlay(
            name=overlay["name"],
            text=overlay["text"],
            start_time=overlay["start_time"],
            duration=overlay["duration"],
            position=("center", "center"),
            futuristic=True,
        ))
    
    # 5. Add sound effects (if folder exists)
    effects_folder = Path(args.effects_folder) if args.effects_folder else Path("effects_sound")
    if effects_folder.exists():
        # You can customize these sound mappings
        sounds = [
            {"name": "whoosh.mp3", "time": 1.0, "volume": 0.8},
            {"name": "impact.wav", "time": 4.0, "volume": 0.7},
        ]
        
        # Filter to only existing files
        sounds = [s for s in sounds if (effects_folder / s["name"]).exists()]
        
        if sounds:
            pipeline.add_node(AddSoundEffectNode(
                effects_folder=effects_folder,
                sounds=sounds,
                fadeout=True,
                fadeout_duration=1.0,
            ))
    
    # 6. Export
    pipeline.add_node(ExportNode(preset=args.preset))
    
    return pipeline, context


def create_pipeline_from_config(config_path: Path):
    """Create pipeline from YAML config file."""
    import yaml
    import json
    
    # #region agent log
    LOG_PATH = "/Users/karel.cervicek/Documents/projects/video_procesing/.cursor/debug.log"
    def debug_log(hyp, loc, msg, data):
        with open(LOG_PATH, "a") as f: f.write(json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": __import__("time").time()}) + "\n")
    # #endregion
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # #region agent log
    debug_log("A", "instagram_neon_pipeline.py:119", "YAML parsed input_files", {"raw_input_files": config.get("input_files", []), "type": str(type(config.get("input_files", [])))})
    # #endregion
    
    # Create context
    input_paths = [Path(f) for f in config.get("input_files", [])]
    
    # #region agent log
    debug_log("C", "instagram_neon_pipeline.py:123", "Path objects created", {"paths": [str(p) for p in input_paths], "exists": [p.exists() for p in input_paths]})
    # #endregion
    
    context = PipelineContext(
        input_files=input_paths,
        output_path=Path(config["output_path"]) if config.get("output_path") else None,
    )
    context.config = config
    
    # Build pipeline with proper dependency chain
    # Order: load_videos -> crop -> text_overlays -> sound_effects -> export
    pipeline = Pipeline(name="ConfiguredInstagramPipeline")
    
    # Track the last node name for chaining dependencies
    last_node = None
    
    # #region agent log
    import json as _jd
    import time as _td
    _DLP = "/Users/karel.cervicek/Documents/projects/video_procesing/.cursor/debug.log"
    def _pdbg(hyp, loc, msg, data):
        with open(_DLP, "a") as f: f.write(_jd.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": _td.time()}) + "\n")
    # #endregion
    
    # 1. Always load videos first
    pipeline.add_node(LoadVideosNode())
    last_node = "load_videos"
    
    # 2. Preview mode (if enabled, trims video early for fast rendering)
    preview_config = config.get("preview", {})
    preview_enabled = config.get("preview_mode", False)
    
    # #region agent log
    _pdbg("A", "pipeline:preview_config_read", "Preview config values", {"preview_mode": preview_enabled, "preview_config": str(preview_config), "config_keys": list(config.keys())})
    # #endregion
    
    if preview_enabled:
        # #region agent log
        _pdbg("B", "pipeline:preview_node_adding", "Adding PreviewModeNode", {"start_time": preview_config.get("start_time", 0.0), "end_time": preview_config.get("end_time")})
        # #endregion
        preview_node = PreviewModeNode(
            enabled=True,
            start_time=preview_config.get("start_time", 0.0),
            end_time=preview_config.get("end_time"),
            scale=preview_config.get("scale"),
        )
        preview_node._dependencies = [last_node]
        pipeline.add_node(preview_node)
        last_node = "preview_mode"
        print(f"PREVIEW MODE: Rendering {preview_config.get('start_time', 0)}s - {preview_config.get('end_time', 'end')}s only")
    else:
        # #region agent log
        _pdbg("A", "pipeline:preview_disabled", "Preview mode is DISABLED", {"preview_enabled": preview_enabled})
        # #endregion
    
    # 3. Crop if specified (depends on load_videos)
    crop_config = config.get("crop", {})
    if any(crop_config.get(k, 0) > 0 for k in ["top", "bottom", "left", "right"]):
        crop_node = CropNode(**crop_config)
        crop_node._dependencies = [last_node]  # Chain to previous
        pipeline.add_node(crop_node)
        last_node = "crop"
    
    # 4. In-video transitions (depends on previous)
    transitions = config.get("in_video_transitions", [])
    if transitions:
        transition_node = InVideoTransitionNode(
            cut_times=transitions,
            transition_type=config.get("transition_type", "slide"),
            transition_duration=config.get("transition_duration", 0.3),
        )
        transition_node._dependencies = [last_node]
        pipeline.add_node(transition_node)
        last_node = "in_video_transition"
    
    # 5. Resolve global font (auto-download if needed)
    global_neon_settings = config.get("neon_settings", {})
    global_font = global_neon_settings.get("font", "Bebas Neue")
    
    # #region agent log
    _pdbg("F", "pipeline:font_resolution_start", "Starting font resolution", {"global_font": global_font})
    # #endregion
    
    resolved_font = get_font_path_for_config(global_font, auto_download=True)
    
    # #region agent log
    _pdbg("F", "pipeline:font_resolution_result", "Font resolution result", {"resolved_font": resolved_font, "changed": resolved_font != global_font})
    # #endregion
    
    if resolved_font != global_font:
        print(f"Font resolved: '{global_font}' -> '{resolved_font}'")
        global_neon_settings["font"] = resolved_font
        config["neon_settings"] = global_neon_settings
    else:
        # Font resolution failed - try to find a fallback system font
        print(f"WARNING: Font '{global_font}' not found, trying system fallback...")
        # Note: Path is already imported at top of file
        fallback_fonts = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/SFNSMono.ttf",
        ]
        for fb_font in fallback_fonts:
            if Path(fb_font).exists():
                print(f"Using fallback font: {fb_font}")
                global_neon_settings["font"] = fb_font
                config["neon_settings"] = global_neon_settings
                resolved_font = fb_font
                # #region agent log
                _pdbg("F", "pipeline:font_fallback_used", "Using fallback font", {"fallback": fb_font})
                # #endregion
                break
    
    # 6. Text overlays - each depends on the previous (chain them)
    text_overlays = config.get("text_overlays", [])
    for i, overlay in enumerate(text_overlays):
        node_name = overlay.get("name", f"text_{i}")
        
        # #region agent log
        _pdbg("E", "pipeline:text_overlay_config", "Text overlay from config", {"name": node_name, "text": overlay.get("text"), "width_percent_in_config": overlay.get("width_percent"), "overlay_keys": list(overlay.keys())})
        # #endregion
        
        # Merge global neon settings with overlay-specific settings
        overlay_neon_config = {**global_neon_settings, **(overlay.get("neon_config") or {})}
        
        # Resolve font for this specific overlay if different
        overlay_font = overlay_neon_config.get("font", global_font)
        if overlay_font != resolved_font and overlay_font != global_font:
            overlay_neon_config["font"] = get_font_path_for_config(overlay_font, auto_download=True)
        
        width_percent_value = overlay.get("width_percent")
        # #region agent log
        _pdbg("E", "pipeline:text_overlay_creating", "Creating CreateNeonTextOverlay", {"name": node_name, "width_percent_value": width_percent_value, "type": str(type(width_percent_value))})
        # #endregion
        
        text_node = CreateNeonTextOverlay(
            name=node_name,
            text=overlay["text"],
            start_time=overlay.get("start_time", 0),
            duration=overlay.get("duration", 2.0),
            position=tuple(overlay.get("position", ["center", "center"])),
            futuristic=overlay.get("futuristic", True),
            neon_config=overlay_neon_config,
            width_percent=width_percent_value,
        )
        text_node._dependencies = [last_node]  # Chain to previous
        pipeline.add_node(text_node)
        last_node = node_name
    
    # 7. Sound effects (depends on last text overlay or previous)
    sounds_config = config.get("sound_effects", {})
    if sounds_config.get("sounds"):
        sound_node = AddSoundEffectNode(
            effects_folder=Path(sounds_config.get("folder", "effects_sound")),
            sounds=sounds_config["sounds"],
            fadeout=sounds_config.get("fadeout", True),
            fadeout_duration=sounds_config.get("fadeout_duration", 1.0),
        )
        sound_node._dependencies = [last_node]  # Chain to previous
        pipeline.add_node(sound_node)
        last_node = "add_sound_effects"
    
    # 8. Export (depends on everything before it)
    export_config = config.get("export", {})
    export_node = ExportNode(
        preset=export_config.get("preset", "prores_422_hq"),
    )
    export_node._dependencies = [last_node]  # Chain to previous (always last)
    pipeline.add_node(export_node)
    
    return pipeline, context


def main():
    parser = argparse.ArgumentParser(
        description="Instagram Neon Pipeline - Process video with neon effects"
    )
    parser.add_argument(
        "-i", "--input",
        help="Input video file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output video file",
    )
    parser.add_argument(
        "-c", "--config",
        help="YAML configuration file",
    )
    parser.add_argument(
        "--crop-bottom",
        type=float,
        default=0,
        help="Percentage to crop from bottom (0-50)",
    )
    parser.add_argument(
        "--transition-time",
        type=float,
        help="Time for in-video transition (seconds)",
    )
    parser.add_argument(
        "--effects-folder",
        default="effects_sound",
        help="Folder containing sound effects",
    )
    parser.add_argument(
        "--preset",
        default="prores_422_hq",
        help="Export preset (default: prores_422_hq)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pipeline without running",
    )
    
    args = parser.parse_args()
    
    # Use config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        pipeline, context = create_pipeline_from_config(config_path)
    else:
        # Require input/output for non-config mode
        if not args.input or not args.output:
            parser.print_help()
            print("\nError: --input and --output required (or use --config)")
            sys.exit(1)
        
        if not Path(args.input).exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
        
        pipeline, context = create_pipeline_from_args(args)
    
    # Dry run
    if args.dry_run:
        pipeline.dry_run(context)
        return
    
    # Run pipeline
    print("\n" + "=" * 60)
    print("Instagram Neon Pipeline")
    print("=" * 60)
    
    results = pipeline.run(context)
    
    # Report
    failed = [name for name, result in results.items() if not result.success]
    
    if failed:
        print(f"\nPipeline FAILED at: {', '.join(failed)}")
        for name in failed:
            if results[name].error:
                print(f"  {name}: {results[name].error}")
        sys.exit(1)
    else:
        print(f"\nSuccess! Output: {context.output_path}")
    
    context.cleanup()


if __name__ == "__main__":
    main()
