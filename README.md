# VideoPipe

**Professional Video Processing Pipeline for Instagram Content**

A configurable, AI-powered video editing pipeline designed for continuous development. No GUI - pure data pipeline architecture with an extensible plugin system.

## Features

- **Auto Subtitles**: Generate subtitles using OpenAI Whisper with word-level timing
- **Text Animation Effects**: Kinetic typography with word-by-word reveal
- **Spoken Word Highlighting**: Soft pill background follows the current word
- **Neon Effects**: Futuristic neon green glow effects for special words
- **Professional Transitions**: Quick slide transitions with motion blur
- **ProRes 422 HQ Export**: 10-bit professional quality output
- **Extensible Architecture**: Plugin-based system for adding new effects and transitions

## Installation

### Prerequisites

1. **Python 3.10+**
2. **FFmpeg** (required for video processing)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Video Processing

```bash
# Process a single video with subtitles
python cli.py process -i input.mp4 -o output.mov --subtitles

# Merge videos with slide transition
python cli.py process -i clip1.mp4 clip2.mp4 -o merged.mov --transition slide

# Use Instagram-optimized settings
python cli.py process -i video.mp4 -o output.mp4 --config configs/instagram_reels.yaml
```

### Generate Subtitles Only

```bash
python cli.py subtitles -i video.mp4 -o subtitles.srt --model medium
```

### Get Video Information

```bash
python cli.py info video.mp4
```

### List Export Presets

```bash
python cli.py presets
```

## Configuration

Configuration files use YAML format. See `configs/default.yaml` for all options.

### Example Configuration

```yaml
# my_project.yaml
input_files:
  - clip1.mp4
  - clip2.mp4
output_path: output.mov

# Export as ProRes 422 HQ (10-bit)
export_settings:
  codec: prores_ks
  profile: 3  # HQ
  pix_fmt: yuv422p10le

# Subtitle styling
subtitle_settings:
  font: Arial-Bold
  font_size: 48
  color: white
  stroke_width: 2

# Spoken-word highlight (karaoke-style)
spoken_word_highlight:
  enabled: true
  effect: soft_pill
  bg_color: "#FDE68A"
  bg_opacity: 0.85
  text_color: "#111827"
  padding_x: 12
  padding_y: 4
  corner_radius: 8
  reveal_mode: full

# Special words get neon effects
special_words:
  amazing:
    type: neon
    color: "#39FF14"
  epic:
    type: neon
    color: "#00FFFF"

# Quick slide transitions
transition_settings:
  type: slide
  duration: 0.3
  direction: left
```

Run with config:

```bash
python cli.py process --config my_project.yaml
```

## Python API

```python
from pathlib import Path
from videopipe.core.pipeline import Pipeline
from videopipe.core.context import PipelineContext
from videopipe.nodes import (
    LoadVideosNode,
    GenerateSubtitlesNode,
    RenderSubtitlesNode,
    ApplySpokenWordHighlightNode,
    ApplyTransitionNode,
    ExportNode,
)

# Create context
context = PipelineContext(
    input_files=[Path("clip1.mp4"), Path("clip2.mp4")],
    output_path=Path("output.mov"),
)

# Mark special words for neon effect
context.add_special_word("amazing", {"type": "neon", "color": "#39FF14"})

# Build pipeline
pipeline = Pipeline()
pipeline.add_node(LoadVideosNode())
pipeline.add_node(GenerateSubtitlesNode(whisper_model="medium"))
pipeline.add_node(ApplySpokenWordHighlightNode(
    effect="soft_pill",
    highlight_config={"reveal_mode": "full"},
))
pipeline.add_node(RenderSubtitlesNode(animated=True))
pipeline.add_node(ApplyTransitionNode(
    transition_type="slide",
    transition_duration=0.3,
))
pipeline.add_node(ExportNode(preset="prores_422_hq"))

# Run pipeline
results = pipeline.run(context)
```

## Architecture

```
videopipe/
├── core/                 # Core pipeline components
│   ├── pipeline.py      # DAG-based pipeline engine
│   ├── node.py          # Base node class
│   ├── context.py       # Pipeline context/state
│   └── config.py        # Configuration management
├── plugins/              # Plugin system
│   ├── base.py          # Plugin base classes
│   └── registry.py      # Plugin registry
├── effects/              # Visual effects
│   ├── text_effects.py  # Text animations
│   └── neon.py          # Neon glow effects
├── transitions/          # Video transitions
│   ├── base.py          # Base transitions
│   └── slide.py         # Slide transitions
├── subtitles/            # Subtitle processing
│   ├── whisper_stt.py   # Whisper transcription
│   └── renderer.py      # Subtitle rendering
├── video/                # Video operations
│   ├── clip.py          # Clip loading
│   ├── merge.py         # Video merging
│   └── export.py        # Export (ProRes, H.264, etc.)
└── nodes/                # Pre-built pipeline nodes
    ├── video_nodes.py   # Load, merge, export nodes
    ├── subtitle_nodes.py # Subtitle nodes
    └── effect_nodes.py  # Effect nodes
```

## Extending the Pipeline

### Adding a Custom Effect

```python
from videopipe.plugins.base import EffectPlugin, PluginMetadata
from videopipe.plugins.registry import register_effect

@register_effect("my_effect")
class MyCustomEffect(EffectPlugin):
    metadata = PluginMetadata(
        name="MyEffect",
        version="1.0.0",
        description="My custom video effect",
    )
    
    def apply(self, clip, context, **kwargs):
        # Apply your effect to the clip
        return modified_clip
```

### Adding a Custom Transition

```python
from videopipe.transitions.base import Transition

class MyTransition(Transition):
    def make_frame(self, clip_a, clip_b, t, progress):
        # Generate transition frame
        frame_a = clip_a.get_frame(clip_a.duration - self.duration + t)
        frame_b = clip_b.get_frame(t)
        
        # Your blending logic here
        return blended_frame
```

### Adding a Custom Node

```python
from videopipe.core.node import Node, NodeResult

class MyProcessingNode(Node):
    def __init__(self):
        super().__init__(
            name="my_processing",
            dependencies=["load_videos"],
        )
    
    def process(self, context):
        clip = context.get_main_clip()
        # Your processing logic
        context.set_main_clip(modified_clip)
        return NodeResult.success_result(output=modified_clip)
```

## Export Presets

| Preset | Codec | Container | Use Case |
|--------|-------|-----------|----------|
| `prores_422_hq` | ProRes | .mov | Professional editing, archival |
| `prores_422_lt` | ProRes | .mov | Lighter files, editing |
| `prores_4444` | ProRes | .mov | With alpha channel |
| `h264_high` | H.264 | .mp4 | High quality delivery |
| `h264_web` | H.264 | .mp4 | Web streaming |
| `h265_high` | H.265 | .mp4 | Modern high quality |
| `instagram` | H.264 | .mp4 | Instagram Reels |

## Performance Tips

1. **Whisper Model Selection**:
   - `tiny`/`base`: Fast, lower accuracy
   - `small`: Good balance
   - `medium`: Recommended for quality
   - `large`: Best accuracy, slowest

2. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster transcription

3. **Intermediate Files**: Use ProRes for editing, H.264/H.265 for final delivery

## License

MIT License - see LICENSE file for details.

## Contributing

This project is designed for continuous development. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
