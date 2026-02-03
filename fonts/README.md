# Fonts Directory

This folder stores fonts used by the video processing pipeline.

## Auto-Download

Fonts specified in your config (like "Bebas Neue") will be **automatically downloaded from Google Fonts** if not found locally.

Just specify the font name in your config:

```yaml
neon_settings:
  font: "Bebas Neue"    # Will be auto-downloaded
```

## Supported Fonts (Auto-Download)

Common fonts that can be auto-downloaded:
- Bebas Neue (recommended for tech/minimalist)
- Montserrat
- Inter
- Roboto
- Orbitron
- Rajdhani
- Exo 2
- Oswald
- Poppins
- Fira Code
- JetBrains Mono
- Source Code Pro

## Manual Installation

You can also manually place `.ttf` or `.otf` files in this folder. The pipeline will find them automatically.

## Using System Fonts

To use a system font, provide the full path:

```yaml
neon_settings:
  font: "/System/Library/Fonts/Supplemental/Arial.ttf"
```
