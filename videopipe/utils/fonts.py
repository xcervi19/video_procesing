"""
Font utilities for the video processing pipeline.

Provides automatic font downloading from Google Fonts.
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Default fonts directory in the project
DEFAULT_FONTS_DIR = Path(__file__).parent.parent.parent / "fonts"

# Google Fonts download URL template
GOOGLE_FONTS_URL = "https://fonts.google.com/download?family={family}"

# Common font name mappings (display name -> URL family name)
FONT_FAMILY_MAP = {
    "bebas neue": "Bebas+Neue",
    "bebasneue": "Bebas+Neue",
    "montserrat": "Montserrat",
    "inter": "Inter",
    "roboto": "Roboto",
    "orbitron": "Orbitron",
    "rajdhani": "Rajdhani",
    "exo 2": "Exo+2",
    "exo2": "Exo+2",
    "oswald": "Oswald",
    "poppins": "Poppins",
    "fira code": "Fira+Code",
    "jetbrains mono": "JetBrains+Mono",
    "source code pro": "Source+Code+Pro",
}


def get_fonts_dir() -> Path:
    """Get the fonts directory, creating it if needed."""
    fonts_dir = DEFAULT_FONTS_DIR
    fonts_dir.mkdir(parents=True, exist_ok=True)
    return fonts_dir


def normalize_font_name(font_name: str) -> str:
    """Normalize font name for lookup."""
    return font_name.lower().strip()


def get_google_fonts_url(font_name: str) -> str:
    """Get the Google Fonts download URL for a font family."""
    normalized = normalize_font_name(font_name)
    
    # Check if we have a mapping
    if normalized in FONT_FAMILY_MAP:
        family = FONT_FAMILY_MAP[normalized]
    else:
        # Convert to URL format: "Bebas Neue" -> "Bebas+Neue"
        family = font_name.strip().replace(" ", "+")
    
    return GOOGLE_FONTS_URL.format(family=family)


def find_font_file(font_name: str, search_dirs: Optional[list[Path]] = None) -> Optional[Path]:
    """
    Find a font file by name in various locations.
    
    Args:
        font_name: Font name or path
        search_dirs: Additional directories to search
        
    Returns:
        Path to font file if found, None otherwise
    """
    # If it's already a valid path, return it
    font_path = Path(font_name)
    if font_path.exists() and font_path.suffix.lower() in ('.ttf', '.otf', '.ttc'):
        return font_path
    
    # Normalize the font name for searching
    normalized = normalize_font_name(font_name)
    base = normalized.replace(" ", "")
    search_patterns = [
        f"{font_name}*.ttf",
        f"{font_name}*.otf",
        f"{base}*.ttf",
        f"{base}*.otf",
        f"{font_name.replace(' ', '-')}*.ttf",
        f"{font_name.replace(' ', '-')}*.otf",
        f"{base}.ttf",  # exact lowercase filename (e.g. bebasneue.ttf)
        f"{base}.otf",
    ]
    
    # Build search directories (project fonts first)
    dirs_to_search = [get_fonts_dir()]
    
    if search_dirs:
        dirs_to_search.extend(search_dirs)
    
    # Add system font directories
    system_dirs = [
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path.home() / "Library/Fonts",
        Path("/usr/share/fonts/truetype"),
        Path("/usr/share/fonts"),
    ]
    dirs_to_search.extend([d for d in system_dirs if d.exists()])
    
    # Search for the font
    for search_dir in dirs_to_search:
        if not search_dir.exists():
            continue
            
        for pattern in search_patterns:
            matches = list(search_dir.glob(pattern))
            if matches:
                # Prefer Regular weight
                for match in matches:
                    if 'regular' in match.stem.lower() or match.stem.lower().endswith(normalized.replace(' ', '')):
                        return match
                return matches[0]
    
    return None


def download_google_font(font_name: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Download a font from Google Fonts.
    
    Args:
        font_name: Name of the font to download (e.g., "Bebas Neue")
        output_dir: Directory to save the font (defaults to project fonts dir)
        
    Returns:
        Path to the downloaded font file, or None if failed
    """
    if output_dir is None:
        output_dir = get_fonts_dir()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = get_google_fonts_url(font_name)
    logger.info(f"Downloading font '{font_name}' from Google Fonts...")
    
    try:
        # Download the zip file
        zip_path = output_dir / f"{font_name.replace(' ', '_')}_download.zip"
        
        # Add headers to avoid being blocked
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        )
        
        with urllib.request.urlopen(request, timeout=30) as response:
            with open(zip_path, 'wb') as f:
                f.write(response.read())
        
        # Extract the zip file
        font_files = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith(('.ttf', '.otf')):
                    # Extract to output directory
                    extracted_name = Path(file_info.filename).name
                    extracted_path = output_dir / extracted_name
                    
                    with zip_ref.open(file_info) as src, open(extracted_path, 'wb') as dst:
                        dst.write(src.read())
                    
                    font_files.append(extracted_path)
                    logger.info(f"Extracted: {extracted_name}")
        
        # Clean up zip file
        zip_path.unlink()
        
        if not font_files:
            logger.error(f"No font files found in downloaded archive for '{font_name}'")
            return None
        
        # Return the regular weight if available, otherwise first file
        for font_file in font_files:
            if 'regular' in font_file.stem.lower():
                logger.info(f"Font '{font_name}' downloaded successfully: {font_file}")
                return font_file
        
        logger.info(f"Font '{font_name}' downloaded successfully: {font_files[0]}")
        return font_files[0]
        
    except urllib.error.URLError as e:
        logger.error(f"Failed to download font '{font_name}': {e}")
        return None
    except zipfile.BadZipFile:
        logger.error(f"Downloaded file for '{font_name}' is not a valid zip archive")
        if zip_path.exists():
            zip_path.unlink()
        return None
    except Exception as e:
        logger.error(f"Error downloading font '{font_name}': {e}")
        return None


def ensure_font(font_name: str, auto_download: bool = True) -> Optional[Path]:
    """
    Ensure a font is available, downloading if necessary.
    
    Args:
        font_name: Name of the font
        auto_download: If True, download from Google Fonts if not found
        
    Returns:
        Path to the font file, or None if not available
    """
    # First, try to find existing font
    font_path = find_font_file(font_name)
    
    if font_path:
        logger.debug(f"Found font '{font_name}' at: {font_path}")
        return font_path
    
    # If not found and auto-download is enabled, try to download
    if auto_download:
        logger.info(f"Font '{font_name}' not found locally, attempting download...")
        downloaded = download_google_font(font_name)
        if downloaded:
            return downloaded
    
    logger.warning(f"Font '{font_name}' not available")
    return None


def get_font_path_for_config(font_name: str, auto_download: bool = True) -> str:
    """
    Get a font path suitable for use in configuration.
    
    This is the main entry point for getting fonts in the pipeline.
    
    Args:
        font_name: Font name or path
        auto_download: Whether to auto-download from Google Fonts
        
    Returns:
        Path string to the font file, or original name if not found
    """
    font_path = ensure_font(font_name, auto_download=auto_download)
    
    if font_path:
        return str(font_path)
    
    # Return original name as fallback (PIL might find it)
    return font_name
