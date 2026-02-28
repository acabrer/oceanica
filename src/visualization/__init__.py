"""
Sonificació Oceànica - Visualization Module
============================================

Provides echogram visualization tools for MALASPINA acoustic data.

Scripts:
- create_echogram_24h_validation.py: 24h echogram + analysis panels
- create_echogram_video.py: Montage PNG + WAV → MP4 with playhead
- colormaps.py: EK500-standard colormap definitions

Usage:
    from visualization.colormaps import get_colormap
"""

from .colormaps import get_colormap, create_klevjer_colormap, COLORMAPS
