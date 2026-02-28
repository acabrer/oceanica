#!/usr/bin/env python3
"""
================================================================================
SONIFICACIÓ OCEÀNICA - Colormap Definitions
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)

Centralized colormap definitions for echogram visualization.
Provides both publication-standard and analysis colormaps.

References:
- Klevjer et al. (2016) - Blue-scale colormap for acoustic scattering
- Irigoien et al. (2021) - MALASPINA dataset visualization standards
================================================================================
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def create_klevjer_colormap():
    """
    Create Klevjer et al. (2016) style colormap.

    White to dark blue gradient, commonly used in acoustic scattering
    layer visualization. Weak signals appear as white/light blue,
    strong signals as dark blue.

    Returns
    -------
    LinearSegmentedColormap
        Colormap with 256 colors from white to dark blue.
    """
    colors = [
        (1.0, 1.0, 1.0),      # White (weak signal, -90 dB)
        (0.9, 0.95, 1.0),     # Very light blue
        (0.7, 0.85, 0.95),    # Light blue
        (0.4, 0.7, 0.9),      # Blue
        (0.2, 0.5, 0.8),      # Medium blue
        (0.1, 0.3, 0.6),      # Dark blue
        (0.05, 0.15, 0.4),    # Very dark blue
        (0.0, 0.0, 0.2),      # Near black (strong signal, -50 dB)
    ]
    return LinearSegmentedColormap.from_list('klevjer', colors, N=256)


def create_publication_colormap():
    """
    Create EK500-based publication colormap for fisheries acoustic echograms.

    Derived from the Simrad EK500 standard color table (13 colors, ~3 dB per
    step), simplified for publication use by omitting the pink/magenta and
    brown bands that are not present in the Klevjer et al. (2016) reference.

    Color progression for the -90 to -50 dB display range:
        -90 dB: white (background / no signal)
        -85 dB: light gray
        -80 dB: gray
        -76 dB: blue         (weak biological signal)
        -73 dB: dark blue
        -70 dB: green        (medium signal — DSL)
        -67 dB: dark green
        -63 dB: yellow       (strong signal)
        -59 dB: orange
        -55 dB: red          (very strong signal)
        -50 dB: dark red

    Uses non-uniform position spacing so that the grayscale region occupies
    25% of the colormap range (matching the EK500's allocation of 3 gray
    steps for the weakest signals).

    References
    ----------
    - Simrad EK500 color table (13 colors, Simrad documentation)
    - echopype GitHub issue #78 (EK500 RGB values)
    - Blackwell et al. (2020) ICES J. Mar. Sci. — colour maps for echograms
    - Klevjer et al. (2016) Sci. Rep. 6:19873 — visual reference

    Returns
    -------
    LinearSegmentedColormap
        Colormap with 256 colors matching EK500-derived publication standard.
    """
    # (position, (R, G, B)) — positions map linearly to the Sv display range
    # Position 0.0 = vmin (-90 dB), position 1.0 = vmax (-50 dB)
    colors_with_positions = [
        (0.00, (1.0,   1.0,   1.0)),    # White        @ -90 dB
        (0.12, (0.62,  0.62,  0.62)),   # Light gray   @ -85 dB
        (0.25, (0.37,  0.37,  0.37)),   # Gray         @ -80 dB
        (0.35, (0.0,   0.0,   1.0)),    # Blue         @ -76 dB
        (0.42, (0.0,   0.0,   0.5)),    # Dark blue    @ -73 dB
        (0.50, (0.0,   0.75,  0.0)),    # Green        @ -70 dB
        (0.58, (0.0,   0.5,   0.0)),    # Dark green   @ -67 dB
        (0.68, (1.0,   1.0,   0.0)),    # Yellow       @ -63 dB
        (0.78, (1.0,   0.5,   0.0)),    # Orange       @ -59 dB
        (0.88, (1.0,   0.0,   0.0)),    # Red          @ -55 dB
        (1.00, (0.5,   0.0,   0.0)),    # Dark red     @ -50 dB
    ]
    return LinearSegmentedColormap.from_list(
        'ek60_sv',
        [(pos, col) for pos, col in colors_with_positions],
        N=256
    )


# Dictionary of available colormaps
# Use these names in ECHOGRAM_PRESETS['colormap']
COLORMAPS = {
    'klevjer': create_klevjer_colormap(),
    'ek60_sv': create_publication_colormap(),  # Nature paper style (recommended)
    'publication': create_publication_colormap(),  # Alias for ek60_sv
    # Built-in matplotlib alternatives
    'YlGnBu_r': 'YlGnBu_r',      # Reversed Yellow-Green-Blue
    'viridis': 'viridis',        # Standard perceptually uniform
    'plasma': 'plasma',          # Warm alternative
    'turbo': 'turbo',            # Rainbow-like, good for Sv
}


def get_colormap(name: str):
    """
    Get a colormap by name.

    Parameters
    ----------
    name : str
        Name of the colormap. Can be:
        - 'klevjer': Blue-scale (Klevjer et al. style)
        - 'publication': Custom Nature paper style
        - 'YlGnBu_r': Matplotlib built-in (recommended for publication)
        - 'viridis': Matplotlib built-in
        - Any other matplotlib colormap name

    Returns
    -------
    colormap
        Matplotlib colormap object or string name for built-in.
    """
    if name in COLORMAPS:
        return COLORMAPS[name]
    else:
        # Assume it's a matplotlib built-in name
        return name


def preview_colormaps(sv_range=(-90, -50)):
    """
    Preview all available colormaps.

    Creates a figure showing all colormaps with a simulated Sv gradient.

    Parameters
    ----------
    sv_range : tuple
        (min, max) Sv values in dB for the preview.
    """
    fig, axes = plt.subplots(len(COLORMAPS), 1, figsize=(12, 2*len(COLORMAPS)))

    # Create gradient data
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.repeat(gradient, 30, axis=0)

    for ax, (name, cmap) in zip(axes, COLORMAPS.items()):
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=12)
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels([f'{sv_range[0]} dB', f'{(sv_range[0]+sv_range[1])/2:.0f} dB', f'{sv_range[1]} dB'])
        ax.set_yticks([])

    axes[0].set_title('Echogram Colormap Preview (Sv range)', fontsize=14)
    plt.tight_layout()
    plt.savefig('colormap_preview.png', dpi=150, bbox_inches='tight')
    print("Saved: colormap_preview.png")
    plt.close()


if __name__ == "__main__":
    print("Available colormaps:")
    for name, cmap in COLORMAPS.items():
        cmap_type = "custom" if isinstance(cmap, LinearSegmentedColormap) else "matplotlib"
        print(f"  - {name} ({cmap_type})")

    print("\nGenerating preview...")
    preview_colormaps()
