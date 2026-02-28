#!/usr/bin/env python3
"""
================================================================================
SONIFICACIÓ OCEÀNICA - Centralized Configuration
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya, Departament de Cultura

Centralized configuration for paths and scientific parameters.
All scripts should import from this module rather than using hardcoded paths.
================================================================================
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS (relative to this config file)
# =============================================================================

# Config file location determines project root
CONFIG_DIR = Path(__file__).parent  # src/
PROJECT_ROOT = CONFIG_DIR.parent    # oceanica_dev/

# Raw data directories - check multiple possible locations
# Priority: data/raw/ > venv folder (legacy)
DATA_DIR_PRIMARY = PROJECT_ROOT / "data" / "raw"
DATA_DIR_LEGACY = PROJECT_ROOT / "venv_echosound_py311"

def _find_data_dir():
    """Find the data directory, checking primary then legacy locations."""
    if DATA_DIR_PRIMARY.exists():
        return DATA_DIR_PRIMARY
    elif DATA_DIR_LEGACY.exists():
        return DATA_DIR_LEGACY
    else:
        # Return primary as default (for documentation)
        return DATA_DIR_PRIMARY

DATA_DIR = _find_data_dir()
MALASPINA_LEG2 = DATA_DIR / "MALASPINA_LEG2_1"
MALASPINA_LEG7 = DATA_DIR / "MALASPINA_LEG7"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DATA = OUTPUT_DIR / "data"
OUTPUT_VIZ = OUTPUT_DIR / "visualizations"

# SuperCollider data path (for documentation)
SUPERCOLLIDER_DATA = OUTPUT_DATA

# =============================================================================
# SCIENTIFIC PARAMETERS
# =============================================================================

# Environmental parameters (Atlantic/Mediterranean values)
# Reference: MALASPINA 2010 Expedition conditions
ENV_PARAMS = {
    'temperature': 14.0,      # degrees Celsius
    'salinity': 38.5,         # PSU
    'pressure': 10.0          # dbar (approximate for surface)
}

# Depth processing parameters
DEPTH_PARAMS = {
    'surface_exclusion_m': 10.0,      # meters - exclude near-surface noise
    'max_depth_m': 1000.0,            # meters - maximum depth to consider
    'depth_bin_m': 10.0               # meters - vertical binning resolution
}

# Sonification extraction parameters
SONIFICATION_PARAMS = {
    'min_sv_threshold_db': -80.0,     # dB - minimum Sv to consider as signal
    'layer_detection_threshold_db': -70.0,  # dB - threshold for layer detection
    'velocity_smooth_window': 600,    # pings (~30 minutes at ~3s/ping)
    'acceleration_smooth_window': 300  # pings (~15 minutes)
}

# MVBS computation parameters (MacLennan et al. 2002)
# Default preset for backward compatibility
MVBS_PARAMS = {
    'ping_time_bin': '10s',
    'range_bin': '5m'
}

# MVBS presets for different use cases
# Reference: Irigoien et al. (2021) uses 6min × 1m for publication
MVBS_PRESETS = {
    'publication': {
        'ping_time_bin': '6min',   # Matches Nature paper standards
        'range_bin': '1m'          # 1 meter vertical resolution
    },
    'analysis': {
        'ping_time_bin': '1min',   # Medium resolution for analysis
        'range_bin': '2m'
    },
    'sonification': {
        'ping_time_bin': '10s',    # High temporal resolution for audio
        'range_bin': '5m'
    }
}

# Noise estimation parameters (De Robertis & Higginbottom 2007)
NOISE_PARAMS = {
    'ping_num': 30,
    'range_sample_num': 100,
    'SNR_threshold': 3.0  # dB
}

# =============================================================================
# ECHOGRAM VISUALIZATION PRESETS
# =============================================================================
# Reference: Klevjer et al. (2016), Irigoien et al. (2021) Nature papers
# Publication preset matches Figure 1 in srep19873

ECHOGRAM_PRESETS = {
    'publication': {
        'sv_range': (-90, -50),      # dB - matches Nature paper figures
        'depth_range': (0, 1000),    # meters - full mesopelagic zone
        'colormap': 'ek60_sv',       # Custom colormap matching Nature paper (gray→green→yellow→red)
        'figsize': (14, 6),
        'dpi': 300,
        'description': 'Publication-quality echogram matching Nature paper style'
    },
    'analysis': {
        'sv_range': (-85, -40),      # dB - wider range for analysis
        'depth_range': (0, 800),     # meters
        'colormap': 'klevjer',       # Blue-scale (Klevjer et al. style)
        'figsize': (18, 14),
        'dpi': 150,
        'description': 'Multi-panel analysis visualization'
    },
    'preview': {
        'sv_range': (-80, -45),      # dB - focused range
        'depth_range': (0, 700),     # meters
        'colormap': 'viridis',       # Standard scientific colormap
        'figsize': (12, 4),
        'dpi': 100,
        'description': 'Quick preview visualization'
    }
}

# =============================================================================
# DATA CHARACTERISTICS
# =============================================================================

# MALASPINA expedition context
EXPEDITION_INFO = {
    'name': 'MALASPINA 2010',
    'leg': 2,
    'dates': 'January 25-28, 2011',
    'location': 'Atlantic Ocean',
    'instrument': 'Simrad EK60',
    'frequencies_hz': [38000, 120000],
    'note': '120 kHz unusable at depth - seafloor/noise only, use 38 kHz'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT_VIZ.mkdir(parents=True, exist_ok=True)


def get_raw_data_dir(leg: int = 2) -> Path:
    """Get the raw data directory for a specific MALASPINA leg."""
    if leg == 2:
        return MALASPINA_LEG2
    elif leg == 7:
        return MALASPINA_LEG7
    else:
        raise ValueError(f"Unknown MALASPINA leg: {leg}. Available: 2, 7")


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SONIFICACIÓ OCEÀNICA - Configuration Test")
    print("=" * 60)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Config dir:   {CONFIG_DIR}")
    print(f"\nData paths:")
    print(f"  LEG2: {MALASPINA_LEG2}")
    print(f"        exists: {MALASPINA_LEG2.exists()}")
    print(f"  LEG7: {MALASPINA_LEG7}")
    print(f"        exists: {MALASPINA_LEG7.exists()}")
    print(f"\nOutput paths:")
    print(f"  Data: {OUTPUT_DATA}")
    print(f"        exists: {OUTPUT_DATA.exists()}")
    print(f"  Viz:  {OUTPUT_VIZ}")
    print(f"        exists: {OUTPUT_VIZ.exists()}")
    print(f"\nScientific parameters:")
    print(f"  Depth range: {DEPTH_PARAMS['surface_exclusion_m']}-{DEPTH_PARAMS['max_depth_m']}m")
    print(f"  Sv threshold: {SONIFICATION_PARAMS['min_sv_threshold_db']} dB")
    print(f"\nExpedition: {EXPEDITION_INFO['name']} - {EXPEDITION_INFO['dates']}")
    print("=" * 60)
