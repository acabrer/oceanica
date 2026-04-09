"""
Sonification extraction configuration.

Contains the SonificationConfigV3 dataclass with all extraction parameters
(depth ranges, smoothing windows, DVM corridor bounds, etc.).
"""

from dataclasses import dataclass


@dataclass
class SonificationConfigV3:
    """Configuration for v3+ extraction."""
    surface_exclusion_m: float = 10.0
    max_depth_m: float = 1000.0
    min_sv_threshold_db: float = -80.0
    layer_detection_threshold_db: float = -70.0
    layer_prominence_db: float = 5.0    # minimum peak prominence for find_peaks
    layer_width_bins: int = 2           # minimum peak width in depth bins
    depth_bin_m: float = 10.0
    histogram_bands: int = 32           # v5: number of depth histogram bands
    # Smoothing windows (in number of pings, ~3s each)
    velocity_smooth_window: int = 600       # ~30 minutes
    velocity_stable_smooth_window: int = 100  # ~5 minutes (captures rapid DVM events)
    acceleration_smooth_window: int = 300   # ~15 minutes
    # --- v6: Sonification mode (shorter smoothing, event features) ---
    sonification_mode: bool = False
    son_depth_smooth_window: int = 60       # ~3 min (was 600)
    son_velocity_smooth_window: int = 30    # ~1.5 min (was 300)
    son_acceleration_smooth_window: int = 60  # ~3 min (was 300)
    son_onset_smooth_window: int = 10       # ~30 sec (was 60)
    son_spread_change_window: int = 30      # ~1.5 min (was 100)
    son_autocorr_smooth_window: int = 0     # none (was 30)
    son_histogram_bands: int = 8            # coarser for sonification (was 32)
    max_tracked_layers: int = 4             # persistent layer tracking slots
    layer_match_max_distance_m: float = 50.0  # max depth shift to match same layer
    cusum_threshold: float = 12.0           # CUSUM alarm threshold (std units)
    cusum_drift: float = 1.5               # CUSUM drift parameter
    # --- v7: Improved histogram zones ---
    son_diff_short_window: int = 30         # ~1.5 min (feeding events, fast oscillations)
    son_diff_medium_window: int = 300       # ~15 min (DVM structural movement)
    zone_edges: tuple = (10, 50, 150, 300, 500, 700, 850, 1000)  # oceanographic boundaries (m)
    dvm_corridor: tuple = (280, 580)        # Zone 7: DVM transit corridor (m)
    # --- v8: DVM depth tracking (dual-zone log-ratio) ---
    dvm_shallow_narrow: tuple = (150, 250)  # Core night DVM band (m)
    dvm_deep_narrow: tuple = (450, 550)     # Core day DVM band (m)
    dvm_shallow_wide: tuple = (100, 300)    # Wider night corridor (m)
    dvm_deep_wide: tuple = (400, 600)       # Wider day corridor (m)
    dvm_blend_weight: float = 0.5           # Weight for narrow zones (0=wide-only, 1=narrow-only)
    dvm_depth_shallow: float = 120.0        # Night surface feeding depth (mapping floor)
    dvm_depth_deep: float = 620.0           # Day deep refuge depth (mapping ceiling)
    dvm_norm_percentile: tuple = (2, 98)    # Percentile range for ratio->depth mapping
    son_dvm_avg_window: int = 301            # ~15 min sliding average (sonification)
    son_dvm_median_window: int = 201        # ~10 min median filter (sonification; must be odd)
    son_dvm_smooth_window: int = 101        # ~5 min uniform filter (sonification)
    dvm_avg_window: int = 601               # ~30 min sliding average (analysis mode)
    dvm_median_window: int = 601            # ~30 min median filter (analysis; must be odd)
    dvm_smooth_window: int = 301            # ~15 min uniform filter (analysis mode)
