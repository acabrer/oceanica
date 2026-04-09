#!/usr/bin/env python3
"""
================================================================================
SONIFICATION FEATURE EXTRACTOR
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya, Departament de Cultura

Reads cleaned 24h NetCDF data (output of run_24h_processing.py) and extracts
sonification features for SuperCollider.

Produces:
  sonification_sc_v5_{date}.json — prenormalized arrays (0.0-1.0) for each
                                    ping (~3s resolution), ready for SC import.

Features extracted per ping:
  - center_of_mass_m: biomass-weighted mean depth (whole-column average)
  - dominant_peak_depth_m: depth of strongest scattering layer (tracks visible DVM)
  - total_intensity_db: integrated acoustic backscatter (biomass proxy; linear mean)
  - peak_depth_m: depth of strongest return (raw argmax)
  - vertical_spread_m: standard deviation of depth distribution
  - acoustic_entropy: Shannon entropy of depth distribution (0=tonal, 1=noise-like)
  - layer_count: number of distinct scattering layers (scipy.signal.find_peaks)
  - layer_separation_m: depth gap between dominant peaks when layer_count >= 2
  - surface/midwater/deep_intensity_db: depth-stratified backscatter (linear mean)
  - inter_ping_correlation: cosine similarity with previous ping (texture continuity)
  - velocity_m_h: migration speed (derived, 30-min smoothed)
  - acceleration_m_h2: rate of velocity change (derived)
  - depth_anomaly_m: deviation from expected DVM depth (South Atlantic Jan 2011 model)
  - spread_change_m_min: rate of spread change (derived)
  - onset_strength: smoothed abs first-difference of total intensity (percussive events)
  v5 additions:
  - depth_histogram: 32-band depth energy profile (normalized probability distribution)
  - dist_skewness: weighted skewness of depth distribution (asymmetry)
  - dist_kurtosis: weighted excess kurtosis of depth distribution (peakedness)
  - vertical_gradient_db_m: mean |dSv/dz| (boundary sharpness)
  - peak_max_prominence_db: prominence of strongest peak (layer distinctness)
  - peak_mean_width_m: mean width of detected peaks (layer thickness)
  - peak_max_height_db: absolute height of strongest peak

Usage:
    python src/extraction/sonification_extractor.py [YYYYMMDD]
    # default date: 20110126

Author: Oceànica (Alex Cabrer & Joan Sala)
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DATA, ensure_dirs

import numpy as np
import xarray as xr
import json
from datetime import datetime
from dataclasses import dataclass
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SonificationConfigV3:
    """Configuration for v3 extraction."""
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
    # Two zone pairs: narrow (core DVM bands) + wide (stability).
    # The log-ratio of shallow/deep energy captures the DVM phase, which is
    # then mapped to a physical depth range via percentile normalization.
    dvm_shallow_narrow: tuple = (150, 250)  # Core night DVM band (m)
    dvm_deep_narrow: tuple = (450, 550)     # Core day DVM band (m)
    dvm_shallow_wide: tuple = (100, 300)    # Wider night corridor (m)
    dvm_deep_wide: tuple = (400, 600)       # Wider day corridor (m)
    dvm_blend_weight: float = 0.5           # Weight for narrow zones (0=wide-only, 1=narrow-only)
    dvm_depth_shallow: float = 120.0        # Night surface feeding depth (mapping floor; widened from 180)
    dvm_depth_deep: float = 620.0           # Day deep refuge depth (mapping ceiling; widened from 550)
    dvm_norm_percentile: tuple = (2, 98)    # Percentile range for ratio→depth mapping (wider preserves intra-regime variation)
    son_dvm_avg_window: int = 301            # ~15 min sliding average (sonification; reduced from 601)
    son_dvm_median_window: int = 201        # ~10 min median filter (sonification; must be odd; reduced from 601)
    son_dvm_smooth_window: int = 101        # ~5 min uniform filter (sonification; reduced from 301)
    dvm_avg_window: int = 601               # ~30 min sliding average (analysis mode)
    dvm_median_window: int = 601            # ~30 min median filter (analysis; must be odd)
    dvm_smooth_window: int = 301            # ~15 min uniform filter (analysis mode)


def extract_ping_features(sv_column, depth_values, config):
    """Extract features from a single ping."""
    valid_mask = (
        (depth_values >= config.surface_exclusion_m) &
        (depth_values <= config.max_depth_m) &
        (~np.isnan(sv_column)) &
        (sv_column > config.min_sv_threshold_db)
    )

    if not np.any(valid_mask):
        return {
            'center_of_mass_m': np.nan,
            'total_intensity_db': -90.0,
            'peak_depth_m': np.nan,
            'dominant_peak_depth_m': np.nan,
            'vertical_spread_m': 0.0,
            'acoustic_entropy': 0.0,
            'layer_count': 0,
            'layer_separation_m': 0.0,
            'surface_intensity_db': -90.0,
            'midwater_intensity_db': -90.0,
            'deep_intensity_db': -90.0,
            # v5 features
            'depth_histogram': [0.0] * config.histogram_bands,
            'dist_skewness': 0.0,
            'dist_kurtosis': 0.0,
            'vertical_gradient_db_m': 0.0,
            'peak_max_prominence_db': 0.0,
            'peak_mean_width_m': 0.0,
            'peak_max_height_db': -90.0,
        }

    valid_sv = sv_column[valid_mask]
    valid_depth = depth_values[valid_mask]
    sv_linear = 10 ** (valid_sv / 10)
    total_linear = np.sum(sv_linear)

    if total_linear > 0:
        center_of_mass = np.sum(valid_depth * sv_linear) / total_linear
        total_intensity = 10 * np.log10(total_linear)
        variance = np.sum(sv_linear * (valid_depth - center_of_mass) ** 2) / total_linear
        vertical_spread = np.sqrt(variance)
        # Acoustic entropy: Shannon entropy of normalized depth distribution
        p = sv_linear / total_linear
        p = p[p > 0]
        entropy_bits = float(-np.sum(p * np.log2(p)))
        max_entropy = np.log2(len(sv_linear)) if len(sv_linear) > 1 else 1.0
        entropy_norm = float(entropy_bits / max_entropy)  # 0=tonal, 1=noise-like
    else:
        center_of_mass = np.nan
        total_intensity = -90.0
        vertical_spread = 0.0
        entropy_norm = 0.0

    peak_idx = np.argmax(valid_sv)
    peak_depth = valid_depth[peak_idx]

    # Layer detection (returns tuple with peak properties)
    layer_count, layer_peaks, layer_separation, peak_props = detect_layers(sv_column, depth_values, config)

    # Dominant peak depth: depth of the most prominent scattering layer.
    # Uses find_peaks prominence (how much a peak stands out from its surroundings)
    # rather than raw height, because the persistent deep layer can have high Sv
    # but low prominence. This preferentially tracks the migrating DVM layer.
    # Falls back to raw argmax for pings where no layer meets detection criteria.
    if peak_props['prominences_db']:
        dominant_idx = int(np.argmax(peak_props['prominences_db']))
        dominant_peak_depth = layer_peaks[dominant_idx]
    else:
        dominant_peak_depth = peak_depth

    # Depth-stratified intensities
    surface_intensity = compute_layer_intensity(sv_column, depth_values, 10, 150, config.min_sv_threshold_db)
    midwater_intensity = compute_layer_intensity(sv_column, depth_values, 150, 400, config.min_sv_threshold_db)
    deep_intensity = compute_layer_intensity(sv_column, depth_values, 400, 800, config.min_sv_threshold_db)

    # --- v5: Depth histogram (32 bands, log-spaced) ---
    n_bands = config.histogram_bands
    hist_edges = np.logspace(
        np.log10(max(config.surface_exclusion_m, 10.0)),
        np.log10(config.max_depth_m),
        n_bands + 1
    )
    hist_values = np.zeros(n_bands)
    for b in range(n_bands):
        band_mask = (
            (depth_values >= hist_edges[b]) &
            (depth_values < hist_edges[b + 1]) &
            (~np.isnan(sv_column)) &
            (sv_column > config.min_sv_threshold_db)
        )
        if np.any(band_mask):
            band_linear = 10 ** (sv_column[band_mask] / 10)
            hist_values[b] = float(np.mean(band_linear))

    hist_sum = hist_values.sum()
    hist_norm = (hist_values / hist_sum) if hist_sum > 0 else np.zeros(n_bands)

    # --- v5: Distribution shape (weighted skewness and kurtosis) ---
    if total_linear > 0 and len(sv_linear) > 3 and vertical_spread > 0:
        z = (valid_depth - center_of_mass) / vertical_spread
        dist_skewness = float(np.sum(sv_linear * z**3) / total_linear)
        dist_kurtosis = float(np.sum(sv_linear * z**4) / total_linear - 3.0)
    else:
        dist_skewness = 0.0
        dist_kurtosis = 0.0

    # --- v5: Vertical gradient (mean |dSv/dz|) ---
    # Use the full column within valid depth range for gradient
    range_mask = (
        (depth_values >= config.surface_exclusion_m) &
        (depth_values <= config.max_depth_m)
    )
    range_sv = sv_column[range_mask]
    range_depth = depth_values[range_mask]
    if len(range_sv) > 1:
        d_diff = np.diff(range_depth)
        d_diff[d_diff == 0] = 1e-6
        sv_grad = np.abs(np.diff(np.nan_to_num(range_sv, nan=-90.0)) / d_diff)
        valid_grad = sv_grad[np.isfinite(sv_grad)]
        vertical_gradient = float(np.mean(valid_grad)) if len(valid_grad) > 0 else 0.0
    else:
        vertical_gradient = 0.0

    # --- v5: Peak property scalars ---
    max_prominence = float(max(peak_props['prominences_db'])) if peak_props['prominences_db'] else 0.0
    mean_peak_width = float(np.mean(peak_props['widths_m'])) if peak_props['widths_m'] else 0.0
    max_peak_height = float(max(peak_props['heights_db'])) if peak_props['heights_db'] else -90.0

    return {
        'center_of_mass_m': center_of_mass,
        'total_intensity_db': total_intensity,
        'peak_depth_m': peak_depth,
        'dominant_peak_depth_m': dominant_peak_depth,
        'vertical_spread_m': vertical_spread,
        'acoustic_entropy': entropy_norm,
        'layer_count': layer_count,
        'layer_separation_m': layer_separation,
        'surface_intensity_db': surface_intensity,
        'midwater_intensity_db': midwater_intensity,
        'deep_intensity_db': deep_intensity,
        # v5 features
        'depth_histogram': hist_norm.tolist(),
        'dist_skewness': dist_skewness,
        'dist_kurtosis': dist_kurtosis,
        'vertical_gradient_db_m': vertical_gradient,
        'peak_max_prominence_db': max_prominence,
        'peak_mean_width_m': mean_peak_width,
        'peak_max_height_db': max_peak_height,
        'peak_depths': list(layer_peaks),  # variable-length list for layer tracking
    }


def compute_layer_intensity(sv_column, depth_values, min_depth, max_depth, threshold_db):
    """Compute mean intensity within a depth layer (correct linear averaging)."""
    mask = (
        (depth_values >= min_depth) &
        (depth_values <= max_depth) &
        (~np.isnan(sv_column)) &
        (sv_column > threshold_db)
    )
    if np.any(mask):
        linear = 10 ** (sv_column[mask] / 10)
        return float(10 * np.log10(np.mean(linear)))
    return -90.0


def detect_layers(sv_column, depth_values, config):
    """
    Detect distinct scattering layers using find_peaks.

    Returns
    -------
    (layer_count, peak_depths_m, layer_separation_m, peak_properties)
        layer_count       : number of distinct peaks
        peak_depths_m     : list of peak centre depths (m)
        layer_separation_m: depth gap between shallowest and deepest peak (0 if <2 peaks)
        peak_properties   : dict with heights_db, prominences_db, widths_m lists
    """
    depth_bins = np.arange(config.surface_exclusion_m, config.max_depth_m, config.depth_bin_m)
    binned_sv = np.full(len(depth_bins) - 1, -90.0)

    for i in range(len(depth_bins) - 1):
        mask = (depth_values >= depth_bins[i]) & (depth_values < depth_bins[i+1])
        if np.any(mask) and np.any(~np.isnan(sv_column[mask])):
            binned_sv[i] = np.nanmean(sv_column[mask])

    peaks, properties = find_peaks(
        binned_sv,
        height=config.layer_detection_threshold_db,
        prominence=config.layer_prominence_db,
        width=config.layer_width_bins
    )

    count = len(peaks)
    peak_depths_m = (depth_bins[peaks] + config.depth_bin_m / 2).tolist() if count > 0 else []
    separation_m = float(peak_depths_m[-1] - peak_depths_m[0]) if count >= 2 else 0.0

    if count > 0:
        peak_props = {
            'heights_db': properties['peak_heights'].tolist(),
            'prominences_db': properties['prominences'].tolist(),
            'widths_m': (properties['widths'] * config.depth_bin_m).tolist(),
        }
    else:
        peak_props = {
            'heights_db': [],
            'prominences_db': [],
            'widths_m': [],
        }

    return count, peak_depths_m, separation_m, peak_props


def _sigmoid(x, k=8.0):
    """Logistic sigmoid mapped to [0, 1] for smooth DVM transitions."""
    return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))


def compute_expected_depth_for_time(hour_of_day):
    """
    Model of expected DVM depth calibrated for South Atlantic (~25°S), January 2011.

    Sunrise ≈ 05:45 UTC, sunset ≈ 20:15 UTC.
    Night depth (00:00–05:45): ~280 m  (mesopelagic community in South Atlantic)
    Dawn descent (05:45–08:30): sigmoid 280 → 580 m over ~2.75 h (slow)
    Day depth (08:30–19:30): ~580 m
    Dusk ascent (19:30–20:15): sigmoid 580 → 280 m over ~0.75 h (fast)
    Early night (20:15–24:00): ~280 m
    """
    NIGHT_DEPTH = 280.0
    DAY_DEPTH   = 580.0
    SUNRISE     = 5.75   # 05:45 UTC
    DAWN_END    = 8.5    # 08:30 UTC
    DUSK_START  = 19.5   # 19:30 UTC
    SUNSET      = 20.25  # 20:15 UTC

    hour = hour_of_day % 24

    if hour < SUNRISE:
        return NIGHT_DEPTH
    elif hour < DAWN_END:
        progress = (hour - SUNRISE) / (DAWN_END - SUNRISE)
        return NIGHT_DEPTH + _sigmoid(progress, k=6) * (DAY_DEPTH - NIGHT_DEPTH)
    elif hour < DUSK_START:
        return DAY_DEPTH
    elif hour < SUNSET:
        progress = (hour - DUSK_START) / (SUNSET - DUSK_START)
        return DAY_DEPTH - _sigmoid(progress, k=10) * (DAY_DEPTH - NIGHT_DEPTH)
    else:
        return NIGHT_DEPTH


def add_derived_features(features, config, sonification_mode=False):
    """Add velocity, acceleration, and anomaly features after collecting all data.

    When sonification_mode=True, uses shorter smoothing windows from config.son_*
    fields to preserve temporal dynamics for sonification.
    """
    if len(features) < 10:
        return features

    # Select smoothing windows based on mode
    if sonification_mode:
        depth_window = config.son_depth_smooth_window        # 60 (~3 min)
        vel_window = config.son_velocity_smooth_window        # 30 (~1.5 min)
        accel_window = config.son_acceleration_smooth_window   # 60 (~3 min)
        onset_window = config.son_onset_smooth_window          # 10 (~30 sec)
        spread_window = config.son_spread_change_window        # 30 (~1.5 min)
    else:
        depth_window = config.velocity_smooth_window           # 600 (~30 min)
        vel_window = config.velocity_smooth_window // 2        # 300 (~15 min)
        accel_window = config.acceleration_smooth_window       # 300 (~15 min)
        onset_window = 60                                      # ~3 min
        spread_window = 100                                    # ~5 min

    # Extract arrays
    times = np.array([f['timestamp_seconds'] for f in features])
    depths = np.array([f['center_of_mass_m'] for f in features])
    hours = np.array([f['hour_of_day'] for f in features])

    # Handle NaN in depths
    depths_clean = np.copy(depths)
    nan_mask = np.isnan(depths_clean)
    if np.any(~nan_mask):
        depths_clean[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            depths_clean[~nan_mask]
        )
    # Clamp to physical range to prevent interpolation artefacts causing phantom velocity
    depths_clean = np.clip(depths_clean, 50.0, config.max_depth_m)

    # Smooth depth for velocity calculation
    depth_smooth = uniform_filter1d(depths_clean, depth_window, mode='nearest')

    # Calculate velocity (m/hour) - positive = ascending (toward surface)
    dt = np.gradient(times)
    dt[dt == 0] = 1  # Avoid division by zero
    velocity = -np.gradient(depth_smooth, times) * 3600  # Negative because increasing depth = descending

    # Smooth velocity
    velocity_smooth = uniform_filter1d(velocity, vel_window, mode='nearest')

    # Stable-window velocity (5-min window for capturing rapid DVM events)
    depth_smooth_stable = uniform_filter1d(depths_clean, config.velocity_stable_smooth_window, mode='nearest')
    velocity_stable = -np.gradient(depth_smooth_stable, times) * 3600
    velocity_stable_smooth = uniform_filter1d(velocity_stable, config.velocity_stable_smooth_window // 2, mode='nearest')

    # Calculate acceleration (m/h²)
    acceleration = np.gradient(velocity_smooth, times) * 3600
    acceleration_smooth = uniform_filter1d(acceleration, accel_window, mode='nearest')

    # Calculate depth anomaly (deviation from expected for time of day)
    expected_depths = np.array([compute_expected_depth_for_time(h) for h in hours])
    depth_anomaly = depths_clean - expected_depths  # Positive = deeper than expected

    # Rate of change of spread (layer dynamics)
    spreads = np.array([f['vertical_spread_m'] for f in features])
    spreads_clean = np.nan_to_num(spreads, nan=200)
    spread_change = np.gradient(uniform_filter1d(spreads_clean, spread_window, mode='nearest'), times) * 60  # m/min

    # Onset strength: smoothed abs first-difference of total intensity (in linear space)
    intensity_arr = np.array([f['total_intensity_db'] for f in features])
    intensity_linear = 10 ** (intensity_arr / 10)
    onset_raw = np.abs(np.gradient(intensity_linear, times))
    onset_smooth = uniform_filter1d(onset_raw, size=onset_window, mode='nearest')

    # Outlier detection: z-scores for intensity, depth, spread
    intensity_z = (intensity_arr - np.nanmean(intensity_arr)) / (np.nanstd(intensity_arr) + 1e-10)
    depth_z = (depths_clean - np.nanmean(depths_clean)) / (np.nanstd(depths_clean) + 1e-10)
    spread_z = (spreads_clean - np.nanmean(spreads_clean)) / (np.nanstd(spreads_clean) + 1e-10)
    outlier_score = np.maximum(np.abs(intensity_z), np.maximum(np.abs(depth_z), np.abs(spread_z)))

    # ===================================================================
    # DVM depth: track the dominant scattering layer (not center-of-mass)
    # ===================================================================
    # Uses the most PROMINENT peak per ping from find_peaks (detect_layers).
    # Prominence measures how much a peak stands out from its surroundings,
    # which preferentially selects the migrating DVM layer over the persistent
    # deep scattering layer (which has high absolute Sv but lower prominence).
    #
    # Smoothing: large median filter (preserves sharp dawn/dusk transitions
    # while rejecting ping-level noise and brief layer-switching artifacts),
    # followed by gentle uniform filter for musical continuity.
    # The DVM transitions are 45 min (dusk) to 2.5 h (dawn), so a 15-minute
    # median window is safe — it cannot smear the transitions.
    dvm_raw = np.array([f['dominant_peak_depth_m'] for f in features])

    # Select DVM smoothing windows based on mode
    if sonification_mode:
        dvm_med_window = config.son_dvm_median_window    # ~5 min median
        dvm_uni_window = config.son_dvm_smooth_window    # ~1.5 min uniform
    else:
        dvm_med_window = config.dvm_median_window        # ~10 min median
        dvm_uni_window = config.dvm_smooth_window        # ~5 min uniform

    # Interpolate NaN pings
    dvm_clean = np.copy(dvm_raw)
    dvm_nan = np.isnan(dvm_clean)
    if np.any(~dvm_nan):
        dvm_clean[dvm_nan] = np.interp(
            np.flatnonzero(dvm_nan),
            np.flatnonzero(~dvm_nan),
            dvm_clean[~dvm_nan]
        )
    dvm_clean = np.clip(dvm_clean, 10.0, config.max_depth_m)

    # Two-stage smoothing: median (robust) then uniform (gentle)
    dvm_median = median_filter(dvm_clean, size=dvm_med_window, mode='nearest')
    dvm_smooth = uniform_filter1d(dvm_median, dvm_uni_window, mode='nearest')

    # DVM-specific velocity (m/hour, positive = ascending toward surface)
    dvm_velocity = -np.gradient(dvm_smooth, times) * 3600
    dvm_velocity_smooth = uniform_filter1d(dvm_velocity, vel_window, mode='nearest')

    # Add to features
    for i, f in enumerate(features):
        f['velocity_m_h'] = float(velocity_smooth[i])
        f['velocity_stable_m_h'] = float(velocity_stable_smooth[i])
        f['acceleration_m_h2'] = float(acceleration_smooth[i])
        f['depth_anomaly_m'] = float(depth_anomaly[i])
        f['spread_change_m_min'] = float(spread_change[i])
        f['depth_smooth_m'] = float(depth_smooth[i])
        f['onset_strength'] = float(onset_smooth[i])
        f['onset_peak'] = float(onset_raw[i])  # v7: unsmoothed onset for trigger detection
        f['outlier_score'] = float(outlier_score[i])
        f['dvm_depth_smooth_m'] = float(dvm_smooth[i])
        f['dvm_velocity_m_h'] = float(dvm_velocity_smooth[i])

    return features


def compute_lagged_autocorrelation(sv_data, time_seconds):
    """Compute autocorrelation at 3-min and 10-min lags using cosine similarity.

    Reveals cyclic patterns (feeding pulses, oscillatory migrations) that
    single-ping IPC misses. Low autocorrelation = turbulent/changing,
    high autocorrelation = periodic/stable pattern.
    """
    n_pings = sv_data.shape[0]
    median_dt = float(np.median(np.diff(time_seconds)))

    lag_3min = max(int(3 * 60 / median_dt), 1)    # ~60 pings
    lag_10min = max(int(10 * 60 / median_dt), 1)   # ~200 pings

    autocorr_3min = np.zeros(n_pings)
    autocorr_10min = np.zeros(n_pings)

    for i in range(n_pings):
        profile_i = np.nan_to_num(sv_data[i, :], nan=-90.0)
        norm_i = np.linalg.norm(profile_i)

        if i >= lag_3min and norm_i > 0:
            profile_lag = np.nan_to_num(sv_data[i - lag_3min, :], nan=-90.0)
            norm_lag = np.linalg.norm(profile_lag)
            if norm_lag > 0:
                autocorr_3min[i] = float(np.clip(
                    np.dot(profile_i, profile_lag) / (norm_i * norm_lag + 1e-10), 0, 1))

        if i >= lag_10min and norm_i > 0:
            profile_lag = np.nan_to_num(sv_data[i - lag_10min, :], nan=-90.0)
            norm_lag = np.linalg.norm(profile_lag)
            if norm_lag > 0:
                autocorr_10min[i] = float(np.clip(
                    np.dot(profile_i, profile_lag) / (norm_i * norm_lag + 1e-10), 0, 1))

    return autocorr_3min, autocorr_10min


def compute_histogram_differential(features, config):
    """Compute 8-band oceanographic histogram and 8-band differential from 32-band data.

    Static histogram: 8 oceanographic zones computed fresh from per-ping depth_histogram.
    Differential: group-of-4 from existing 32-band, first-difference along time.

    Returns
    -------
    hist_8band : np.ndarray, shape (n_pings, 8)
        Static 8-band oceanographic histogram, per-band normalized [0, 1].
    hist_diff : np.ndarray, shape (n_pings, 8)
        Per-band rate of change (positive = gaining energy).
    """
    n_pings = len(features)

    # --- Static 8-band: oceanographic zones ---
    # Zone edges in meters
    zone_edges = [10, 50, 150, 300, 500, 700, 850, 1000]
    n_zones = len(zone_edges)  # 7 zones + 1 broadband = 8
    hist_8band = np.zeros((n_pings, 8))

    # Extract the 32-band histograms and their edges
    n_32 = config.histogram_bands  # 32
    hist_edges_32 = np.logspace(
        np.log10(max(config.surface_exclusion_m, 10.0)),
        np.log10(config.max_depth_m),
        n_32 + 1
    )
    hist_32 = np.array([f['depth_histogram'] for f in features])  # (n_pings, 32)

    # Map each 32-band to its oceanographic zone
    band_centers_32 = (hist_edges_32[:-1] + hist_edges_32[1:]) / 2
    for b in range(n_32):
        center = band_centers_32[b]
        # Find which zone this band center falls into
        zone_idx = 0
        for z in range(len(zone_edges) - 1):
            if center >= zone_edges[z]:
                zone_idx = z
        hist_8band[:, zone_idx] += hist_32[:, b]

    # Zone 7 = broadband summary (total energy)
    hist_8band[:, 7] = hist_32.sum(axis=1)

    # Per-band normalization (2nd-98th percentile)
    for b in range(8):
        band = hist_8band[:, b]
        valid = band[band > 0]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                hist_8band[:, b] = np.clip((band - p2) / (p98 - p2), 0, 1)

    # --- Differential: group-of-4 from 32-band ---
    diff_8band_raw = np.zeros((n_pings, 8))
    for g in range(8):
        start = g * 4
        end = min(start + 4, n_32)
        group_sum = hist_32[:, start:end].sum(axis=1)
        diff_8band_raw[:, g] = np.diff(group_sum, prepend=group_sum[0])

    # Minimal 3-ping smoothing per band
    hist_diff = np.zeros_like(diff_8band_raw)
    for b in range(8):
        hist_diff[:, b] = uniform_filter1d(diff_8band_raw[:, b], size=3, mode='nearest')

    return hist_8band, hist_diff


def compute_histogram_zones(sv_data, depth_values, config):
    """Compute 8-band oceanographic histogram directly from raw Sv data.

    v7: Computes zone energy directly from raw Sv profiles (bypasses 32-band
    intermediate), eliminates log-to-linear summation bias, replaces Zone 7
    (broadband) with DVM corridor, and provides both local and global
    normalization plus multi-scale differentials.

    Parameters
    ----------
    sv_data : np.ndarray, shape (n_pings, n_depth)
        Raw Sv values in dB.
    depth_values : np.ndarray, shape (n_depth,)
        Depth array in meters.
    config : SonificationConfigV3
        Extraction configuration.

    Returns
    -------
    hist_local : np.ndarray, shape (n_pings, 8)
        Per-band 2-98th percentile normalized [0, 1]. For SC filter bank.
    hist_global : np.ndarray, shape (n_pings, 8)
        Globally normalized [0, 1] across all bands. Preserves inter-zone ratios.
    diff_short : np.ndarray, shape (n_pings, 8)
        Short-term differential (~1.5 min), normalized [0, 1] centered at 0.5.
    diff_medium : np.ndarray, shape (n_pings, 8)
        Medium-term differential (~15 min), normalized [0, 1] centered at 0.5.
    """
    n_pings = sv_data.shape[0]
    zone_edges = list(config.zone_edges)
    dvm_lo, dvm_hi = config.dvm_corridor
    threshold = config.min_sv_threshold_db

    # 8 zones: 7 oceanographic + Zone 7 = DVM corridor (280-580m)
    # Zone boundaries: [10-50, 50-150, 150-300, 300-500, 500-700, 700-850, 850-1000, 280-580]
    zone_ranges = []
    for z in range(len(zone_edges) - 1):
        zone_ranges.append((zone_edges[z], zone_edges[z + 1]))
    zone_ranges.append((dvm_lo, dvm_hi))  # Zone 7 = DVM corridor

    # Pre-compute depth masks for each zone (fixed across pings)
    zone_masks = []
    for lo, hi in zone_ranges:
        mask = (depth_values >= lo) & (depth_values < hi) & np.isfinite(depth_values)
        zone_masks.append(mask)

    # Compute mean linear backscatter per zone per ping
    hist_raw = np.zeros((n_pings, 8))
    for i in range(n_pings):
        sv_ping = sv_data[i, :]
        for z in range(8):
            zone_sv = sv_ping[zone_masks[z]]
            valid = zone_sv[~np.isnan(zone_sv) & (zone_sv > threshold)]
            if len(valid) > 0:
                hist_raw[i, z] = float(np.mean(10 ** (valid / 10)))

    # --- Local normalization: per-band 2-98th percentile ---
    hist_local = np.zeros_like(hist_raw)
    local_ranges = []
    for b in range(8):
        band = hist_raw[:, b]
        valid = band[band > 0]
        if len(valid) > 10:
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                hist_local[:, b] = np.clip((band - p2) / (p98 - p2), 0, 1)
                local_ranges.append((float(p2), float(p98)))
            else:
                local_ranges.append((0.0, 1.0))
        else:
            local_ranges.append((0.0, 1.0))

    # --- Global normalization: single range across all bands ---
    all_valid = hist_raw[hist_raw > 0]
    if len(all_valid) > 10:
        g_p2, g_p98 = np.percentile(all_valid, [2, 98])
        if g_p98 > g_p2:
            hist_global = np.clip((hist_raw - g_p2) / (g_p98 - g_p2), 0, 1)
        else:
            hist_global = np.zeros_like(hist_raw)
    else:
        hist_global = np.zeros_like(hist_raw)

    # --- Differentials from pre-normalization zone time series ---
    # Short-term (~1.5 min): captures feeding events, fast oscillations
    diff_short_raw = np.zeros_like(hist_raw)
    for b in range(8):
        smoothed = uniform_filter1d(hist_raw[:, b], size=config.son_diff_short_window, mode='nearest')
        diff_short_raw[:, b] = np.gradient(smoothed)

    # Medium-term (~15 min): captures DVM structural movement
    diff_medium_raw = np.zeros_like(hist_raw)
    for b in range(8):
        smoothed = uniform_filter1d(hist_raw[:, b], size=config.son_diff_medium_window, mode='nearest')
        diff_medium_raw[:, b] = np.gradient(smoothed)

    # Normalize differentials symmetrically around 0.5
    def normalize_diff(diff_arr):
        result = np.zeros_like(diff_arr)
        ranges = []
        for b in range(diff_arr.shape[1]):
            band = diff_arr[:, b]
            abs_max = max(float(np.percentile(np.abs(band), 98)), 1e-10)
            result[:, b] = np.clip((band / abs_max + 1.0) / 2.0, 0, 1)
            ranges.append(float(abs_max))
        return result, ranges

    diff_short, diff_short_ranges = normalize_diff(diff_short_raw)
    diff_medium, diff_medium_ranges = normalize_diff(diff_medium_raw)

    # Print zone statistics
    zone_labels = ['10-50m', '50-150m', '150-300m', '300-500m',
                   '500-700m', '700-850m', '850-1000m', f'DVM({dvm_lo}-{dvm_hi}m)']
    for z in range(8):
        band = hist_raw[:, z]
        active = (band > 0).sum()
        print(f"    Zone {z} ({zone_labels[z]}): "
              f"active={active}/{n_pings} ({100*active/n_pings:.0f}%), "
              f"local_range={local_ranges[z][0]:.2e}..{local_ranges[z][1]:.2e}")

    return hist_local, hist_global, diff_short, diff_medium


def compute_dvm_depth(sv_data, depth_values, config, sonification_mode=False,
                      time_seconds=None):
    """Compute DVM depth using corridor center-of-mass.

    v9: Replaces the dual-zone log-ratio with corridor center-of-mass (CoM)
    restricted to 50–600 m. This excludes both surface noise (0–50 m) and
    the persistent deep scattering layer (DSL, >600 m), which at 38 kHz is
    always brighter than the migrating layer due to myctophid swim bladder
    resonance. The DSL biased all previous ratio-based approaches.

    The corridor CoM directly tracks the energy-weighted mean depth of
    organisms within the DVM-active corridor. Compared to the v8 log-ratio:
      - Night depth: 156 m vs 308 m (echogram shows ~180 m → CoM is closer)
      - Day depth: 494 m vs 484 m (similar)
      - Total swing: 338 m vs 176 m (nearly 2× more pitch range)
      - Dusk recovery: 39% vs 22% (still limited by 38 kHz DSL physics)

    Auto-detection:
      After smoothing, the algorithm extracts night/day depth medians from the
      bimodal distribution of the CoM signal and detects sunrise/sunset from
      velocity extrema. All parameters are stored in the returned metadata dict
      so the SC sketch needs no hardcoded geographic/temporal values.

    Algorithm:
      1. Restrict Sv to corridor (50–600 m), convert to linear power
      2. Time-average each depth sample with a sliding window
      3. Compute energy-weighted center-of-mass per ping
      4. Median filter (preserves sharp dawn/dusk transitions)
      5. Uniform filter for musical continuity
      6. Auto-detect night/day depth from signal bimodality
      7. Percentile normalization → physical depth mapping
      8. Auto-detect sunrise/sunset from velocity peaks

    Parameters
    ----------
    sv_data : np.ndarray, shape (n_pings, n_depth)
        Raw Sv values in dB.
    depth_values : np.ndarray, shape (n_depth,)
        Depth array in meters.
    config : SonificationConfigV3
    sonification_mode : bool
    time_seconds : np.ndarray, optional
        Timestamp array for velocity/sunrise detection. If None, velocity
        and sunrise/sunset detection are skipped.

    Returns
    -------
    dvm_depth : np.ndarray, shape (n_pings,)
        Smoothed DVM depth in meters.
    dvm_meta : dict
        Auto-detected metadata:
          corridor_m: (lo, hi) — corridor boundaries used
          night_depth_m: median night depth
          day_depth_m: median day depth
          sunrise_h: detected sunrise hour (UTC)
          sunset_h: detected sunset hour (UTC)
          dawn_end_h: end of dawn descent
          dusk_start_h: start of dusk ascent
    """
    n_pings = sv_data.shape[0]
    threshold = config.min_sv_threshold_db

    if sonification_mode:
        avg_window = config.son_dvm_avg_window
        med_window = config.son_dvm_median_window
        smooth_window = config.son_dvm_smooth_window
    else:
        avg_window = config.dvm_avg_window
        med_window = config.dvm_median_window
        smooth_window = config.dvm_smooth_window

    # --- Step 1: Restrict to DVM corridor and convert to linear ---
    corridor_lo, corridor_hi = 50.0, 600.0
    corr_mask = (depth_values >= corridor_lo) & (depth_values < corridor_hi)
    corr_depth = depth_values[corr_mask]
    corr_sv = sv_data[:, corr_mask]
    corr_clean = np.where(
        np.isnan(corr_sv) | (corr_sv <= threshold), -90.0, corr_sv
    )
    corr_linear = 10 ** (corr_clean / 10)

    # --- Step 2: Time-average each depth sample ---
    corr_smooth = np.zeros_like(corr_linear)
    for j in range(corr_linear.shape[1]):
        corr_smooth[:, j] = uniform_filter1d(
            corr_linear[:, j], size=avg_window, mode='nearest'
        )

    # --- Step 3: Energy-weighted center-of-mass per ping ---
    total_energy = corr_smooth.sum(axis=1)
    total_energy = np.where(total_energy < 1e-20, 1e-20, total_energy)
    com_raw = (corr_smooth * corr_depth[np.newaxis, :]).sum(axis=1) / total_energy

    # --- Step 4: Median filter (edge-preserving) ---
    com_median = median_filter(com_raw, size=med_window, mode='nearest')

    # --- Step 5: Uniform filter for musical continuity ---
    com_smooth = uniform_filter1d(com_median, size=smooth_window, mode='nearest')

    # --- Step 6: Auto-detect night/day depth from bimodal distribution ---
    # The CoM signal is bimodal: shallow cluster (night) and deep cluster (day).
    # Use percentile separation: night = lower quartile, day = upper quartile.
    p15 = float(np.percentile(com_smooth, 15))
    p85 = float(np.percentile(com_smooth, 85))
    night_depth_m = float(np.median(com_smooth[com_smooth <= p15 + 0.3 * (p85 - p15)]))
    day_depth_m = float(np.median(com_smooth[com_smooth >= p85 - 0.3 * (p85 - p15)]))

    # --- Step 7: Percentile normalization → physical depth ---
    # Map the 15th–85th percentile range of the CoM to a physical depth range.
    # The auto-detected night/day medians anchor the endpoints.
    phase = np.clip((com_smooth - p15) / (p85 - p15 + 1e-10), 0, 1)
    # phase: 0 = shallow (night), 1 = deep (day)
    shallow_anchor = max(night_depth_m - 30, 100.0)  # margin below night depth
    deep_anchor = min(day_depth_m + 30, 650.0)        # margin above day depth
    dvm_depth = shallow_anchor + phase * (deep_anchor - shallow_anchor)

    # --- Step 8: Auto-detect sunrise/sunset from velocity ---
    sunrise_h, sunset_h = 5.75, 20.25  # defaults (South Atlantic Jan 2011)
    dawn_end_h, dusk_start_h = 8.5, 19.5
    if time_seconds is not None and len(time_seconds) == n_pings:
        hours = time_seconds / 3600.0
        # Velocity: negative gradient = descending (depth increasing)
        dvm_vel = np.gradient(com_smooth, time_seconds) * 3600  # m/h (positive=deepening)
        # Use heavy smoothing for sunrise/sunset detection (structural, not real-time)
        detect_window = max(smooth_window * 3, 601)
        dvm_vel_smooth = uniform_filter1d(dvm_vel, size=detect_window, mode='nearest')

        # Dawn = strongest positive velocity (descent, depth increasing)
        # Dusk = strongest negative velocity (ascent, depth decreasing)
        # Search in reasonable hour windows to avoid false peaks
        dawn_mask = (hours >= 3) & (hours <= 12)
        dusk_mask = (hours >= 15) & (hours <= 23)

        if np.any(dawn_mask):
            dawn_peak_idx = np.where(dawn_mask)[0][np.argmax(dvm_vel_smooth[dawn_mask])]
            dawn_peak_h = float(hours[dawn_peak_idx])
            # Sunrise ≈ 1h before peak descent velocity
            sunrise_h = round(max(dawn_peak_h - 1.0, 3.0), 2)
            # Dawn end ≈ when velocity drops below 30% of peak (after peak)
            dawn_peak_val = dvm_vel_smooth[dawn_peak_idx]
            post_dawn = np.where((hours > dawn_peak_h) & dawn_mask)[0]
            if len(post_dawn) > 0:
                below_thresh = post_dawn[dvm_vel_smooth[post_dawn] < dawn_peak_val * 0.3]
                if len(below_thresh) > 0:
                    dawn_end_h = round(float(hours[below_thresh[0]]), 2)
                else:
                    dawn_end_h = round(dawn_peak_h + 2.5, 2)
            else:
                dawn_end_h = round(dawn_peak_h + 2.5, 2)

        if np.any(dusk_mask):
            dusk_peak_idx = np.where(dusk_mask)[0][np.argmin(dvm_vel_smooth[dusk_mask])]
            dusk_peak_h = float(hours[dusk_peak_idx])
            # Dusk start ≈ 1h before peak ascent velocity
            dusk_start_h = round(max(dusk_peak_h - 1.0, 15.0), 2)
            # Sunset ≈ 0.75h after peak ascent velocity
            sunset_h = round(min(dusk_peak_h + 0.75, 23.5), 2)

        print(f"    Auto-detected: sunrise={sunrise_h:.2f}h, dawn_end={dawn_end_h:.2f}h, "
              f"dusk_start={dusk_start_h:.2f}h, sunset={sunset_h:.2f}h")

    print(f"    Corridor CoM ({corridor_lo:.0f}-{corridor_hi:.0f}m): "
          f"night={night_depth_m:.0f}m, day={day_depth_m:.0f}m, "
          f"swing={abs(day_depth_m - night_depth_m):.0f}m")
    print(f"    Depth mapping: {shallow_anchor:.0f}m (shallow) to {deep_anchor:.0f}m (deep)")

    dvm_meta = {
        'corridor_m': (corridor_lo, corridor_hi),
        'night_depth_m': night_depth_m,
        'day_depth_m': day_depth_m,
        'shallow_anchor_m': shallow_anchor,
        'deep_anchor_m': deep_anchor,
        'sunrise_h': sunrise_h,
        'sunset_h': sunset_h,
        'dawn_end_h': dawn_end_h,
        'dusk_start_h': dusk_start_h,
    }

    return dvm_depth, dvm_meta


def compute_regime_changepoints(features, config):
    """Derivative-based transition detection on composite acoustic signal.

    Computes continuous transition intensity (high = regime shift happening)
    and discrete regime labels from peak detection of major transitions.

    Returns
    -------
    regime_change_score : np.ndarray, shape (n_pings,)
        Continuous [0, 1]. High = significant regime transition happening.
    regime_id : np.ndarray, shape (n_pings,)
        Integer regime label (increments at each major transition).
    """
    n_pings = len(features)

    # Build composite signal from 3 orthogonal features
    intensity = np.array([f['total_intensity_db'] for f in features])
    depth = np.array([f['center_of_mass_m'] for f in features])
    entropy = np.array([f['acoustic_entropy'] for f in features])

    # Z-score normalize each
    def zscore(arr):
        mu = np.nanmean(arr)
        sigma = np.nanstd(arr)
        return (np.nan_to_num(arr, nan=mu) - mu) / (sigma + 1e-10)

    composite = np.abs(zscore(intensity)) + np.abs(zscore(depth)) + np.abs(zscore(entropy))

    # Heavily smooth to get baseline trend (~15 min window)
    trend = uniform_filter1d(composite, size=300, mode='nearest')

    # Transition signal = absolute rate of change of trend
    transition_raw = np.abs(np.gradient(trend))
    transition_smooth = uniform_filter1d(transition_raw, size=60, mode='nearest')

    # Normalize to [0, 1]
    t_max = np.max(transition_smooth)
    regime_change_score = transition_smooth / (t_max + 1e-10)

    # Find major transition peaks → regime boundaries
    # distance=600 (~30 min) ensures regimes are at least 30 min apart
    peaks, _ = find_peaks(regime_change_score, height=0.4, distance=600, prominence=0.1)

    # Build regime IDs from peaks
    regime_id = np.zeros(n_pings, dtype=int)
    current = 0
    peak_set = set(peaks)
    for i in range(n_pings):
        if i in peak_set:
            current += 1
        regime_id[i] = current

    return regime_change_score, regime_id


def detect_layer_events(features, config):
    """Detect layer birth/death events by comparing peaks between consecutive pings.

    Returns
    -------
    event_types : np.ndarray, shape (n_pings,)
        1 = birth (new layer), -1 = death (layer vanished), 0 = no event.
    event_depths : np.ndarray, shape (n_pings,)
        Depth of the born/died layer (meters). 0 if no event.
    event_counts_diff : np.ndarray, shape (n_pings,)
        layer_count[i] - layer_count[i-1].
    """
    n_pings = len(features)
    event_types = np.zeros(n_pings, dtype=int)
    event_depths = np.zeros(n_pings)
    event_counts_diff = np.zeros(n_pings, dtype=int)
    max_dist = config.layer_match_max_distance_m

    for i in range(1, n_pings):
        prev_peaks = list(features[i - 1].get('peak_depths', []))
        curr_peaks = list(features[i].get('peak_depths', []))

        event_counts_diff[i] = len(curr_peaks) - len(prev_peaks)

        # Greedy nearest-neighbor matching
        matched_prev = set()
        matched_curr = set()

        if prev_peaks and curr_peaks:
            # Build all pairs sorted by distance
            pairs = []
            for ci, cd in enumerate(curr_peaks):
                for pi, pd in enumerate(prev_peaks):
                    pairs.append((abs(cd - pd), ci, pi))
            pairs.sort()

            for dist, ci, pi in pairs:
                if ci in matched_curr or pi in matched_prev:
                    continue
                if dist <= max_dist:
                    matched_curr.add(ci)
                    matched_prev.add(pi)

        births = [curr_peaks[ci] for ci in range(len(curr_peaks)) if ci not in matched_curr]
        deaths = [prev_peaks[pi] for pi in range(len(prev_peaks)) if pi not in matched_prev]

        if len(births) > len(deaths):
            event_types[i] = 1
            event_depths[i] = births[0]  # shallowest new peak
        elif len(deaths) > len(births):
            event_types[i] = -1
            event_depths[i] = deaths[0]
        # else: balanced or no events → 0

    return event_types, event_depths, event_counts_diff


def track_layers(features, config):
    """Track persistent layer identities across pings using greedy matching.

    Returns
    -------
    tracked_depths : np.ndarray, shape (n_pings, max_tracked_layers)
        Depth of each tracked layer (0 if slot empty).
    tracked_ages : np.ndarray, shape (n_pings, max_tracked_layers)
        Age of each tracked layer in pings (0 if slot empty).
    """
    n_pings = len(features)
    max_layers = config.max_tracked_layers
    max_dist = config.layer_match_max_distance_m

    tracked_depths = np.zeros((n_pings, max_layers))
    tracked_ages = np.zeros((n_pings, max_layers))

    # Active layer state: [depth, age, unmatched_count] per slot
    active = [[0.0, 0, 99] for _ in range(max_layers)]  # 99 = effectively dead

    for i in range(n_pings):
        curr_peaks = list(features[i].get('peak_depths', []))

        # Build cost matrix: distance from each active slot to each current peak
        matched_slots = set()
        matched_peaks = set()

        if curr_peaks:
            pairs = []
            for si in range(max_layers):
                if active[si][1] == 0 and active[si][2] >= 99:
                    continue  # empty slot
                for pi, pd in enumerate(curr_peaks):
                    dist = abs(active[si][0] - pd)
                    pairs.append((dist, si, pi))
            pairs.sort()

            for dist, si, pi in pairs:
                if si in matched_slots or pi in matched_peaks:
                    continue
                if dist <= max_dist:
                    matched_slots.add(si)
                    matched_peaks.add(pi)
                    active[si][0] = curr_peaks[pi]  # update depth
                    active[si][1] += 1               # increment age
                    active[si][2] = 0                # reset unmatched count

        # Unmatched active slots: increment unmatched count
        for si in range(max_layers):
            if si not in matched_slots and active[si][1] > 0:
                active[si][2] += 1
                if active[si][2] > 3:
                    # Declare dead
                    active[si] = [0.0, 0, 99]

        # Unmatched new peaks: assign to empty slots
        for pi in range(len(curr_peaks)):
            if pi in matched_peaks:
                continue
            # Find empty slot (age=0, unmatched>=99)
            empty_slot = None
            for si in range(max_layers):
                if active[si][1] == 0 and active[si][2] >= 99:
                    empty_slot = si
                    break
            if empty_slot is None:
                # Replace shortest-lived active layer
                min_age = float('inf')
                for si in range(max_layers):
                    if active[si][1] < min_age:
                        min_age = active[si][1]
                        empty_slot = si
            if empty_slot is not None:
                active[empty_slot] = [curr_peaks[pi], 1, 0]

        # Record state
        for si in range(max_layers):
            tracked_depths[i, si] = active[si][0]
            tracked_ages[i, si] = active[si][1]

    return tracked_depths, tracked_ages


def load_and_extract(date_str, config=None):
    """
    Load cleaned 24h data from NetCDF and extract sonification features.

    Reads the pre-cleaned NetCDF produced by run_24h_processing.py,
    which already has calibrated, noise-removed, impulse-masked 38 kHz Sv data.

    Parameters
    ----------
    date_str : str
        Date in YYYYMMDD format.
    config : SonificationConfigV3, optional
        Extraction configuration. Uses defaults if not provided.

    Returns
    -------
    features : list[dict]
        Per-ping feature dictionaries including derived features.
    sc_data : dict
        SuperCollider-ready JSON structure with normalized arrays.
    """
    if config is None:
        config = SonificationConfigV3()

    nc_path = OUTPUT_DATA / f"cleaned_Sv_24h_{date_str}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Cleaned NetCDF not found: {nc_path}\n"
            f"Run 'python src/processing/run_24h_processing.py {date_str}' first."
        )

    print("=" * 70)
    print("SONIFICATION FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Date: {date_str}")
    print(f"Source: {nc_path.name}")
    print("=" * 70)

    # Load cleaned data
    print("\nLoading cleaned NetCDF...")
    ds = xr.open_dataset(nc_path)

    sv_data = ds['Sv'].values          # (n_pings, n_depth)
    depths = ds['depth'].values        # (n_depth,)
    times = ds['ping_time'].values     # (n_pings,)
    ds.close()

    n_pings, n_depth = sv_data.shape
    print(f"  {n_pings} pings × {n_depth} depth samples")
    print(f"  Depth range: 0 – {depths[-1]:.0f} m")
    print(f"  Time range:  {times[0]} → {times[-1]}")

    # Compute time in seconds from start of day
    start_time = times[0]
    time_seconds = (times - start_time) / np.timedelta64(1, 's')
    time_seconds = np.array(time_seconds, dtype=float)

    # Ensure depths are finite floats
    depth_values = np.array(depths, dtype=float)
    depth_values = np.nan_to_num(depth_values, nan=0)

    # Extract per-ping features
    print(f"\nExtracting features from {n_pings} pings...")
    all_features = []
    prev_sv = None  # for inter_ping_correlation

    for ping_idx in range(n_pings):
        if ping_idx % 5000 == 0:
            print(f"  [{ping_idx+1}/{n_pings}] ({100*ping_idx/n_pings:.0f}%)")

        sv_column = sv_data[ping_idx, :]

        # Inter-ping correlation: cosine similarity with previous ping
        if prev_sv is None:
            ipc = 1.0
        else:
            valid = ~(np.isnan(sv_column) | np.isnan(prev_sv))
            if valid.sum() > 0:
                a, b = sv_column[valid], prev_sv[valid]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                ipc = float(np.clip(np.dot(a, b) / (na * nb + 1e-10), 0.0, 1.0))
            else:
                ipc = 0.0
        prev_sv = sv_column.copy()

        # Extract per-ping features
        feats = extract_ping_features(sv_column, depth_values, config)

        # Calculate hour of day
        ping_dt = times[ping_idx].astype('datetime64[s]').astype(datetime)
        hour_of_day = ping_dt.hour + ping_dt.minute / 60

        feat_dict = {
            'timestamp': str(times[ping_idx]),
            'timestamp_seconds': float(time_seconds[ping_idx]),
            'hour_of_day': hour_of_day,
            'inter_ping_correlation': ipc,
            **feats
        }
        all_features.append(feat_dict)

    print(f"  Extracted {len(all_features)} points")

    # Count NaN pings (impulse noise — no valid signal)
    nan_pings = sum(1 for f in all_features if np.isnan(f['center_of_mass_m']))
    print(f"  NaN pings (impulse noise): {nan_pings} ({100*nan_pings/len(all_features):.1f}%)")

    # Save raw features before derived computation (for v6 re-derivation)
    all_features_raw = [dict(f) for f in all_features]

    # Add derived features (velocity, acceleration, anomaly, outlier) — scientific smoothing
    print("\nComputing derived features (velocity, acceleration, anomaly, outlier)...")
    all_features = add_derived_features(all_features, config)

    # Multi-lag autocorrelation (needs raw Sv profiles, not just feature dicts)
    print("Computing multi-lag autocorrelation (3-min, 10-min)...")
    autocorr_3min, autocorr_10min = compute_lagged_autocorrelation(sv_data, time_seconds)
    for i, f in enumerate(all_features):
        f['autocorr_3min'] = float(autocorr_3min[i])
        f['autocorr_10min'] = float(autocorr_10min[i])

    # Create SuperCollider v5 format (scientific, unchanged)
    print("\nCreating SuperCollider v5 JSON (scientific)...")
    sc_data_v5 = create_supercollider_format_v5(all_features, config)
    sc_path_v5 = OUTPUT_DATA / f"sonification_sc_v5_{date_str}.json"
    with open(sc_path_v5, 'w') as f:
        json.dump(sc_data_v5, f, indent=2)
    print(f"Saved: {sc_path_v5}")

    # ================================================================
    # v6: Sonification-optimized export (shorter smoothing + events)
    # ================================================================
    print("\n" + "-" * 50)
    print("Computing v6 sonification-optimized features...")
    print("-" * 50)

    # Re-derive with sonification smoothing windows
    son_features = [dict(f) for f in all_features_raw]
    son_features = add_derived_features(son_features, config, sonification_mode=True)

    # Attach autocorrelation (raw, no smoothing for v6)
    for i, f in enumerate(son_features):
        f['autocorr_3min'] = float(autocorr_3min[i])
        f['autocorr_10min'] = float(autocorr_10min[i])

    # Phase 2: Event-level features
    print("  Computing 8-band histogram + differential...")
    hist_8band, hist_diff = compute_histogram_differential(son_features, config)

    print("  Computing regime changepoints (CUSUM)...")
    regime_score, regime_id = compute_regime_changepoints(son_features, config)

    print("  Detecting layer events (births/deaths)...")
    event_types, event_depths, event_counts = detect_layer_events(son_features, config)

    print("  Tracking persistent layers...")
    tracked_depths, tracked_ages = track_layers(son_features, config)

    # Export v6 JSON
    print("\nCreating SuperCollider v6 JSON (sonification)...")
    sc_data_v6 = create_supercollider_format_v6(
        son_features, config, hist_8band, hist_diff,
        regime_score, regime_id, event_types, event_depths,
        tracked_depths, tracked_ages
    )
    sc_path_v6 = OUTPUT_DATA / f"sonification_sc_v6_{date_str}.json"
    with open(sc_path_v6, 'w') as f:
        json.dump(sc_data_v6, f, indent=2)
    print(f"Saved: {sc_path_v6}")

    # Print v6 verification stats
    print("\n  v6 Feature Statistics:")
    son_vel = np.array([f['velocity_m_h'] for f in son_features])
    sci_vel = np.array([f['velocity_m_h'] for f in all_features])
    print(f"    Velocity std — scientific: {sci_vel.std():.1f}, sonification: {son_vel.std():.1f} m/h")
    son_onset = np.array([f['onset_strength'] for f in son_features])
    sci_onset = np.array([f['onset_strength'] for f in all_features])
    print(f"    Onset std — scientific: {sci_onset.std():.2e}, sonification: {son_onset.std():.2e}")
    n_cp = int((np.diff(regime_id) > 0).sum())
    print(f"    Regime changepoints: {n_cp}")
    n_births = int((event_types == 1).sum())
    n_deaths = int((event_types == -1).sum())
    print(f"    Layer events — births: {n_births}, deaths: {n_deaths}")
    active_mask = tracked_ages > 0
    if active_mask.any():
        print(f"    Tracked layer ages — mean: {tracked_ages[active_mask].mean():.0f} pings, max: {tracked_ages.max():.0f} pings")
    for b in range(8):
        std = hist_diff[:, b].std()
        print(f"    Histogram diff band {b} std: {std:.6f}")

    # ================================================================
    # v7: Fixed 8-band zones + dual normalization + multi-scale diff
    # ================================================================
    print("\n" + "-" * 50)
    print("Computing v8 sonification features (corridor CoM)...")
    print("-" * 50)

    print("  Computing 8-band histogram directly from raw Sv...")
    hist_local, hist_global, diff_short, diff_medium = compute_histogram_zones(
        sv_data, depth_values, config
    )

    print("  Computing DVM depth (corridor center-of-mass)...")
    son_times = np.array([f['timestamp_seconds'] for f in son_features])
    dvm_depth_smooth, dvm_meta = compute_dvm_depth(
        sv_data, depth_values, config, sonification_mode=True,
        time_seconds=son_times
    )

    # Export v8 JSON (corridor CoM + auto-detect)
    print("\nCreating SuperCollider v8 JSON...")
    sc_data_v7 = create_supercollider_format_v7(
        son_features, config,
        hist_local, hist_global, diff_short, diff_medium,
        regime_score, regime_id, event_types, event_depths,
        tracked_depths, tracked_ages,
        dvm_depth_smooth=dvm_depth_smooth,
        dvm_meta=dvm_meta
    )
    sc_path_v8 = OUTPUT_DATA / f"sonification_sc_v8_{date_str}.json"
    with open(sc_path_v8, 'w') as f:
        json.dump(sc_data_v7, f, indent=2)
    print(f"Saved: {sc_path_v8}")

    # v8 verification stats
    print("\n  v8 Feature Statistics:")
    print(f"    Zone 7 = DVM corridor ({config.dvm_corridor[0]}-{config.dvm_corridor[1]}m)")
    for b in range(8):
        local_std = hist_local[:, b].std()
        global_std = hist_global[:, b].std()
        diff_s_std = diff_short[:, b].std()
        diff_m_std = diff_medium[:, b].std()
        print(f"    Band {b}: local_std={local_std:.3f}  global_std={global_std:.3f}  "
              f"diff_short_std={diff_s_std:.3f}  diff_medium_std={diff_m_std:.3f}")

    # DVM depth statistics (using auto-detected sunrise/sunset from dvm_meta)
    dvm_depths = dvm_depth_smooth
    com_depths = np.array([f['depth_smooth_m'] for f in son_features])
    hours = np.array([f['hour_of_day'] for f in son_features])
    if dvm_meta is not None:
        sunrise_h = dvm_meta.get('sunrise_h', 5.75)
        sunset_h = dvm_meta.get('sunset_h', 20.25)
    else:
        sunrise_h, sunset_h = 5.75, 20.25
    night = (hours < sunrise_h - 1) | (hours > sunset_h + 1)
    day = (hours > sunrise_h + 3) & (hours < sunset_h - 1)
    print(f"\n  DVM Depth Statistics (sunrise={sunrise_h:.1f}h, sunset={sunset_h:.1f}h):")
    if dvm_meta is not None:
        print(f"    Auto-detected: night={dvm_meta.get('night_depth_m', '?'):.0f}m, "
              f"day={dvm_meta.get('day_depth_m', '?'):.0f}m")
        print(f"    Anchors: shallow={dvm_meta.get('shallow_anchor_m', '?'):.0f}m, "
              f"deep={dvm_meta.get('deep_anchor_m', '?'):.0f}m")
    print(f"    DVM track — night: {np.nanmean(dvm_depths[night]):.0f}m, "
          f"day: {np.nanmean(dvm_depths[day]):.0f}m, "
          f"swing: {np.nanmean(dvm_depths[day]) - np.nanmean(dvm_depths[night]):.0f}m")
    print(f"    CoM       — night: {np.nanmean(com_depths[night]):.0f}m, "
          f"day: {np.nanmean(com_depths[day]):.0f}m, "
          f"swing: {np.nanmean(com_depths[day]) - np.nanmean(com_depths[night]):.0f}m")
    print(f"    DVM range: {np.nanmin(dvm_depths):.0f} – {np.nanmax(dvm_depths):.0f}m")

    return all_features, sc_data_v5


def _percentile_range(arr, lo=2, hi=98):
    """Return (p_lo, p_hi) from actual data, ignoring NaN. Safe for collapsed ranges."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    if np.isclose(p_lo, p_hi):
        p_hi = p_lo + 1e-3
    return float(p_lo), float(p_hi)


def create_supercollider_format_v5(features, config):
    """Create SuperCollider v5 format with v4 fields + depth histogram, peak props, shape metrics."""

    def normalize(arr, min_val, max_val):
        return np.clip((arr - min_val) / (max_val - min_val + 1e-10), 0, 1)

    # Extract arrays (v4)
    times = np.array([f['timestamp_seconds'] for f in features])
    depth = np.array([f['center_of_mass_m'] for f in features])
    depth_smooth = np.array([f['depth_smooth_m'] for f in features])
    intensity = np.array([f['total_intensity_db'] for f in features])
    spread = np.array([f['vertical_spread_m'] for f in features])
    layers = np.array([f['layer_count'] for f in features], dtype=float)
    velocity = np.array([f['velocity_m_h'] for f in features])
    acceleration = np.array([f['acceleration_m_h2'] for f in features])
    anomaly = np.array([f['depth_anomaly_m'] for f in features])
    spread_change = np.array([f['spread_change_m_min'] for f in features])
    hour = np.array([f['hour_of_day'] for f in features])
    surface = np.array([f['surface_intensity_db'] for f in features])
    midwater = np.array([f['midwater_intensity_db'] for f in features])
    deep = np.array([f['deep_intensity_db'] for f in features])
    ipc = np.array([f['inter_ping_correlation'] for f in features])
    entropy = np.array([f['acoustic_entropy'] for f in features])
    layer_sep = np.array([f['layer_separation_m'] for f in features])
    onset = np.array([f['onset_strength'] for f in features])

    # Extract arrays (v5 new)
    skewness_arr = np.array([f['dist_skewness'] for f in features])
    kurtosis_arr = np.array([f['dist_kurtosis'] for f in features])
    vert_grad_arr = np.array([f['vertical_gradient_db_m'] for f in features])
    peak_prom_arr = np.array([f['peak_max_prominence_db'] for f in features])
    peak_width_arr = np.array([f['peak_mean_width_m'] for f in features])

    # Extract arrays (Tier 2 additions)
    velocity_stable = np.array([f['velocity_stable_m_h'] for f in features])
    outlier_arr = np.array([f['outlier_score'] for f in features])
    autocorr_3min_arr = np.array([f['autocorr_3min'] for f in features])
    autocorr_10min_arr = np.array([f['autocorr_10min'] for f in features])
    peak_height_arr = np.array([f['peak_max_height_db'] for f in features])
    hist_matrix = np.array([f['depth_histogram'] for f in features])  # (n_pings, 32)

    # Handle NaN
    depth = np.nan_to_num(depth, nan=500)
    depth_smooth = np.nan_to_num(depth_smooth, nan=500)

    # ------------------------------------------------------------------
    # Data-driven normalization ranges (2nd–98th percentile)
    # ------------------------------------------------------------------
    intensity_min, intensity_max = _percentile_range(intensity)
    spread_min, spread_max       = _percentile_range(spread)
    velocity_min, velocity_max   = _percentile_range(velocity)
    accel_min, accel_max         = _percentile_range(acceleration)
    spread_chg_min, spread_chg_max = _percentile_range(spread_change)
    anom_abs = float(np.nanpercentile(np.abs(anomaly), 98)) if np.any(~np.isnan(anomaly)) else 200.0
    anomaly_min, anomaly_max = -max(anom_abs, 1.0), max(anom_abs, 1.0)
    _, sep_max = _percentile_range(layer_sep)
    _, onset_max = _percentile_range(onset)
    surface_min, surface_max     = _percentile_range(surface)
    midwater_min, midwater_max   = _percentile_range(midwater)
    deep_min, deep_max           = _percentile_range(deep)

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------
    # Depth: inverted (shallow=1, deep=0) — semantic anchor kept at 100–800m
    depth_norm = 1.0 - normalize(depth_smooth, 100, 800)

    intensity_norm = normalize(intensity, intensity_min, intensity_max)
    spread_norm = normalize(spread, spread_min, spread_max)

    # Layers: 0–8 typical maximum
    layers_norm = normalize(layers, 0, 8)

    # Velocity: 0=descending fast, 0.5=stationary, 1=ascending fast
    velocity_norm = normalize(velocity, velocity_min, velocity_max)
    acceleration_norm = normalize(acceleration, accel_min, accel_max)

    # Anomaly: inverted to match depth convention (deep=0, shallow=1)
    # 0=deeper than expected, 0.5=normal, 1=shallower than expected
    anomaly_norm = 1.0 - normalize(anomaly, anomaly_min, anomaly_max)

    spread_change_norm = normalize(spread_change, spread_chg_min, spread_chg_max)

    hour_norm = hour / 24.0

    surface_norm  = normalize(surface, surface_min, surface_max)
    midwater_norm = normalize(midwater, midwater_min, midwater_max)
    deep_norm     = normalize(deep, deep_min, deep_max)

    # New features (ipc and entropy already [0,1])
    layer_sep_norm = normalize(layer_sep, 0.0, max(sep_max, 10.0))
    onset_norm     = normalize(onset, 0.0, max(onset_max, 1e-10))

    # ------------------------------------------------------------------
    # v5: Normalize new features
    # ------------------------------------------------------------------
    skew_min, skew_max = _percentile_range(skewness_arr)
    kurt_min, kurt_max = _percentile_range(kurtosis_arr)
    vgrad_min, vgrad_max = _percentile_range(vert_grad_arr)
    pprom_min, pprom_max = _percentile_range(peak_prom_arr)
    pwidth_min, pwidth_max = _percentile_range(peak_width_arr)

    skewness_norm = normalize(skewness_arr, skew_min, skew_max)
    kurtosis_norm = normalize(kurtosis_arr, kurt_min, kurt_max)
    vert_grad_norm = normalize(vert_grad_arr, vgrad_min, vgrad_max)
    peak_prom_norm = normalize(peak_prom_arr, pprom_min, pprom_max)
    peak_width_norm = normalize(peak_width_arr, pwidth_min, pwidth_max)

    # ------------------------------------------------------------------
    # Tier 2: Normalize new features
    # ------------------------------------------------------------------
    vel_stable_min, vel_stable_max = _percentile_range(velocity_stable)
    velocity_stable_norm = normalize(velocity_stable, vel_stable_min, vel_stable_max)

    # Outlier: fixed range (z-score 0-4), not percentile-based
    outlier_norm = normalize(outlier_arr, 0.0, 4.0)

    # Autocorrelation: already in [0,1] (cosine similarity), smooth for SC
    autocorr_3min_smooth = uniform_filter1d(autocorr_3min_arr, 30, mode='nearest')
    autocorr_10min_smooth = uniform_filter1d(autocorr_10min_arr, 30, mode='nearest')

    # Histogram: normalize each band independently across time (column-wise)
    n_bands = hist_matrix.shape[1] if hist_matrix.ndim == 2 else config.histogram_bands
    hist_norm_matrix = np.zeros_like(hist_matrix)
    hist_band_ranges = []
    for b in range(n_bands):
        band = hist_matrix[:, b]
        b_min, b_max = _percentile_range(band)
        hist_norm_matrix[:, b] = normalize(band, b_min, b_max)
        hist_band_ranges.append([float(b_min), float(b_max)])

    sc_data = {
        'info': {
            'version': 'v5',
            'description': '38 kHz sonification features (v5: depth histogram, peak properties, shape metrics)',
            'source': 'Cleaned NetCDF (noise-removed, impulse-masked)',
            'sample_rate_hz': float(1.0 / np.median(np.diff(times))) if len(times) > 1 else 0.3,
            'duration_seconds': float(times[-1]) if len(times) > 0 else 0,
            'num_points': len(times),
            'normalization': {
                'depth':       '0=deep (800m), 1=shallow (100m)',
                'intensity':   '0=quiet (data p2), 1=loud (data p98)',
                'velocity':    '0=descending fast, 0.5=stationary, 1=ascending fast',
                'anomaly':     '0=deeper than expected, 0.5=normal, 1=shallower than expected',
                'hour':        '0=midnight, 0.5=noon, 1=midnight',
                'entropy':     '0=tonal (concentrated layer), 1=noise-like (diffuse)',
                'ipc':         '0=turbulent/changing, 1=stable/sustained',
                'layer_sep':   '0=single layer, 1=maximum separation (data p98)',
                'onset':       '0=gradual, 1=sudden biomass change (data p98)',
                'skewness':    '0=left-skewed (data p2), 1=right-skewed (data p98)',
                'kurtosis':    '0=flat distribution (data p2), 1=peaked (data p98)',
                'vert_grad':   '0=smooth profile, 1=sharp boundaries (data p98)',
                'peak_prom':   '0=no prominent peaks, 1=strong layer distinctness (data p98)',
                'peak_width':  '0=thin layers, 1=thick layers (data p98)',
                'histogram':       '32-band depth energy profile, each band normalized [0,1] independently',
                'velocity_stable':   '0=descending fast (fine), 0.5=stationary, 1=ascending fast (5-min window)',
                'outlier':         '0=normal, 1=extreme anomaly (z-score >= 4)',
                'autocorr_3min':   '0=turbulent (3-min lag), 1=periodic/stable',
                'autocorr_10min':  '0=turbulent (10-min lag), 1=periodic/stable',
            },
            'normalization_ranges': {
                'intensity_db':  [intensity_min, intensity_max],
                'spread_m':      [spread_min, spread_max],
                'velocity_m_h':  [velocity_min, velocity_max],
                'accel_m_h2':    [accel_min, accel_max],
                'anomaly_m':     [anomaly_min, anomaly_max],
                'layer_sep_m':   [0.0, float(sep_max)],
                'onset':         [0.0, float(onset_max)],
                'surface_db':    [surface_min, surface_max],
                'midwater_db':   [midwater_min, midwater_max],
                'deep_db':       [deep_min, deep_max],
                # v5 ranges
                'skewness':      [float(skew_min), float(skew_max)],
                'kurtosis':      [float(kurt_min), float(kurt_max)],
                'vertical_gradient_db_m': [float(vgrad_min), float(vgrad_max)],
                'peak_prominence_db':     [float(pprom_min), float(pprom_max)],
                'peak_width_m':           [float(pwidth_min), float(pwidth_max)],
                'histogram_bands':        hist_band_ranges,
                # Tier 2 ranges
                'velocity_stable_m_h': [float(vel_stable_min), float(vel_stable_max)],
                'outlier_score':     [0.0, 4.0],
            }
        },
        '38kHz': {
            # Time
            'time_seconds': times.tolist(),
            'hour_norm': hour_norm.tolist(),

            # Core features (normalized)
            'depth_norm': depth_norm.tolist(),
            'intensity_norm': intensity_norm.tolist(),
            'spread_norm': spread_norm.tolist(),
            'layers_norm': layers_norm.tolist(),

            # Dynamic features (normalized)
            'velocity_norm': velocity_norm.tolist(),
            'acceleration_norm': acceleration_norm.tolist(),
            'anomaly_norm': anomaly_norm.tolist(),
            'spread_change_norm': spread_change_norm.tolist(),

            # Depth zone intensities
            'surface_norm': surface_norm.tolist(),
            'midwater_norm': midwater_norm.tolist(),
            'deep_norm': deep_norm.tolist(),

            # New features (v4)
            'inter_ping_correlation': ipc.tolist(),
            'acoustic_entropy': entropy.tolist(),
            'layer_separation_norm': layer_sep_norm.tolist(),
            'onset_strength_norm': onset_norm.tolist(),

            # Raw values for display / debugging
            'depth_m': depth_smooth.tolist(),
            'velocity_m_h': velocity.tolist(),
            'intensity_db': intensity.tolist(),
            'hour_of_day': hour.tolist(),

            # v5: Depth histogram (32 bands per ping, column-normalized)
            'depth_histogram': np.round(hist_norm_matrix, 6).tolist(),

            # v5: Distribution shape metrics
            'skewness_norm': skewness_norm.tolist(),
            'kurtosis_norm': kurtosis_norm.tolist(),

            # v5: Vertical gradient
            'vertical_gradient_norm': vert_grad_norm.tolist(),

            # v5: Peak properties
            'peak_prominence_norm': peak_prom_norm.tolist(),
            'peak_width_norm': peak_width_norm.tolist(),

            # v5: Raw peak height for display
            'peak_max_height_db': peak_height_arr.tolist(),

            # Tier 2: Stable-window velocity (5-min window)
            'velocity_stable_norm': velocity_stable_norm.tolist(),
            'velocity_stable_m_h': velocity_stable.tolist(),

            # Tier 2: Outlier score (z-score based)
            'outlier_norm': outlier_norm.tolist(),

            # Tier 2: Multi-lag autocorrelation
            'autocorr_3min_norm': autocorr_3min_smooth.tolist(),
            'autocorr_10min_norm': autocorr_10min_smooth.tolist(),
        }
    }

    return sc_data


def create_supercollider_format_v6(features, config, hist_8band, hist_diff,
                                    regime_score, regime_id,
                                    event_types, event_depths,
                                    tracked_depths, tracked_ages):
    """Create SuperCollider v6 format: sonification-optimized smoothing + event features.

    Drops redundant features (surface/midwater/deep/vert_grad/autocorr).
    Adds: 8-band histogram, histogram differential, layer events, regime changes, layer tracking.
    """
    def normalize(arr, min_val, max_val):
        return np.clip((arr - min_val) / (max_val - min_val + 1e-10), 0, 1)

    n_pings = len(features)

    # Extract arrays
    times = np.array([f['timestamp_seconds'] for f in features])
    depth_smooth = np.array([f['depth_smooth_m'] for f in features])
    intensity = np.array([f['total_intensity_db'] for f in features])
    spread = np.array([f['vertical_spread_m'] for f in features])
    layers = np.array([f['layer_count'] for f in features], dtype=float)
    velocity = np.array([f['velocity_m_h'] for f in features])
    velocity_stable = np.array([f['velocity_stable_m_h'] for f in features])
    acceleration = np.array([f['acceleration_m_h2'] for f in features])
    anomaly = np.array([f['depth_anomaly_m'] for f in features])
    spread_change = np.array([f['spread_change_m_min'] for f in features])
    hour = np.array([f['hour_of_day'] for f in features])
    ipc = np.array([f['inter_ping_correlation'] for f in features])
    entropy = np.array([f['acoustic_entropy'] for f in features])
    layer_sep = np.array([f['layer_separation_m'] for f in features])
    onset = np.array([f['onset_strength'] for f in features])
    outlier_arr = np.array([f['outlier_score'] for f in features])
    skewness_arr = np.array([f['dist_skewness'] for f in features])
    kurtosis_arr = np.array([f['dist_kurtosis'] for f in features])
    peak_prom_arr = np.array([f['peak_max_prominence_db'] for f in features])
    peak_width_arr = np.array([f['peak_mean_width_m'] for f in features])

    # Handle NaN
    depth_smooth = np.nan_to_num(depth_smooth, nan=500)

    # Normalization ranges (2nd-98th percentile)
    intensity_min, intensity_max = _percentile_range(intensity)
    spread_min, spread_max = _percentile_range(spread)
    velocity_min, velocity_max = _percentile_range(velocity)
    vel_stable_min, vel_stable_max = _percentile_range(velocity_stable)
    accel_min, accel_max = _percentile_range(acceleration)
    spread_chg_min, spread_chg_max = _percentile_range(spread_change)
    anom_abs = float(np.nanpercentile(np.abs(anomaly), 98)) if np.any(~np.isnan(anomaly)) else 200.0
    anomaly_min, anomaly_max = -max(anom_abs, 1.0), max(anom_abs, 1.0)
    _, sep_max = _percentile_range(layer_sep)
    _, onset_max = _percentile_range(onset)
    skew_min, skew_max = _percentile_range(skewness_arr)
    kurt_min, kurt_max = _percentile_range(kurtosis_arr)
    pprom_min, pprom_max = _percentile_range(peak_prom_arr)
    pwidth_min, pwidth_max = _percentile_range(peak_width_arr)

    # Normalize core features
    depth_norm = 1.0 - normalize(depth_smooth, 100, 800)
    intensity_norm = normalize(intensity, intensity_min, intensity_max)
    spread_norm = normalize(spread, spread_min, spread_max)
    layers_norm = normalize(layers, 0, 8)
    velocity_norm = normalize(velocity, velocity_min, velocity_max)
    velocity_stable_norm = normalize(velocity_stable, vel_stable_min, vel_stable_max)
    acceleration_norm = normalize(acceleration, accel_min, accel_max)
    anomaly_norm = 1.0 - normalize(anomaly, anomaly_min, anomaly_max)
    spread_change_norm = normalize(spread_change, spread_chg_min, spread_chg_max)
    hour_norm = hour / 24.0
    layer_sep_norm = normalize(layer_sep, 0.0, max(sep_max, 10.0))
    onset_norm = normalize(onset, 0.0, max(onset_max, 1e-10))
    outlier_norm = normalize(outlier_arr, 0.0, 4.0)
    skewness_norm = normalize(skewness_arr, skew_min, skew_max)
    kurtosis_norm = normalize(kurtosis_arr, kurt_min, kurt_max)
    peak_prom_norm = normalize(peak_prom_arr, pprom_min, pprom_max)
    peak_width_norm = normalize(peak_width_arr, pwidth_min, pwidth_max)

    # Normalize event features
    # Layer events: -1=death→0, 0=none→0.5, 1=birth→1
    layer_event_norm = (event_types.astype(float) + 1.0) / 2.0
    layer_event_depth_norm = normalize(event_depths, 0, config.max_depth_m)

    # Regime change score: already [0,1]
    regime_change_norm = regime_score.copy()
    n_regimes = max(int(regime_id.max()), 1)
    regime_id_norm = regime_id.astype(float) / n_regimes

    # Histogram differential: normalize per band to [0,1] (center at 0.5 = no change)
    hist_diff_norm = np.zeros_like(hist_diff)
    hist_diff_ranges = []
    for b in range(hist_diff.shape[1]):
        band = hist_diff[:, b]
        abs_max = max(float(np.percentile(np.abs(band), 98)), 1e-10)
        hist_diff_norm[:, b] = np.clip((band / abs_max + 1.0) / 2.0, 0, 1)
        hist_diff_ranges.append(float(abs_max))

    # Tracked layers: normalize depths and ages
    tracked_depth_norm = np.zeros_like(tracked_depths)
    tracked_age_norm = np.zeros_like(tracked_ages)
    max_age = max(float(tracked_ages.max()), 1.0)
    for s in range(config.max_tracked_layers):
        tracked_depth_norm[:, s] = 1.0 - normalize(tracked_depths[:, s], 0, config.max_depth_m)
        tracked_age_norm[:, s] = tracked_ages[:, s] / max_age

    # Build JSON
    data_38 = {
        # Time
        'time_seconds': times.tolist(),
        'hour_norm': hour_norm.tolist(),

        # Core features (shorter smoothing)
        'depth_norm': depth_norm.tolist(),
        'intensity_norm': intensity_norm.tolist(),
        'spread_norm': spread_norm.tolist(),
        'layers_norm': layers_norm.tolist(),

        # Dynamics (shorter smoothing)
        'velocity_norm': velocity_norm.tolist(),
        'velocity_stable_norm': velocity_stable_norm.tolist(),
        'acceleration_norm': acceleration_norm.tolist(),
        'anomaly_norm': anomaly_norm.tolist(),
        'spread_change_norm': spread_change_norm.tolist(),
        'onset_strength_norm': onset_norm.tolist(),
        'outlier_norm': outlier_norm.tolist(),

        # Texture (per-ping, no smoothing change)
        'acoustic_entropy': entropy.tolist(),
        'inter_ping_correlation': ipc.tolist(),
        'layer_separation_norm': layer_sep_norm.tolist(),

        # Shape (per-ping)
        'skewness_norm': skewness_norm.tolist(),
        'kurtosis_norm': kurtosis_norm.tolist(),
        'peak_prominence_norm': peak_prom_norm.tolist(),
        'peak_width_norm': peak_width_norm.tolist(),

        # Raw values for display
        'depth_m': depth_smooth.tolist(),
        'velocity_m_h': velocity.tolist(),
        'intensity_db': intensity.tolist(),
        'hour_of_day': hour.tolist(),

        # 8-band oceanographic histogram (static)
        'histogram_8band_0': hist_8band[:, 0].tolist(),
        'histogram_8band_1': hist_8band[:, 1].tolist(),
        'histogram_8band_2': hist_8band[:, 2].tolist(),
        'histogram_8band_3': hist_8band[:, 3].tolist(),
        'histogram_8band_4': hist_8band[:, 4].tolist(),
        'histogram_8band_5': hist_8band[:, 5].tolist(),
        'histogram_8band_6': hist_8band[:, 6].tolist(),
        'histogram_8band_7': hist_8band[:, 7].tolist(),

        # 8-band histogram differential (0.5=no change, >0.5=gaining, <0.5=losing)
        'histogram_diff_0': hist_diff_norm[:, 0].tolist(),
        'histogram_diff_1': hist_diff_norm[:, 1].tolist(),
        'histogram_diff_2': hist_diff_norm[:, 2].tolist(),
        'histogram_diff_3': hist_diff_norm[:, 3].tolist(),
        'histogram_diff_4': hist_diff_norm[:, 4].tolist(),
        'histogram_diff_5': hist_diff_norm[:, 5].tolist(),
        'histogram_diff_6': hist_diff_norm[:, 6].tolist(),
        'histogram_diff_7': hist_diff_norm[:, 7].tolist(),

        # Layer events
        'layer_event_type': layer_event_norm.tolist(),
        'layer_event_depth_norm': layer_event_depth_norm.tolist(),

        # Regime changepoints
        'regime_change_score': regime_change_norm.tolist(),
        'regime_id_norm': regime_id_norm.tolist(),

        # Tracked layer trajectories (4 slots)
        'tracked_layer_0_depth_norm': tracked_depth_norm[:, 0].tolist(),
        'tracked_layer_0_age_norm': tracked_age_norm[:, 0].tolist(),
        'tracked_layer_1_depth_norm': tracked_depth_norm[:, 1].tolist(),
        'tracked_layer_1_age_norm': tracked_age_norm[:, 1].tolist(),
        'tracked_layer_2_depth_norm': tracked_depth_norm[:, 2].tolist(),
        'tracked_layer_2_age_norm': tracked_age_norm[:, 2].tolist(),
        'tracked_layer_3_depth_norm': tracked_depth_norm[:, 3].tolist(),
        'tracked_layer_3_age_norm': tracked_age_norm[:, 3].tolist(),
    }

    sc_data = {
        'info': {
            'version': 'v6',
            'description': '38 kHz sonification features (v6: sonification-optimized smoothing, event features)',
            'mode': 'sonification',
            'smoothing_windows': {
                'depth': config.son_depth_smooth_window,
                'velocity': config.son_velocity_smooth_window,
                'acceleration': config.son_acceleration_smooth_window,
                'onset': config.son_onset_smooth_window,
                'spread_change': config.son_spread_change_window,
                'autocorr': config.son_autocorr_smooth_window,
            },
            'sample_rate_hz': float(1.0 / np.median(np.diff(times))) if len(times) > 1 else 0.3,
            'duration_seconds': float(times[-1]) if len(times) > 0 else 0,
            'num_points': n_pings,
            'n_regimes': n_regimes,
            'max_tracked_layer_age': float(max_age),
            'normalization': {
                'depth':           '0=deep (800m), 1=shallow (100m)',
                'intensity':       '0=quiet (p2), 1=loud (p98)',
                'velocity':        '0=descending fast, 0.5=stationary, 1=ascending fast',
                'anomaly':         '0=deeper than expected, 0.5=normal, 1=shallower',
                'entropy':         '0=tonal (concentrated), 1=noise-like (diffuse)',
                'ipc':             '0=turbulent/changing, 1=stable/sustained',
                'onset':           '0=gradual, 1=sudden biomass change',
                'outlier':         '0=normal, 1=extreme anomaly (z>=4)',
                'histogram_8band': '8 oceanographic zones (10-50, 50-150, 150-300, 300-500, 500-700, 700-850, 850-1000m, broadband)',
                'histogram_diff':  '0=losing energy, 0.5=stable, 1=gaining energy (8 log-spaced bands)',
                'layer_event':     '0=death, 0.5=none, 1=birth',
                'regime_change':   '0=stable, 1=maximum transition',
                'regime_id':       '0=first regime, 1=last regime',
                'tracked_depth':   '0=deep, 1=shallow (per slot, 0 if empty)',
                'tracked_age':     '0=just born/empty, 1=maximum persistence',
            },
            'normalization_ranges': {
                'intensity_db': [intensity_min, intensity_max],
                'spread_m': [spread_min, spread_max],
                'velocity_m_h': [velocity_min, velocity_max],
                'velocity_stable_m_h': [float(vel_stable_min), float(vel_stable_max)],
                'accel_m_h2': [accel_min, accel_max],
                'anomaly_m': [anomaly_min, anomaly_max],
                'layer_sep_m': [0.0, float(sep_max)],
                'onset': [0.0, float(onset_max)],
                'outlier_score': [0.0, 4.0],
                'skewness': [float(skew_min), float(skew_max)],
                'kurtosis': [float(kurt_min), float(kurt_max)],
                'peak_prominence_db': [float(pprom_min), float(pprom_max)],
                'peak_width_m': [float(pwidth_min), float(pwidth_max)],
                'histogram_diff_abs_max': hist_diff_ranges,
            },
        },
        '38kHz': data_38,
    }

    return sc_data


def create_supercollider_format_v7(features, config,
                                    hist_local, hist_global,
                                    diff_short, diff_medium,
                                    regime_score, regime_id,
                                    event_types, event_depths,
                                    tracked_depths, tracked_ages,
                                    dvm_depth_smooth=None,
                                    dvm_meta=None):
    """Create SuperCollider v7 format: fixed 8-band zones + dual normalization + multi-scale diff.

    v7 changes from v6:
    - 8-band histogram computed directly from raw Sv (no 32-band intermediate bias)
    - Zone 7 = DVM corridor (280-580m) instead of broadband
    - Both local (per-band) and global normalization
    - Differentials computed from same zone definitions at two timescales
    - Unsmoothed onset_peak for trigger detection
    - DVM depth: corridor CoM (50-600m) with auto-detected depth ranges
    - Auto-detected sunrise/sunset from velocity peaks (stored in metadata)
    """
    def normalize(arr, min_val, max_val):
        return np.clip((arr - min_val) / (max_val - min_val + 1e-10), 0, 1)

    n_pings = len(features)

    # Extract arrays
    times = np.array([f['timestamp_seconds'] for f in features])
    depth_smooth = np.array([f['depth_smooth_m'] for f in features])

    # DVM depth: use the time-averaged profile-based computation if provided,
    # otherwise fall back to per-ping peak tracking from add_derived_features
    if dvm_depth_smooth is None:
        dvm_depth_smooth = np.array([f['dvm_depth_smooth_m'] for f in features])
    dvm_velocity = np.array([f['dvm_velocity_m_h'] for f in features])
    # Recompute DVM velocity from the (potentially new) dvm_depth_smooth
    dvm_velocity = -np.gradient(dvm_depth_smooth, times) * 3600
    vel_window = 30  # ~1.5 min
    dvm_velocity = uniform_filter1d(dvm_velocity, vel_window, mode='nearest')

    intensity = np.array([f['total_intensity_db'] for f in features])
    spread = np.array([f['vertical_spread_m'] for f in features])
    layers = np.array([f['layer_count'] for f in features], dtype=float)
    velocity = np.array([f['velocity_m_h'] for f in features])
    velocity_stable = np.array([f['velocity_stable_m_h'] for f in features])
    acceleration = np.array([f['acceleration_m_h2'] for f in features])
    anomaly = np.array([f['depth_anomaly_m'] for f in features])
    spread_change = np.array([f['spread_change_m_min'] for f in features])
    hour = np.array([f['hour_of_day'] for f in features])
    ipc = np.array([f['inter_ping_correlation'] for f in features])
    entropy = np.array([f['acoustic_entropy'] for f in features])
    layer_sep = np.array([f['layer_separation_m'] for f in features])
    onset = np.array([f['onset_strength'] for f in features])
    onset_peak_arr = np.array([f['onset_peak'] for f in features])
    outlier_arr = np.array([f['outlier_score'] for f in features])
    skewness_arr = np.array([f['dist_skewness'] for f in features])
    kurtosis_arr = np.array([f['dist_kurtosis'] for f in features])
    peak_prom_arr = np.array([f['peak_max_prominence_db'] for f in features])
    peak_width_arr = np.array([f['peak_mean_width_m'] for f in features])

    # Handle NaN
    depth_smooth = np.nan_to_num(depth_smooth, nan=500)
    dvm_depth_smooth = np.nan_to_num(dvm_depth_smooth, nan=400)

    # Normalization ranges (2nd-98th percentile)
    intensity_min, intensity_max = _percentile_range(intensity)
    spread_min, spread_max = _percentile_range(spread)
    velocity_min, velocity_max = _percentile_range(velocity)
    vel_stable_min, vel_stable_max = _percentile_range(velocity_stable)
    accel_min, accel_max = _percentile_range(acceleration)
    spread_chg_min, spread_chg_max = _percentile_range(spread_change)
    anom_abs = float(np.nanpercentile(np.abs(anomaly), 98)) if np.any(~np.isnan(anomaly)) else 200.0
    anomaly_min, anomaly_max = -max(anom_abs, 1.0), max(anom_abs, 1.0)
    _, sep_max = _percentile_range(layer_sep)
    _, onset_max = _percentile_range(onset)
    _, onset_peak_max = _percentile_range(onset_peak_arr)
    skew_min, skew_max = _percentile_range(skewness_arr)
    kurt_min, kurt_max = _percentile_range(kurtosis_arr)
    pprom_min, pprom_max = _percentile_range(peak_prom_arr)
    pwidth_min, pwidth_max = _percentile_range(peak_width_arr)
    dvm_vel_min, dvm_vel_max = _percentile_range(dvm_velocity)

    # Normalize core features
    depth_norm = 1.0 - normalize(depth_smooth, 100, 800)
    intensity_norm = normalize(intensity, intensity_min, intensity_max)
    spread_norm = normalize(spread, spread_min, spread_max)
    layers_norm = normalize(layers, 0, 8)
    velocity_norm = normalize(velocity, velocity_min, velocity_max)
    velocity_stable_norm = normalize(velocity_stable, vel_stable_min, vel_stable_max)
    acceleration_norm = normalize(acceleration, accel_min, accel_max)
    anomaly_norm = 1.0 - normalize(anomaly, anomaly_min, anomaly_max)
    spread_change_norm = normalize(spread_change, spread_chg_min, spread_chg_max)
    hour_norm = hour / 24.0
    layer_sep_norm = normalize(layer_sep, 0.0, max(sep_max, 10.0))
    onset_norm = normalize(onset, 0.0, max(onset_max, 1e-10))
    onset_peak_norm = normalize(onset_peak_arr, 0.0, max(onset_peak_max, 1e-10))
    outlier_norm = normalize(outlier_arr, 0.0, 4.0)
    skewness_norm = normalize(skewness_arr, skew_min, skew_max)
    kurtosis_norm = normalize(kurtosis_arr, kurt_min, kurt_max)
    peak_prom_norm = normalize(peak_prom_arr, pprom_min, pprom_max)
    peak_width_norm = normalize(peak_width_arr, pwidth_min, pwidth_max)

    # DVM depth: inverted (shallow=1, deep=0).
    # Use auto-detected anchors if available, otherwise fixed range.
    if dvm_meta is not None:
        dvm_norm_lo = dvm_meta['shallow_anchor_m']
        dvm_norm_hi = dvm_meta['deep_anchor_m']
    else:
        dvm_norm_lo, dvm_norm_hi = 100, 800
    dvm_depth_norm = 1.0 - normalize(dvm_depth_smooth, dvm_norm_lo, dvm_norm_hi)
    # DVM velocity: 0=descending fast, 0.5=stationary, 1=ascending fast
    dvm_velocity_norm = normalize(dvm_velocity, dvm_vel_min, dvm_vel_max)

    # Normalize event features
    layer_event_norm = (event_types.astype(float) + 1.0) / 2.0
    layer_event_depth_norm = normalize(event_depths, 0, config.max_depth_m)

    # Regime change score: already [0,1]
    regime_change_norm = regime_score.copy()
    n_regimes = max(int(regime_id.max()), 1)
    regime_id_norm = regime_id.astype(float) / n_regimes

    # Tracked layers: normalize depths and ages
    tracked_depth_norm = np.zeros_like(tracked_depths)
    tracked_age_norm = np.zeros_like(tracked_ages)
    max_age = max(float(tracked_ages.max()), 1.0)
    for s in range(config.max_tracked_layers):
        tracked_depth_norm[:, s] = 1.0 - normalize(tracked_depths[:, s], 0, config.max_depth_m)
        tracked_age_norm[:, s] = tracked_ages[:, s] / max_age

    # Zone labels for documentation
    dvm_lo, dvm_hi = config.dvm_corridor
    zone_desc = (f'8 zones: 10-50, 50-150, 150-300, 300-500, 500-700, '
                 f'700-850, 850-1000m, DVM({dvm_lo}-{dvm_hi}m)')

    # Build JSON
    data_38 = {
        # Time
        'time_seconds': times.tolist(),
        'hour_norm': hour_norm.tolist(),

        # Core features (shorter smoothing)
        'depth_norm': depth_norm.tolist(),
        'intensity_norm': intensity_norm.tolist(),
        'spread_norm': spread_norm.tolist(),
        'layers_norm': layers_norm.tolist(),

        # Dynamics (shorter smoothing)
        'velocity_norm': velocity_norm.tolist(),
        'velocity_stable_norm': velocity_stable_norm.tolist(),
        'acceleration_norm': acceleration_norm.tolist(),
        'anomaly_norm': anomaly_norm.tolist(),
        'spread_change_norm': spread_change_norm.tolist(),
        'onset_strength_norm': onset_norm.tolist(),
        'onset_peak_norm': onset_peak_norm.tolist(),
        'outlier_norm': outlier_norm.tolist(),

        # Texture (per-ping, no smoothing change)
        'acoustic_entropy': entropy.tolist(),
        'inter_ping_correlation': ipc.tolist(),
        'layer_separation_norm': layer_sep_norm.tolist(),

        # Shape (per-ping)
        'skewness_norm': skewness_norm.tolist(),
        'kurtosis_norm': kurtosis_norm.tolist(),
        'peak_prominence_norm': peak_prom_norm.tolist(),
        'peak_width_norm': peak_width_norm.tolist(),

        # DVM peak depth (tracks dominant scattering layer, not center-of-mass)
        'dvm_depth_norm': dvm_depth_norm.tolist(),
        'dvm_velocity_norm': dvm_velocity_norm.tolist(),

        # Raw values for display
        'depth_m': depth_smooth.tolist(),
        'dvm_depth_m': dvm_depth_smooth.tolist(),
        'dvm_velocity_m_h': dvm_velocity.tolist(),
        'velocity_m_h': velocity.tolist(),
        'intensity_db': intensity.tolist(),
        'hour_of_day': hour.tolist(),

        # v7: 8-band local-normalized histogram (per-band p2-p98, for SC filter bank)
        'histogram_8band_local_0': hist_local[:, 0].tolist(),
        'histogram_8band_local_1': hist_local[:, 1].tolist(),
        'histogram_8band_local_2': hist_local[:, 2].tolist(),
        'histogram_8band_local_3': hist_local[:, 3].tolist(),
        'histogram_8band_local_4': hist_local[:, 4].tolist(),
        'histogram_8band_local_5': hist_local[:, 5].tolist(),
        'histogram_8band_local_6': hist_local[:, 6].tolist(),
        'histogram_8band_local_7': hist_local[:, 7].tolist(),

        # v7: 8-band global-normalized histogram (single range, preserves inter-zone ratios)
        'histogram_8band_global_0': hist_global[:, 0].tolist(),
        'histogram_8band_global_1': hist_global[:, 1].tolist(),
        'histogram_8band_global_2': hist_global[:, 2].tolist(),
        'histogram_8band_global_3': hist_global[:, 3].tolist(),
        'histogram_8band_global_4': hist_global[:, 4].tolist(),
        'histogram_8band_global_5': hist_global[:, 5].tolist(),
        'histogram_8band_global_6': hist_global[:, 6].tolist(),
        'histogram_8band_global_7': hist_global[:, 7].tolist(),

        # v7: Short-term differential (~1.5 min; 0.5=stable, >0.5=gaining, <0.5=losing)
        'histogram_diff_short_0': diff_short[:, 0].tolist(),
        'histogram_diff_short_1': diff_short[:, 1].tolist(),
        'histogram_diff_short_2': diff_short[:, 2].tolist(),
        'histogram_diff_short_3': diff_short[:, 3].tolist(),
        'histogram_diff_short_4': diff_short[:, 4].tolist(),
        'histogram_diff_short_5': diff_short[:, 5].tolist(),
        'histogram_diff_short_6': diff_short[:, 6].tolist(),
        'histogram_diff_short_7': diff_short[:, 7].tolist(),

        # v7: Medium-term differential (~15 min; 0.5=stable, >0.5=gaining, <0.5=losing)
        'histogram_diff_medium_0': diff_medium[:, 0].tolist(),
        'histogram_diff_medium_1': diff_medium[:, 1].tolist(),
        'histogram_diff_medium_2': diff_medium[:, 2].tolist(),
        'histogram_diff_medium_3': diff_medium[:, 3].tolist(),
        'histogram_diff_medium_4': diff_medium[:, 4].tolist(),
        'histogram_diff_medium_5': diff_medium[:, 5].tolist(),
        'histogram_diff_medium_6': diff_medium[:, 6].tolist(),
        'histogram_diff_medium_7': diff_medium[:, 7].tolist(),

        # Layer events
        'layer_event_type': layer_event_norm.tolist(),
        'layer_event_depth_norm': layer_event_depth_norm.tolist(),

        # Regime changepoints
        'regime_change_score': regime_change_norm.tolist(),
        'regime_id_norm': regime_id_norm.tolist(),

        # Tracked layer trajectories (4 slots)
        'tracked_layer_0_depth_norm': tracked_depth_norm[:, 0].tolist(),
        'tracked_layer_0_age_norm': tracked_age_norm[:, 0].tolist(),
        'tracked_layer_1_depth_norm': tracked_depth_norm[:, 1].tolist(),
        'tracked_layer_1_age_norm': tracked_age_norm[:, 1].tolist(),
        'tracked_layer_2_depth_norm': tracked_depth_norm[:, 2].tolist(),
        'tracked_layer_2_age_norm': tracked_age_norm[:, 2].tolist(),
        'tracked_layer_3_depth_norm': tracked_depth_norm[:, 3].tolist(),
        'tracked_layer_3_age_norm': tracked_age_norm[:, 3].tolist(),
    }

    # Build auto-detected metadata for SC sketch
    dvm_info = {}
    if dvm_meta is not None:
        dvm_info = {
            'dvm_algorithm': 'corridor_com',
            'dvm_corridor_bounds_m': list(dvm_meta['corridor_m']),
            'dvm_night_depth_m': dvm_meta['night_depth_m'],
            'dvm_day_depth_m': dvm_meta['day_depth_m'],
            'dvm_shallow_anchor_m': dvm_meta['shallow_anchor_m'],
            'dvm_deep_anchor_m': dvm_meta['deep_anchor_m'],
            'sunrise_h': dvm_meta['sunrise_h'],
            'sunset_h': dvm_meta['sunset_h'],
            'dawn_end_h': dvm_meta['dawn_end_h'],
            'dusk_start_h': dvm_meta['dusk_start_h'],
        }

    sc_data = {
        'info': {
            'version': 'v8',
            'description': '38 kHz sonification features (v8: corridor CoM DVM, auto-detected depth/timing)',
            'mode': 'sonification',
            'smoothing_windows': {
                'depth': config.son_depth_smooth_window,
                'velocity': config.son_velocity_smooth_window,
                'acceleration': config.son_acceleration_smooth_window,
                'onset': config.son_onset_smooth_window,
                'spread_change': config.son_spread_change_window,
                'diff_short': config.son_diff_short_window,
                'diff_medium': config.son_diff_medium_window,
            },
            'zone_edges_m': list(config.zone_edges),
            'dvm_corridor_m': list(config.dvm_corridor),
            **dvm_info,
            'sample_rate_hz': float(1.0 / np.median(np.diff(times))) if len(times) > 1 else 0.3,
            'duration_seconds': float(times[-1]) if len(times) > 0 else 0,
            'num_points': n_pings,
            'n_regimes': n_regimes,
            'max_tracked_layer_age': float(max_age),
            'normalization': {
                'depth':              '0=deep (800m), 1=shallow (100m) — center-of-mass of all backscatter',
                'dvm_depth':          f'0=deep ({dvm_norm_hi:.0f}m), 1=shallow ({dvm_norm_lo:.0f}m) — corridor CoM (50-600m, excludes DSL)',
                'dvm_velocity':       '0=descending fast, 0.5=stationary, 1=ascending fast (DVM layer)',
                'intensity':          '0=quiet (p2), 1=loud (p98)',
                'velocity':           '0=descending fast, 0.5=stationary, 1=ascending fast',
                'anomaly':            '0=deeper than expected, 0.5=normal, 1=shallower',
                'entropy':            '0=tonal (concentrated), 1=noise-like (diffuse)',
                'ipc':                '0=turbulent/changing, 1=stable/sustained',
                'onset':              '0=gradual, 1=sudden biomass change',
                'onset_peak':         '0=gradual, 1=sudden (unsmoothed, for triggers)',
                'outlier':            '0=normal, 1=extreme anomaly (z>=4)',
                'histogram_local':    f'Per-band [0,1] normalized. {zone_desc}',
                'histogram_global':   f'Global [0,1] normalized (preserves inter-zone ratios). {zone_desc}',
                'diff_short':         '0=losing energy, 0.5=stable, 1=gaining (~1.5 min window)',
                'diff_medium':        '0=losing energy, 0.5=stable, 1=gaining (~15 min window)',
                'layer_event':        '0=death, 0.5=none, 1=birth',
                'regime_change':      '0=stable, 1=maximum transition',
                'regime_id':          '0=first regime, 1=last regime',
                'tracked_depth':      '0=deep, 1=shallow (per slot, 0 if empty)',
                'tracked_age':        '0=just born/empty, 1=maximum persistence',
            },
            'normalization_ranges': {
                'intensity_db': [intensity_min, intensity_max],
                'spread_m': [spread_min, spread_max],
                'velocity_m_h': [velocity_min, velocity_max],
                'velocity_stable_m_h': [float(vel_stable_min), float(vel_stable_max)],
                'dvm_velocity_m_h': [float(dvm_vel_min), float(dvm_vel_max)],
                'accel_m_h2': [accel_min, accel_max],
                'anomaly_m': [anomaly_min, anomaly_max],
                'layer_sep_m': [0.0, float(sep_max)],
                'onset': [0.0, float(onset_max)],
                'onset_peak': [0.0, float(onset_peak_max)],
                'outlier_score': [0.0, 4.0],
                'skewness': [float(skew_min), float(skew_max)],
                'kurtosis': [float(kurt_min), float(kurt_max)],
                'peak_prominence_db': [float(pprom_min), float(pprom_max)],
                'peak_width_m': [float(pwidth_min), float(pwidth_max)],
            },
        },
        '38kHz': data_38,
    }

    return sc_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract sonification features from cleaned 24h data."
    )
    parser.add_argument(
        'date', nargs='?', default='20110126',
        help='Date string in YYYYMMDD format (default: 20110126)'
    )
    args = parser.parse_args()

    ensure_dirs()
    date_str = args.date

    config = SonificationConfigV3()
    features, sc_data = load_and_extract(date_str, config)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    # Print summary statistics
    velocities = [f['velocity_m_h'] for f in features]
    depths = [f['depth_smooth_m'] for f in features]

    print(f"Date: {date_str}")
    print(f"Duration: {features[-1]['timestamp_seconds'] / 3600:.1f} hours")
    print(f"Points: {len(features)}")
    print(f"Depth range: {np.min(depths):.0f} - {np.max(depths):.0f} m")
    print(f"Velocity range: {np.min(velocities):.1f} - {np.max(velocities):.1f} m/h")
