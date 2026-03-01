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
  sonification_sc_v4_{date}.json — prenormalized arrays (0.0-1.0) for each
                                    ping (~3s resolution), ready for SC import.

Features extracted per ping:
  - center_of_mass_m: biomass-weighted mean depth
  - total_intensity_db: integrated acoustic backscatter (biomass proxy; linear mean)
  - peak_depth_m: depth of strongest return
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
from scipy.ndimage import uniform_filter1d
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
    # Smoothing windows (in number of pings, ~3s each)
    velocity_smooth_window: int = 600  # ~30 minutes
    acceleration_smooth_window: int = 300  # ~15 minutes


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
            'vertical_spread_m': 0.0,
            'acoustic_entropy': 0.0,
            'layer_count': 0,
            'layer_separation_m': 0.0,
            'surface_intensity_db': -90.0,
            'midwater_intensity_db': -90.0,
            'deep_intensity_db': -90.0
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

    # Layer detection (returns tuple)
    layer_count, layer_peaks, layer_separation = detect_layers(sv_column, depth_values, config)

    # Depth-stratified intensities
    surface_intensity = compute_layer_intensity(sv_column, depth_values, 10, 150, config.min_sv_threshold_db)
    midwater_intensity = compute_layer_intensity(sv_column, depth_values, 150, 400, config.min_sv_threshold_db)
    deep_intensity = compute_layer_intensity(sv_column, depth_values, 400, 800, config.min_sv_threshold_db)

    return {
        'center_of_mass_m': center_of_mass,
        'total_intensity_db': total_intensity,
        'peak_depth_m': peak_depth,
        'vertical_spread_m': vertical_spread,
        'acoustic_entropy': entropy_norm,
        'layer_count': layer_count,
        'layer_separation_m': layer_separation,
        'surface_intensity_db': surface_intensity,
        'midwater_intensity_db': midwater_intensity,
        'deep_intensity_db': deep_intensity
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
    (layer_count, peak_depths_m, layer_separation_m)
        layer_count       : number of distinct peaks
        peak_depths_m     : list of peak centre depths (m)
        layer_separation_m: depth gap between shallowest and deepest peak (0 if <2 peaks)
    """
    depth_bins = np.arange(config.surface_exclusion_m, config.max_depth_m, config.depth_bin_m)
    binned_sv = np.full(len(depth_bins) - 1, -90.0)

    for i in range(len(depth_bins) - 1):
        mask = (depth_values >= depth_bins[i]) & (depth_values < depth_bins[i+1])
        if np.any(mask) and np.any(~np.isnan(sv_column[mask])):
            binned_sv[i] = np.nanmean(sv_column[mask])

    peaks, _ = find_peaks(
        binned_sv,
        height=config.layer_detection_threshold_db,
        prominence=config.layer_prominence_db,
        width=config.layer_width_bins
    )

    count = len(peaks)
    peak_depths_m = (depth_bins[peaks] + config.depth_bin_m / 2).tolist() if count > 0 else []
    separation_m = float(peak_depths_m[-1] - peak_depths_m[0]) if count >= 2 else 0.0

    return count, peak_depths_m, separation_m


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


def add_derived_features(features, config):
    """Add velocity, acceleration, and anomaly features after collecting all data."""
    if len(features) < 10:
        return features

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
    depth_smooth = uniform_filter1d(depths_clean, config.velocity_smooth_window, mode='nearest')

    # Calculate velocity (m/hour) - positive = ascending (toward surface)
    dt = np.gradient(times)
    dt[dt == 0] = 1  # Avoid division by zero
    velocity = -np.gradient(depth_smooth, times) * 3600  # Negative because increasing depth = descending

    # Smooth velocity
    velocity_smooth = uniform_filter1d(velocity, config.velocity_smooth_window // 2, mode='nearest')

    # Calculate acceleration (m/h²)
    acceleration = np.gradient(velocity_smooth, times) * 3600
    acceleration_smooth = uniform_filter1d(acceleration, config.acceleration_smooth_window, mode='nearest')

    # Calculate depth anomaly (deviation from expected for time of day)
    expected_depths = np.array([compute_expected_depth_for_time(h) for h in hours])
    depth_anomaly = depths_clean - expected_depths  # Positive = deeper than expected

    # Rate of change of spread (layer dynamics)
    spreads = np.array([f['vertical_spread_m'] for f in features])
    spreads_clean = np.nan_to_num(spreads, nan=200)
    spread_change = np.gradient(uniform_filter1d(spreads_clean, 100, mode='nearest'), times) * 60  # m/min

    # Onset strength: smoothed abs first-difference of total intensity (in linear space)
    intensity_arr = np.array([f['total_intensity_db'] for f in features])
    intensity_linear = 10 ** (intensity_arr / 10)
    onset_raw = np.abs(np.gradient(intensity_linear, times))
    onset_smooth = uniform_filter1d(onset_raw, size=60, mode='nearest')  # ~3-min smoothing

    # Add to features
    for i, f in enumerate(features):
        f['velocity_m_h'] = float(velocity_smooth[i])
        f['acceleration_m_h2'] = float(acceleration_smooth[i])
        f['depth_anomaly_m'] = float(depth_anomaly[i])
        f['spread_change_m_min'] = float(spread_change[i])
        f['depth_smooth_m'] = float(depth_smooth[i])
        f['onset_strength'] = float(onset_smooth[i])

    return features


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

    # Add derived features (velocity, acceleration, anomaly)
    print("\nComputing derived features (velocity, acceleration, anomaly)...")
    all_features = add_derived_features(all_features, config)

    # Create SuperCollider format
    print("\nCreating SuperCollider JSON...")
    sc_data = create_supercollider_format_v3(all_features, config)
    sc_path = OUTPUT_DATA / f"sonification_sc_v4_{date_str}.json"
    with open(sc_path, 'w') as f:
        json.dump(sc_data, f, indent=2)
    print(f"Saved: {sc_path}")

    return all_features, sc_data


def _percentile_range(arr, lo=2, hi=98):
    """Return (p_lo, p_hi) from actual data, ignoring NaN. Safe for collapsed ranges."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    if np.isclose(p_lo, p_hi):
        p_hi = p_lo + 1e-3
    return float(p_lo), float(p_hi)


def create_supercollider_format_v3(features, config):
    """Create SuperCollider v4 format with data-driven normalization and new features."""

    def normalize(arr, min_val, max_val):
        return np.clip((arr - min_val) / (max_val - min_val + 1e-10), 0, 1)

    # Extract arrays
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
    # New features
    ipc = np.array([f['inter_ping_correlation'] for f in features])
    entropy = np.array([f['acoustic_entropy'] for f in features])
    layer_sep = np.array([f['layer_separation_m'] for f in features])
    onset = np.array([f['onset_strength'] for f in features])

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

    sc_data = {
        'info': {
            'version': 'v4',
            'description': '38 kHz sonification features from cleaned data (v4: scientific fixes + new features)',
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
                'onset':       '0=gradual, 1=sudden biomass change (data p98)'
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
            'hour_of_day': hour.tolist()
        }
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
