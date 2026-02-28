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
  sonification_sc_v3_{date}.json — prenormalized arrays (0.0-1.0) for each
                                    ping (~3s resolution), ready for SC import.

Features extracted per ping:
  - center_of_mass_m: biomass-weighted mean depth
  - total_intensity_db: integrated acoustic backscatter (biomass proxy)
  - peak_depth_m: depth of strongest return
  - vertical_spread_m: standard deviation of depth distribution
  - layer_count: number of distinct scattering layers
  - surface/midwater/deep_intensity_db: depth-stratified backscatter
  - velocity_m_h: migration speed (derived, 30-min smoothed)
  - acceleration_m_h2: rate of velocity change (derived)
  - depth_anomaly_m: deviation from expected DVM depth (derived)
  - spread_change_m_min: rate of spread change (derived)

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
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SonificationConfigV3:
    """Configuration for v3 extraction."""
    surface_exclusion_m: float = 10.0
    max_depth_m: float = 1000.0
    min_sv_threshold_db: float = -80.0
    layer_detection_threshold_db: float = -70.0
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
            'layer_count': 0,
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
    else:
        center_of_mass = np.nan
        total_intensity = -90.0
        vertical_spread = 0.0

    peak_idx = np.argmax(valid_sv)
    peak_depth = valid_depth[peak_idx]

    # Layer detection
    layer_count = detect_layers(sv_column, depth_values, config)

    # Depth-stratified intensities
    surface_intensity = compute_layer_intensity(sv_column, depth_values, 10, 150, config.min_sv_threshold_db)
    midwater_intensity = compute_layer_intensity(sv_column, depth_values, 150, 400, config.min_sv_threshold_db)
    deep_intensity = compute_layer_intensity(sv_column, depth_values, 400, 800, config.min_sv_threshold_db)

    return {
        'center_of_mass_m': center_of_mass,
        'total_intensity_db': total_intensity,
        'peak_depth_m': peak_depth,
        'vertical_spread_m': vertical_spread,
        'layer_count': layer_count,
        'surface_intensity_db': surface_intensity,
        'midwater_intensity_db': midwater_intensity,
        'deep_intensity_db': deep_intensity
    }


def compute_layer_intensity(sv_column, depth_values, min_depth, max_depth, threshold_db):
    """Compute mean intensity within a depth layer."""
    mask = (
        (depth_values >= min_depth) &
        (depth_values <= max_depth) &
        (~np.isnan(sv_column)) &
        (sv_column > threshold_db)
    )
    if np.any(mask):
        return float(np.mean(sv_column[mask]))
    return -90.0


def detect_layers(sv_column, depth_values, config):
    """Detect distinct scattering layers."""
    depth_bins = np.arange(config.surface_exclusion_m, config.max_depth_m, config.depth_bin_m)
    binned_sv = []

    for i in range(len(depth_bins) - 1):
        mask = (depth_values >= depth_bins[i]) & (depth_values < depth_bins[i+1])
        if np.any(mask) and np.any(~np.isnan(sv_column[mask])):
            binned_sv.append(np.nanmean(sv_column[mask]))
        else:
            binned_sv.append(-90.0)

    binned_sv = np.array(binned_sv)
    above_threshold = binned_sv > config.layer_detection_threshold_db

    layer_count = 0
    in_layer = False
    for val in above_threshold:
        if val and not in_layer:
            layer_count += 1
            in_layer = True
        elif not val:
            in_layer = False

    return layer_count


def compute_expected_depth_for_time(hour_of_day):
    """
    Model of expected DVM depth based on time of day.
    Based on typical Atlantic DVM pattern.

    Night (21:00-05:00): shallow (~200m)
    Dawn descent (05:00-08:00): transitioning deep
    Day (08:00-17:00): deep (~500m)
    Dusk ascent (17:00-21:00): transitioning shallow
    """
    hour = hour_of_day % 24

    if hour < 5:  # Late night
        return 200
    elif hour < 8:  # Dawn descent
        progress = (hour - 5) / 3
        return 200 + progress * 300
    elif hour < 17:  # Day
        return 500
    elif hour < 21:  # Dusk ascent
        progress = (hour - 17) / 4
        return 500 - progress * 300
    else:  # Early night
        return 200


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

    # Add to features
    for i, f in enumerate(features):
        f['velocity_m_h'] = float(velocity_smooth[i])
        f['acceleration_m_h2'] = float(acceleration_smooth[i])
        f['depth_anomaly_m'] = float(depth_anomaly[i])
        f['spread_change_m_min'] = float(spread_change[i])
        f['depth_smooth_m'] = float(depth_smooth[i])

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

    for ping_idx in range(n_pings):
        if ping_idx % 5000 == 0:
            print(f"  [{ping_idx+1}/{n_pings}] ({100*ping_idx/n_pings:.0f}%)")

        sv_column = sv_data[ping_idx, :]

        # Extract per-ping features
        feats = extract_ping_features(sv_column, depth_values, config)

        # Calculate hour of day
        ping_dt = times[ping_idx].astype('datetime64[s]').astype(datetime)
        hour_of_day = ping_dt.hour + ping_dt.minute / 60

        feat_dict = {
            'timestamp': str(times[ping_idx]),
            'timestamp_seconds': float(time_seconds[ping_idx]),
            'hour_of_day': hour_of_day,
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
    sc_path = OUTPUT_DATA / f"sonification_sc_v3_{date_str}.json"
    with open(sc_path, 'w') as f:
        json.dump(sc_data, f, indent=2)
    print(f"Saved: {sc_path}")

    return all_features, sc_data


def create_supercollider_format_v3(features, config):
    """Create SuperCollider format with proper normalization."""

    def normalize(arr, min_val, max_val):
        return np.clip((arr - min_val) / (max_val - min_val + 1e-10), 0, 1)

    # Extract arrays
    times = np.array([f['timestamp_seconds'] for f in features])
    depth = np.array([f['center_of_mass_m'] for f in features])
    depth_smooth = np.array([f['depth_smooth_m'] for f in features])
    intensity = np.array([f['total_intensity_db'] for f in features])
    spread = np.array([f['vertical_spread_m'] for f in features])
    layers = np.array([f['layer_count'] for f in features])
    velocity = np.array([f['velocity_m_h'] for f in features])
    acceleration = np.array([f['acceleration_m_h2'] for f in features])
    anomaly = np.array([f['depth_anomaly_m'] for f in features])
    spread_change = np.array([f['spread_change_m_min'] for f in features])
    hour = np.array([f['hour_of_day'] for f in features])
    surface = np.array([f['surface_intensity_db'] for f in features])
    midwater = np.array([f['midwater_intensity_db'] for f in features])
    deep = np.array([f['deep_intensity_db'] for f in features])

    # Handle NaN
    depth = np.nan_to_num(depth, nan=500)
    depth_smooth = np.nan_to_num(depth_smooth, nan=500)

    # Normalize with actual meaningful ranges
    # Depth: inverted (shallow=1, deep=0)
    depth_norm = 1.0 - normalize(depth_smooth, 100, 800)

    # Intensity: -50 to -30 dB typical active range
    intensity_norm = normalize(intensity, -50, -30)

    # Spread: 100-350m typical
    spread_norm = normalize(spread, 100, 350)

    # Layers: 0-8 typical
    layers_norm = normalize(layers, 0, 8)

    # Velocity: -80 to +80 m/h (migration speed)
    # 0 = fast descending, 0.5 = stationary, 1 = fast ascending
    velocity_norm = normalize(velocity, -80, 80)

    # Acceleration: indicates rate of change
    acceleration_norm = normalize(acceleration, -500, 500)

    # Anomaly: -200 to +200m deviation from expected
    # 0 = much shallower than expected, 0.5 = normal, 1 = much deeper
    anomaly_norm = normalize(anomaly, -200, 200)

    # Spread change
    spread_change_norm = normalize(spread_change, -5, 5)

    # Hour of day normalized to 0-1 (for time-based effects)
    hour_norm = hour / 24.0

    # Depth zone intensities
    surface_norm = normalize(surface, -80, -40)
    midwater_norm = normalize(midwater, -80, -40)
    deep_norm = normalize(deep, -80, -40)

    sc_data = {
        'info': {
            'version': 'v3',
            'description': '38 kHz sonification features from cleaned data',
            'source': 'Cleaned NetCDF (noise-removed, impulse-masked)',
            'sample_rate_hz': 1.0 / np.median(np.diff(times)) if len(times) > 1 else 0.3,
            'duration_seconds': float(times[-1]) if len(times) > 0 else 0,
            'num_points': len(times),
            'normalization': {
                'depth': '0=deep (800m), 1=shallow (100m)',
                'intensity': '0=quiet (-50dB), 1=loud (-30dB)',
                'velocity': '0=descending fast, 0.5=stationary, 1=ascending fast',
                'anomaly': '0=shallower than expected, 0.5=normal, 1=deeper than expected',
                'hour': '0=midnight, 0.5=noon, 1=midnight'
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

            # Raw values for display
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
