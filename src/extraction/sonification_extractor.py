#!/usr/bin/env python3
"""
================================================================================
SONIFICATION FEATURE EXTRACTOR — Modular Orchestrator
================================================================================
Project: Sonificacio Oceanica
Team: Oceanica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya, Departament de Cultura

Reads cleaned 24h NetCDF data (output of run_24h_processing.py) and extracts
sonification features for SuperCollider, producing a v8 JSON file.

This is the modular version. The original monolith is preserved as
sonification_extractor_monolith.py (can still generate v5/v6 JSON if needed).

Feature modules (src/extraction/features/):
  config.py           — SonificationConfigV3 dataclass
  normalization.py    — percentile_range, normalize
  ping_features.py    — per-ping extraction (CoM, entropy, peaks, histogram)
  derived_features.py — velocity, anomaly, onset, outlier (time-series derived)
  histogram.py        — 8-band oceanographic zones + differentials
  dvm.py              — corridor center-of-mass DVM depth tracking
  events.py           — layer events, regime changepoints, layer tracking
  formatter_v8.py     — v8 JSON builder for SuperCollider

Usage:
    python src/extraction/sonification_extractor.py [YYYYMMDD]
    # default date: 20110126

Author: Oceanica (Alex Cabrer & Joan Sala)
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
import warnings
warnings.filterwarnings('ignore')

# Feature modules
from features.config import SonificationConfigV3
from features.ping_features import extract_ping_features
from features.derived_features import add_derived_features
from features.histogram import compute_histogram_zones
from features.dvm import compute_dvm_depth
from features.events import (
    compute_lagged_autocorrelation,
    compute_regime_changepoints,
    detect_layer_events,
    track_layers,
)
from features.formatter_v8 import create_sc_format_v8


def load_and_extract(date_str, config=None):
    """
    Load cleaned 24h data from NetCDF and extract sonification features.

    Produces a single v8 JSON file with all features needed by SC voices.

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
    print(f"  {n_pings} pings x {n_depth} depth samples")
    print(f"  Depth range: 0 - {depths[-1]:.0f} m")
    print(f"  Time range:  {times[0]} -> {times[-1]}")

    # Compute time in seconds from start of day
    start_time = times[0]
    time_seconds = (times - start_time) / np.timedelta64(1, 's')
    time_seconds = np.array(time_seconds, dtype=float)

    # Ensure depths are finite floats
    depth_values = np.array(depths, dtype=float)
    depth_values = np.nan_to_num(depth_values, nan=0)

    # ===================================================================
    # PHASE 1: Per-ping feature extraction
    # ===================================================================
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

    # Count NaN pings (impulse noise)
    nan_pings = sum(1 for f in all_features if np.isnan(f['center_of_mass_m']))
    print(f"  NaN pings (impulse noise): {nan_pings} ({100*nan_pings/len(all_features):.1f}%)")

    # ===================================================================
    # PHASE 2: Derived features (sonification-optimized smoothing)
    # ===================================================================
    print("\nComputing derived features (velocity, acceleration, anomaly, outlier)...")
    son_features = [dict(f) for f in all_features]
    son_features = add_derived_features(son_features, config, sonification_mode=True)

    # Multi-lag autocorrelation
    print("Computing multi-lag autocorrelation (3-min, 10-min)...")
    autocorr_3min, autocorr_10min = compute_lagged_autocorrelation(sv_data, time_seconds)
    for i, f in enumerate(son_features):
        f['autocorr_3min'] = float(autocorr_3min[i])
        f['autocorr_10min'] = float(autocorr_10min[i])

    # ===================================================================
    # PHASE 3: Specialized features
    # ===================================================================
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

    print("  Computing regime changepoints...")
    regime_score, regime_id = compute_regime_changepoints(son_features, config)

    print("  Detecting layer events (births/deaths)...")
    event_types, event_depths, _ = detect_layer_events(son_features, config)

    print("  Tracking persistent layers...")
    tracked_depths, tracked_ages = track_layers(son_features, config)

    # ===================================================================
    # PHASE 4: Build and write v8 JSON
    # ===================================================================
    print("\nCreating SuperCollider v8 JSON...")
    sc_data = create_sc_format_v8(
        son_features, config,
        hist_local, hist_global, diff_short, diff_medium,
        regime_score, regime_id, event_types, event_depths,
        tracked_depths, tracked_ages,
        dvm_depth_smooth=dvm_depth_smooth,
        dvm_meta=dvm_meta
    )
    sc_path = OUTPUT_DATA / f"sonification_sc_v8_{date_str}.json"
    with open(sc_path, 'w') as f:
        json.dump(sc_data, f, indent=2)
    print(f"Saved: {sc_path}")

    # ===================================================================
    # Verification statistics
    # ===================================================================
    print("\n  v8 Feature Statistics:")
    print(f"    Zone 7 = DVM corridor ({config.dvm_corridor[0]}-{config.dvm_corridor[1]}m)")
    for b in range(8):
        local_std = hist_local[:, b].std()
        global_std = hist_global[:, b].std()
        diff_s_std = diff_short[:, b].std()
        diff_m_std = diff_medium[:, b].std()
        print(f"    Band {b}: local_std={local_std:.3f}  global_std={global_std:.3f}  "
              f"diff_short_std={diff_s_std:.3f}  diff_medium_std={diff_m_std:.3f}")

    # DVM depth statistics
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
    print(f"    DVM range: {np.nanmin(dvm_depths):.0f} - {np.nanmax(dvm_depths):.0f}m")

    return son_features, sc_data


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
