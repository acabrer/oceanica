"""
Derived features computed over the full time series.

Velocity, acceleration, anomaly, onset strength, outlier score,
and DVM peak tracking — all computed after per-ping extraction.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter

from .config import SonificationConfigV3


def _sigmoid(x, k=8.0):
    """Logistic sigmoid mapped to [0, 1] for smooth DVM transitions."""
    return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))


def compute_expected_depth_for_time(hour_of_day):
    """
    Model of expected DVM depth calibrated for South Atlantic (~25 S), January 2011.

    Sunrise ~ 05:45 UTC, sunset ~ 20:15 UTC.
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

    Mutates each dict in `features` in-place (adds new keys).
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
    velocity = -np.gradient(depth_smooth, times) * 3600

    # Smooth velocity
    velocity_smooth = uniform_filter1d(velocity, vel_window, mode='nearest')

    # Stable-window velocity (5-min window for capturing rapid DVM events)
    depth_smooth_stable = uniform_filter1d(depths_clean, config.velocity_stable_smooth_window, mode='nearest')
    velocity_stable = -np.gradient(depth_smooth_stable, times) * 3600
    velocity_stable_smooth = uniform_filter1d(velocity_stable, config.velocity_stable_smooth_window // 2, mode='nearest')

    # Calculate acceleration (m/h^2)
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

    # DVM depth: track the dominant scattering layer (not center-of-mass)
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
