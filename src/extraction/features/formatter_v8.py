"""
SuperCollider v8 JSON formatter.

Builds the normalized JSON structure consumed by SC sketches.
v8: corridor CoM DVM, auto-detected depth/timing, fixed 8-band zones,
dual normalization, multi-scale differentials.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import SonificationConfigV3
from .normalization import normalize, percentile_range


def create_sc_format_v8(features, config,
                        hist_local, hist_global,
                        diff_short, diff_medium,
                        regime_score, regime_id,
                        event_types, event_depths,
                        tracked_depths, tracked_ages,
                        dvm_depth_smooth=None,
                        dvm_meta=None):
    """Create SuperCollider v8 JSON format.

    v8: fixed 8-band zones + dual normalization + multi-scale diff +
    corridor CoM DVM + auto-detected sunrise/sunset.
    """
    n_pings = len(features)

    # Extract arrays
    times = np.array([f['timestamp_seconds'] for f in features])
    depth_smooth = np.array([f['depth_smooth_m'] for f in features])

    # DVM depth
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
    velocity_fine = np.array([f['velocity_fine_m_h'] for f in features])
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
    intensity_min, intensity_max = percentile_range(intensity)
    spread_min, spread_max = percentile_range(spread)
    velocity_min, velocity_max = percentile_range(velocity)
    vel_fine_min, vel_fine_max = percentile_range(velocity_fine)
    accel_min, accel_max = percentile_range(acceleration)
    spread_chg_min, spread_chg_max = percentile_range(spread_change)
    anom_abs = float(np.nanpercentile(np.abs(anomaly), 98)) if np.any(~np.isnan(anomaly)) else 200.0
    anomaly_min, anomaly_max = -max(anom_abs, 1.0), max(anom_abs, 1.0)
    _, sep_max = percentile_range(layer_sep)
    _, onset_max = percentile_range(onset)
    _, onset_peak_max = percentile_range(onset_peak_arr)
    skew_min, skew_max = percentile_range(skewness_arr)
    kurt_min, kurt_max = percentile_range(kurtosis_arr)
    pprom_min, pprom_max = percentile_range(peak_prom_arr)
    pwidth_min, pwidth_max = percentile_range(peak_width_arr)
    dvm_vel_min, dvm_vel_max = percentile_range(dvm_velocity)

    # Normalize core features
    depth_norm = 1.0 - normalize(depth_smooth, 100, 800)
    intensity_norm = normalize(intensity, intensity_min, intensity_max)
    spread_norm = normalize(spread, spread_min, spread_max)
    layers_norm = normalize(layers, 0, 8)
    velocity_norm = normalize(velocity, velocity_min, velocity_max)
    velocity_fine_norm = normalize(velocity_fine, vel_fine_min, vel_fine_max)
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

    # Build JSON data_38
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
        'velocity_fine_norm': velocity_fine_norm.tolist(),
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

        # v7: 8-band local-normalized histogram
        'histogram_8band_local_0': hist_local[:, 0].tolist(),
        'histogram_8band_local_1': hist_local[:, 1].tolist(),
        'histogram_8band_local_2': hist_local[:, 2].tolist(),
        'histogram_8band_local_3': hist_local[:, 3].tolist(),
        'histogram_8band_local_4': hist_local[:, 4].tolist(),
        'histogram_8band_local_5': hist_local[:, 5].tolist(),
        'histogram_8band_local_6': hist_local[:, 6].tolist(),
        'histogram_8band_local_7': hist_local[:, 7].tolist(),

        # v7: 8-band global-normalized histogram
        'histogram_8band_global_0': hist_global[:, 0].tolist(),
        'histogram_8band_global_1': hist_global[:, 1].tolist(),
        'histogram_8band_global_2': hist_global[:, 2].tolist(),
        'histogram_8band_global_3': hist_global[:, 3].tolist(),
        'histogram_8band_global_4': hist_global[:, 4].tolist(),
        'histogram_8band_global_5': hist_global[:, 5].tolist(),
        'histogram_8band_global_6': hist_global[:, 6].tolist(),
        'histogram_8band_global_7': hist_global[:, 7].tolist(),

        # v7: Short-term differential
        'histogram_diff_short_0': diff_short[:, 0].tolist(),
        'histogram_diff_short_1': diff_short[:, 1].tolist(),
        'histogram_diff_short_2': diff_short[:, 2].tolist(),
        'histogram_diff_short_3': diff_short[:, 3].tolist(),
        'histogram_diff_short_4': diff_short[:, 4].tolist(),
        'histogram_diff_short_5': diff_short[:, 5].tolist(),
        'histogram_diff_short_6': diff_short[:, 6].tolist(),
        'histogram_diff_short_7': diff_short[:, 7].tolist(),

        # v7: Medium-term differential
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
                'velocity_fine_m_h': [float(vel_fine_min), float(vel_fine_max)],
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
