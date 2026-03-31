"""
DVM (Diel Vertical Migration) depth tracking.

Uses corridor center-of-mass (50-600m) to track the migrating layer,
excluding surface noise and the persistent deep scattering layer.
Auto-detects night/day depth and sunrise/sunset from the signal.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter

from .config import SonificationConfigV3


def compute_dvm_depth(sv_data, depth_values, config, sonification_mode=False,
                      time_seconds=None):
    """Compute DVM depth using corridor center-of-mass.

    v9: Replaces the dual-zone log-ratio with corridor center-of-mass (CoM)
    restricted to 50-600 m. This excludes both surface noise (0-50 m) and
    the persistent deep scattering layer (DSL, >600 m), which at 38 kHz is
    always brighter than the migrating layer due to myctophid swim bladder
    resonance.

    Returns
    -------
    dvm_depth : np.ndarray, shape (n_pings,)
        Smoothed DVM depth in meters.
    dvm_meta : dict
        Auto-detected metadata (corridor bounds, night/day depth,
        sunrise/sunset hours, dawn_end, dusk_start).
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
    p15 = float(np.percentile(com_smooth, 15))
    p85 = float(np.percentile(com_smooth, 85))
    night_depth_m = float(np.median(com_smooth[com_smooth <= p15 + 0.3 * (p85 - p15)]))
    day_depth_m = float(np.median(com_smooth[com_smooth >= p85 - 0.3 * (p85 - p15)]))

    # --- Step 7: Percentile normalization -> physical depth ---
    phase = np.clip((com_smooth - p15) / (p85 - p15 + 1e-10), 0, 1)
    shallow_anchor = max(night_depth_m - 30, 100.0)
    deep_anchor = min(day_depth_m + 30, 650.0)
    dvm_depth = shallow_anchor + phase * (deep_anchor - shallow_anchor)

    # --- Step 8: Auto-detect sunrise/sunset from velocity ---
    sunrise_h, sunset_h = 5.75, 20.25  # defaults (South Atlantic Jan 2011)
    dawn_end_h, dusk_start_h = 8.5, 19.5
    if time_seconds is not None and len(time_seconds) == n_pings:
        hours = time_seconds / 3600.0
        # Velocity: negative gradient = descending (depth increasing)
        dvm_vel = np.gradient(com_smooth, time_seconds) * 3600  # m/h (positive=deepening)
        # Use heavy smoothing for sunrise/sunset detection
        detect_window = max(smooth_window * 3, 601)
        dvm_vel_smooth = uniform_filter1d(dvm_vel, size=detect_window, mode='nearest')

        # Dawn = strongest positive velocity (descent, depth increasing)
        # Dusk = strongest negative velocity (ascent, depth decreasing)
        dawn_mask = (hours >= 3) & (hours <= 12)
        dusk_mask = (hours >= 15) & (hours <= 23)

        if np.any(dawn_mask):
            dawn_peak_idx = np.where(dawn_mask)[0][np.argmax(dvm_vel_smooth[dawn_mask])]
            dawn_peak_h = float(hours[dawn_peak_idx])
            sunrise_h = round(max(dawn_peak_h - 1.0, 3.0), 2)
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
            dusk_start_h = round(max(dusk_peak_h - 1.0, 15.0), 2)
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
