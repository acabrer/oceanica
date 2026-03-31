"""
Histogram-based features: 8-band oceanographic zone energy.

Computes zone energy directly from raw Sv profiles, with both local and
global normalization, plus multi-scale temporal differentials.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import SonificationConfigV3


def compute_histogram_zones(sv_data, depth_values, config):
    """Compute 8-band oceanographic histogram directly from raw Sv data.

    v7: Computes zone energy directly from raw Sv profiles (bypasses 32-band
    intermediate), eliminates log-to-linear summation bias, replaces Zone 7
    (broadband) with DVM corridor, and provides both local and global
    normalization plus multi-scale differentials.

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


def compute_histogram_differential(features, config):
    """Compute 8-band histogram and differential from 32-band data (v6 legacy).

    Returns
    -------
    hist_8band : np.ndarray, shape (n_pings, 8)
    hist_diff : np.ndarray, shape (n_pings, 8)
    """
    n_pings = len(features)

    # --- Static 8-band: oceanographic zones ---
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
