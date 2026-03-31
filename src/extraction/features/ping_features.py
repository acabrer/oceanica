"""
Per-ping feature extraction from single Sv profiles.

Extracts: center-of-mass, intensity, peaks, entropy, histogram (32-band),
vertical gradient, distribution shape metrics.
"""

import numpy as np
from scipy.signal import find_peaks

from .config import SonificationConfigV3


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
