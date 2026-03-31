"""
Normalization utilities for sonification feature extraction.

Shared by all feature modules and the v8 JSON formatter.
"""

import numpy as np


def percentile_range(arr, lo=2, hi=98):
    """Return (p_lo, p_hi) from actual data, ignoring NaN. Safe for collapsed ranges."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    if np.isclose(p_lo, p_hi):
        p_hi = p_lo + 1e-3
    return float(p_lo), float(p_hi)


def normalize(arr, min_val, max_val):
    """Normalize array to [0, 1] given explicit min/max bounds."""
    return np.clip((arr - min_val) / (max_val - min_val + 1e-10), 0, 1)
