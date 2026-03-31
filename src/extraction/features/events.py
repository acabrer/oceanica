"""
Event-level and temporal-pattern features.

Lagged autocorrelation, regime changepoints, layer birth/death events,
and persistent layer identity tracking.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .config import SonificationConfigV3


def compute_lagged_autocorrelation(sv_data, time_seconds):
    """Compute autocorrelation at 3-min and 10-min lags using cosine similarity.

    Reveals cyclic patterns (feeding pulses, oscillatory migrations) that
    single-ping IPC misses.
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


def compute_regime_changepoints(features, config):
    """Derivative-based transition detection on composite acoustic signal.

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

    # Find major transition peaks -> regime boundaries
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
            event_depths[i] = births[0]
        elif len(deaths) > len(births):
            event_types[i] = -1
            event_depths[i] = deaths[0]

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

        # Build cost matrix
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
                    active[si][0] = curr_peaks[pi]
                    active[si][1] += 1
                    active[si][2] = 0

        # Unmatched active slots: increment unmatched count
        for si in range(max_layers):
            if si not in matched_slots and active[si][1] > 0:
                active[si][2] += 1
                if active[si][2] > 3:
                    active[si] = [0.0, 0, 99]

        # Unmatched new peaks: assign to empty slots
        for pi in range(len(curr_peaks)):
            if pi in matched_peaks:
                continue
            empty_slot = None
            for si in range(max_layers):
                if active[si][1] == 0 and active[si][2] >= 99:
                    empty_slot = si
                    break
            if empty_slot is None:
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
