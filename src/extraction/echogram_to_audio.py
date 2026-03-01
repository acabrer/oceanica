#!/usr/bin/env python3
"""
================================================================================
DIRECT ECHOGRAM → AUDIO MAPPING
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya, Departament de Cultura

Treats the echogram as an audio spectrogram and synthesizes a WAV file
via two alternative methods:

    Method: additive — sums N sinusoids with time-varying amplitude envelopes
    Method: ifft     — inverts echogram as magnitude spectrogram (ISTFT),
                       optionally refined with Griffin-Lim phase iteration

Both methods use the same scientific mapping:

    depth axis   → frequency axis  (shallow=high pitch, deep=low pitch)
    Sv intensity → amplitude       (strong backscatter=loud, weak=quiet)
    ping time    → audio time      (24 hours compressed to ~4 minutes)

Reads cleaned 24h NetCDF data (output of run_24h_processing.py).

Usage:
    python src/extraction/echogram_to_audio.py [YYYYMMDD]
    python src/extraction/echogram_to_audio.py 20110126 --method additive
    python src/extraction/echogram_to_audio.py 20110126 --method ifft
    python src/extraction/echogram_to_audio.py 20110126 --method ifft --ifft-iter 32

Author: Oceànica (Alex Cabrer & Joan Sala)
================================================================================
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DATA, ensure_dirs

import numpy as np
import xarray as xr
from scipy.ndimage import zoom, uniform_filter1d
from scipy.io import wavfile
from scipy.signal import stft, istft
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EchogramAudioConfig:
    """Configuration for direct echogram-to-audio mapping."""
    audio_duration_s: float = 240.0       # 24h → 4 minutes
    freq_min_hz: float = 50.0             # lowest frequency (deepest depth)
    freq_max_hz: float = 8000.0           # highest frequency (shallowest depth)
    sample_rate: int = 44100              # audio sample rate
    n_freq_bins: int = 256                # sinusoidal partials (additive method only;
                                          # fewer bins reduces intermodulation 4× vs 512)
    depth_min_m: float = 50.0            # surface exclusion (50m excludes near-field noise band)
    depth_max_m: float = 1000.0           # max depth
    sv_min_db: float = -90.0              # maps to amplitude 0
    sv_max_db: float = -50.0              # maps to amplitude 1
    # Data source
    sv_variable: str = 'Sv'             # 'Sv' (raw calibrated, impulse-masked) — the noise floor
                                         # acts as a continuous carrier that gives smoother audio.
                                         # Use 'Sv_corrected' only for quantitative/scientific output.
    # Synthesis quality
    perceptual_exponent: float = 2.0     # amplitude shaping: 1.0=linear, 2.0=quiets background hum
    envelope_smooth_ms: float = 150.0    # amplitude envelope smoothing (ms).
                                         # Each ping = ~8ms audio at 360× compression.
                                         # 150ms audio = ~54s real time, well below meaningful
                                         # biological timescales (DVM acts over hours).
    # Output
    normalisation: str = 'percentile'    # 'percentile' (robust to impulses) or 'peak'
    normalisation_percentile: float = 99.5  # top (100-N)% clipped before normalising
    output_bits: int = 32                # WAV bit depth: 32 (float32) or 16 (int16)
    # Synthesis method
    synthesis_method: str = 'additive'  # 'additive' (sinusoidal) or 'ifft' (ISTFT-based)
    ifft_nperseg: int = 8192            # STFT window length (samples). 8192 @ 44100 Hz → 186ms
                                         # audio = ~67s real time at 360× compression.
                                         # Frequency resolution = 44100 / 8192 ≈ 5.4 Hz.
    ifft_iter: int = 0                  # Griffin-Lim iterations. 0 = random phase (fast, oceanic
                                         # noise texture). ~32 = iterative refinement (slower,
                                         # more spectrally consistent with echogram).


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def silence_nan_pings(sv_data, fill_db):
    """
    Set all-NaN pings (impulse-masked rows) to fill_db (silence).

    Replaces the previous nearest-neighbor approach, which introduced false
    biology data at ping boundaries. Silencing is scientifically honest:
    a masked ping plays as silence, not as a fabricated copy of a neighbour.
    """
    sv_filled = sv_data.copy()
    all_nan_mask = np.all(np.isnan(sv_filled), axis=1)
    n_silent = int(np.sum(all_nan_mask))

    if n_silent > 0:
        sv_filled[all_nan_mask, :] = fill_db
        pct = 100 * n_silent / sv_data.shape[0]
        print(f"  Silenced {n_silent} all-NaN pings ({pct:.1f}%) → {fill_db} dB")

    return sv_filled


def detect_and_silence_outlier_pings(sv_data, fill_db, sigma=3.0):
    """
    Detect and silence pings whose column-mean Sv is a statistical outlier.

    Uses Modified Z-score (median ± sigma × 1.4826 × MAD) on the per-ping
    mean Sv. Bright outlier pings (electrical interference, sonar ringing)
    that survived the upstream impulse-noise cleaning are silenced here,
    preventing them from causing broadband beep/bleep artifacts in audio.
    """
    col_means = np.nanmean(sv_data, axis=1)
    median = np.nanmedian(col_means)
    mad = np.nanmedian(np.abs(col_means - median))

    if mad == 0:
        print("  Outlier detection: MAD=0, no outliers detected")
        return sv_data, 0

    modified_z = np.abs(col_means - median) / (1.4826 * mad)
    outlier_mask = modified_z > sigma

    n_outliers = int(np.sum(outlier_mask))
    if n_outliers > 0:
        sv_data[outlier_mask, :] = fill_db
        pct = 100 * n_outliers / sv_data.shape[0]
        print(f"  Silenced {n_outliers} outlier pings ({pct:.1f}%) "
              f"[Modified Z-score > {sigma}, MAD={mad:.2f} dB]")
    else:
        print(f"  Outlier detection: 0 outliers found (MAD={mad:.2f} dB)")

    return sv_data, n_outliers


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_echogram(date_str, config):
    """Load cleaned 24h NetCDF and return (sv_data, depths, ping_times)."""
    nc_path = OUTPUT_DATA / f"cleaned_Sv_24h_{date_str}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Cleaned NetCDF not found: {nc_path}\n"
            f"Run 'python src/processing/run_24h_processing.py {date_str}' first."
        )

    print(f"\nLoading {nc_path.name}...")
    ds = xr.open_dataset(nc_path)

    sv_var = config.sv_variable
    if sv_var not in ds:
        print(f"  WARNING: '{sv_var}' not found in NetCDF — falling back to 'Sv'")
        sv_var = 'Sv'
    else:
        print(f"  Using variable: {sv_var}")

    sv_data = ds[sv_var].values
    depths = ds['depth'].values
    ping_times = ds['ping_time'].values
    ds.close()

    print(f"  {sv_data.shape[0]} pings × {sv_data.shape[1]} depth samples")

    sv_data = silence_nan_pings(sv_data, config.sv_min_db)
    sv_data, _ = detect_and_silence_outlier_pings(sv_data, config.sv_min_db)

    return sv_data, depths, ping_times


# ---------------------------------------------------------------------------
# Amplitude grid preparation (shared by both synthesis paths)
# ---------------------------------------------------------------------------

def prepare_amplitude_grid(sv_data, depths, config):
    """
    Crop to depth range and convert dB to linear amplitude.

    Shared by both additive and IFFT synthesis paths. Returns the native-
    resolution depth grid — no downsampling (additive path zooms afterwards).

    Returns
    -------
    amplitude_grid : (n_pings, n_depth_crop) array, values in [0, 1]
    depths_crop    : (n_depth_crop,) depth values in metres
    """
    depth_mask = (depths >= config.depth_min_m) & (depths <= config.depth_max_m)
    sv_crop = sv_data[:, depth_mask]
    depths_crop = depths[depth_mask]
    n_pings, n_depth_crop = sv_crop.shape
    print(f"  Depth crop: {n_depth_crop} bins ({config.depth_min_m:.0f}–{config.depth_max_m:.0f}m)")

    valid_sv = sv_crop[~np.isnan(sv_crop) & (sv_crop > -200)]
    if len(valid_sv) > 0:
        p5, p99 = np.percentile(valid_sv, [5, 99])
        print(f"  Observed Sv range: p5={p5:.1f} dB, p99={p99:.1f} dB  "
              f"(configured: {config.sv_min_db}–{config.sv_max_db} dB)")

    # Replace NaN with sv_min (silence)
    sv_crop = np.nan_to_num(sv_crop, nan=config.sv_min_db)

    # dB → linear amplitude [0, 1]
    amplitude = np.clip(
        (sv_crop - config.sv_min_db) / (config.sv_max_db - config.sv_min_db),
        0.0, 1.0
    )

    # Perceptual shaping: exponent > 1 suppresses near-zero noise
    amplitude = amplitude ** config.perceptual_exponent

    print(f"  Amplitude range: {amplitude.min():.4f} – {amplitude.max():.4f}")
    print(f"  Non-zero bins: {np.count_nonzero(amplitude)}/{amplitude.size} "
          f"({100*np.count_nonzero(amplitude)/amplitude.size:.1f}%)")

    return amplitude, depths_crop


def prepare_spectrogram(sv_data, depths, config):
    """
    Crop depth range, downsample to n_freq_bins, convert dB to linear amplitude.

    Additive synthesis path only. Wraps prepare_amplitude_grid with a zoom step
    to reduce depth axis from native resolution to config.n_freq_bins.

    Returns (n_pings, n_freq_bins) array with values in [0, 1].
    Index 0 = shallowest (highest frequency), last index = deepest (lowest frequency).
    """
    amplitude_grid, depths_crop = prepare_amplitude_grid(sv_data, depths, config)
    n_pings, n_depth_crop = amplitude_grid.shape

    zoom_factor = config.n_freq_bins / n_depth_crop
    spectrogram = zoom(amplitude_grid, (1.0, zoom_factor), order=1)
    print(f"  Downsampled: {spectrogram.shape[0]} × {spectrogram.shape[1]}")

    return spectrogram


# ---------------------------------------------------------------------------
# Additive synthesis
# ---------------------------------------------------------------------------

def generate_frequency_table(config):
    """
    Generate log-spaced frequencies: index 0 = highest (shallow), last = lowest (deep).
    """
    freqs = np.logspace(
        np.log10(config.freq_max_hz),
        np.log10(config.freq_min_hz),
        config.n_freq_bins
    )
    print(f"\n  Frequency mapping: {freqs[0]:.0f} Hz (shallow) → {freqs[-1]:.0f} Hz (deep)")
    return freqs


def synthesize_additive(spectrogram, freqs, config):
    """
    Additive synthesis: each frequency bin is a sinusoid whose amplitude
    envelope follows the corresponding column of the spectrogram.

    - Random initial phase per bin eliminates constructive-interference crackling
    - Amplitude envelope smoothing (uniform filter) reduces click artifacts
      at ping boundaries without affecting scientific information content

    Processes in blocks of time and frequency to manage memory.
    """
    n_pings, n_bins = spectrogram.shape
    n_total_samples = int(config.audio_duration_s * config.sample_rate)

    print(f"\n  Synthesizing {config.audio_duration_s:.0f}s audio "
          f"({n_total_samples} samples @ {config.sample_rate} Hz)")
    print(f"  {n_bins} frequency bins, block-based processing...")

    audio = np.zeros(n_total_samples, dtype=np.float64)

    ping_positions = np.linspace(0, n_total_samples - 1, n_pings)
    sample_indices = np.arange(n_total_samples, dtype=np.float64)

    # Random phase offsets per bin — fixed seed for reproducible output.
    # Spreading phases prevents all sinusoids peaking simultaneously.
    rng = np.random.default_rng(seed=42)
    random_phases = rng.uniform(0.0, 2.0 * np.pi, n_bins)

    smooth_samples = max(1, int(config.envelope_smooth_ms * config.sample_rate / 1000))
    print(f"  Envelope smoothing: {config.envelope_smooth_ms:.0f} ms "
          f"({smooth_samples} samples)")

    FREQ_BLOCK = 32
    TIME_BLOCK = config.sample_rate * 10  # 10 seconds of audio

    n_freq_blocks = (n_bins + FREQ_BLOCK - 1) // FREQ_BLOCK
    n_time_blocks = (n_total_samples + TIME_BLOCK - 1) // TIME_BLOCK
    total_blocks = n_freq_blocks * n_time_blocks
    block_count = 0

    for fb in range(n_freq_blocks):
        f_start = fb * FREQ_BLOCK
        f_end = min(f_start + FREQ_BLOCK, n_bins)
        block_freqs = freqs[f_start:f_end]
        block_phases = random_phases[f_start:f_end]

        block_amps_full = np.zeros((n_total_samples, f_end - f_start))
        for i, bin_idx in enumerate(range(f_start, f_end)):
            envelope = np.interp(
                sample_indices, ping_positions, spectrogram[:, bin_idx]
            )
            if smooth_samples > 1:
                envelope = uniform_filter1d(envelope, smooth_samples)
            block_amps_full[:, i] = envelope

        for tb in range(n_time_blocks):
            t_start = tb * TIME_BLOCK
            t_end = min(t_start + TIME_BLOCK, n_total_samples)
            chunk_len = t_end - t_start

            t_chunk = (np.arange(chunk_len) + t_start).astype(np.float64) / config.sample_rate
            block_amps = block_amps_full[t_start:t_end, :]

            phases = (2.0 * np.pi * t_chunk[:, np.newaxis] * block_freqs[np.newaxis, :]
                      + block_phases[np.newaxis, :])
            sinusoids = np.sin(phases)

            audio[t_start:t_end] += np.sum(sinusoids * block_amps, axis=1)

            block_count += 1
            if block_count % 50 == 0 or block_count == total_blocks:
                print(f"    [{block_count}/{total_blocks}] blocks processed "
                      f"({100*block_count/total_blocks:.0f}%)")

    audio /= n_bins

    return audio


# ---------------------------------------------------------------------------
# IFFT synthesis (Griffin-Lim / random phase)
# ---------------------------------------------------------------------------

def synthesize_ifft(amplitude_grid, depths_crop, config):
    """
    IFFT (ISTFT) synthesis: treat echogram as magnitude spectrogram and invert.

    Instead of summing amplitude-modulated sinusoids, this method:
      1. Builds a magnitude spectrogram M[freq_bin, time_frame] by reading
         amplitude values from the echogram at each STFT bin's corresponding depth.
      2. Assigns random phases (or iteratively refines with Griffin-Lim).
      3. Inverts via ISTFT to produce the audio signal.

    This avoids all AM sidebands (intermodulation) inherent to additive synthesis.
    The result has a smooth, oceanic noise texture with faithful spectral content.

    Parameters
    ----------
    amplitude_grid : (n_pings, n_depth_crop) linear amplitude, values in [0, 1]
    depths_crop    : (n_depth_crop,) depth in metres
    config         : EchogramAudioConfig
    """
    nperseg = config.ifft_nperseg
    hop = nperseg // 4                     # 75% overlap
    n_total_samples = int(config.audio_duration_s * config.sample_rate)

    # STFT frequency axis (linear Hz, rfft)
    freqs_stft = np.fft.rfftfreq(nperseg, 1.0 / config.sample_rate)
    n_stft_bins = len(freqs_stft)

    # Number of STFT frames for the target audio length
    n_stft_frames = 1 + (n_total_samples - nperseg) // hop

    print(f"\n  IFFT synthesis: nperseg={nperseg}, hop={hop} (75% overlap)")
    print(f"  STFT grid: {n_stft_bins} freq bins × {n_stft_frames} time frames")
    print(f"  Freq resolution: {freqs_stft[1]:.2f} Hz/bin")

    n_pings = amplitude_grid.shape[0]
    ping_positions = np.linspace(0, n_pings - 1, n_stft_frames)

    smooth_frames = max(1, int(config.envelope_smooth_ms * config.sample_rate / 1000 / hop))
    print(f"  Envelope smoothing: {config.envelope_smooth_ms:.0f} ms → {smooth_frames} frames")

    # Depth → fractional index lookup
    depth_indices = np.arange(len(depths_crop), dtype=np.float64)

    # Log mapping constants (depth → frequency, same as additive path)
    log_fmax = np.log10(config.freq_max_hz)
    log_fmin = np.log10(config.freq_min_hz)

    # Build magnitude grid M[n_stft_bins, n_stft_frames]
    print("  Building magnitude grid...")
    M = np.zeros((n_stft_bins, n_stft_frames), dtype=np.float64)

    for bin_idx in range(n_stft_bins):
        f = freqs_stft[bin_idx]
        if f < config.freq_min_hz or f > config.freq_max_hz:
            continue

        # Invert log mapping: freq → depth
        frac = (log_fmax - np.log10(f)) / (log_fmax - log_fmin)
        depth = config.depth_min_m + frac * (config.depth_max_m - config.depth_min_m)

        # Fractional index into depths_crop axis
        d_idx = np.interp(depth, depths_crop, depth_indices)
        d_lo = int(np.floor(d_idx))
        d_hi = min(d_lo + 1, len(depths_crop) - 1)
        d_frac = d_idx - d_lo

        # Bilinear interpolation across pings at this depth
        amp_lo = np.interp(ping_positions, np.arange(n_pings), amplitude_grid[:, d_lo])
        amp_hi = np.interp(ping_positions, np.arange(n_pings), amplitude_grid[:, d_hi])
        amp_frames = amp_lo * (1.0 - d_frac) + amp_hi * d_frac

        if smooth_frames > 1:
            amp_frames = uniform_filter1d(amp_frames, smooth_frames)

        M[bin_idx, :] = amp_frames

    n_nonzero = np.count_nonzero(M)
    print(f"  M range: {M.min():.4f} – {M.max():.4f}  "
          f"| Non-zero: {n_nonzero}/{M.size} ({100*n_nonzero/M.size:.1f}%)")

    # Initialise random phases
    rng = np.random.default_rng(seed=42)
    phases = rng.uniform(0.0, 2.0 * np.pi, M.shape)

    # Griffin-Lim iterations (skip when ifft_iter == 0)
    if config.ifft_iter > 0:
        print(f"  Griffin-Lim: {config.ifft_iter} iterations...")
        for i in range(config.ifft_iter):
            spec_complex = M * np.exp(1j * phases)
            _, audio_iter = istft(spec_complex, fs=config.sample_rate,
                                  nperseg=nperseg, noverlap=nperseg - hop,
                                  window='hann')
            _, _, spec_new = stft(audio_iter, fs=config.sample_rate,
                                  nperseg=nperseg, noverlap=nperseg - hop,
                                  window='hann')
            # Update phases (trim to n_stft_frames in case lengths differ slightly)
            n_frames_new = spec_new.shape[1]
            n_update = min(n_frames_new, n_stft_frames)
            phases[:, :n_update] = np.angle(spec_new[:, :n_update])

            if (i + 1) % 8 == 0 or i == config.ifft_iter - 1:
                print(f"    [{i+1}/{config.ifft_iter}] iterations")
    else:
        print("  Using random phases (ifft_iter=0).")

    # Final synthesis
    spec_complex = M * np.exp(1j * phases)
    _, audio = istft(spec_complex, fs=config.sample_rate,
                     nperseg=nperseg, noverlap=nperseg - hop,
                     window='hann')

    # Trim or pad to exact target length
    if len(audio) > n_total_samples:
        audio = audio[:n_total_samples]
    elif len(audio) < n_total_samples:
        audio = np.pad(audio, (0, n_total_samples - len(audio)))

    print(f"  Output: {len(audio)} samples ({len(audio)/config.sample_rate:.1f}s)")

    return audio.astype(np.float64)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_wav(audio, config, output_path):
    """
    Normalise to -1 dBFS and write WAV file.

    Normalisation modes:
    - 'percentile': clips top (100 - normalisation_percentile)% before setting
      reference level. Prevents a single impulsive event suppressing everything else.
    - 'peak': classic peak normalisation.

    Output bit depth: 32-bit float or 16-bit int, per config.output_bits.
    """
    if config.normalisation == 'percentile':
        clip_level = np.percentile(np.abs(audio), config.normalisation_percentile)
        if clip_level > 0:
            audio = np.clip(audio, -clip_level, clip_level)
            peak = clip_level
        else:
            peak = np.max(np.abs(audio))
        print(f"\n  Normalisation: percentile {config.normalisation_percentile} "
              f"(clip level: {20*np.log10(clip_level + 1e-10):.1f} dB)")
    else:
        peak = np.max(np.abs(audio))

    if peak > 0:
        target = 10 ** (-1.0 / 20.0)  # -1 dBFS ≈ 0.891
        audio = audio * (target / peak)

    if config.output_bits == 32:
        wavfile.write(str(output_path), config.sample_rate, audio.astype(np.float32))
    else:
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(str(output_path), config.sample_rate, audio_int16)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB, {config.output_bits}-bit)")


def save_provenance(date_str, config, wav_path):
    """Save a JSON sidecar recording all parameters used to produce the WAV."""
    sidecar = {
        'date': date_str,
        'source_nc': f'cleaned_Sv_24h_{date_str}.nc',
        'output_wav': wav_path.name,
        'synthesis_method': config.synthesis_method,
        'sv_variable': config.sv_variable,
        'depth_range_m': [config.depth_min_m, config.depth_max_m],
        'sv_range_db': [config.sv_min_db, config.sv_max_db],
        'freq_range_hz': [config.freq_min_hz, config.freq_max_hz],
        'n_freq_bins': config.n_freq_bins,
        'audio_duration_s': config.audio_duration_s,
        'sample_rate_hz': config.sample_rate,
        'perceptual_exponent': config.perceptual_exponent,
        'envelope_smooth_ms': config.envelope_smooth_ms,
        'normalisation': f'{config.normalisation}_{config.normalisation_percentile}',
        'output_bits': config.output_bits,
        'synthesis_phase_seed': 42,
        'ifft_nperseg': config.ifft_nperseg,
        'ifft_iter': config.ifft_iter,
        'created': str(np.datetime64('now')),
    }
    json_path = wav_path.with_name(wav_path.stem + '_params.json')
    with open(json_path, 'w') as f:
        json.dump(sidecar, f, indent=2)
    print(f"  Provenance: {json_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(date_str, config):
    """Full pipeline: load → prepare → synthesize → save."""
    print("=" * 70)
    print("ECHOGRAM → AUDIO DIRECT MAPPING")
    print("=" * 70)
    print(f"Date:       {date_str}")
    print(f"Method:     {config.synthesis_method}")
    print(f"Sv source:  {config.sv_variable}")
    print(f"Duration:   {config.audio_duration_s:.0f}s | "
          f"Freq: {config.freq_min_hz:.0f}–{config.freq_max_hz:.0f} Hz | "
          f"SR: {config.sample_rate}")
    print(f"Depth:      {config.depth_min_m:.0f}–{config.depth_max_m:.0f} m")
    if config.synthesis_method == 'additive':
        print(f"Synthesis:  additive | bins={config.n_freq_bins} | "
              f"exponent={config.perceptual_exponent} | "
              f"envelope_smooth={config.envelope_smooth_ms}ms | "
              f"norm={config.normalisation} | bits={config.output_bits}")
    else:
        print(f"Synthesis:  IFFT | nperseg={config.ifft_nperseg} | "
              f"GL_iter={config.ifft_iter} | "
              f"exponent={config.perceptual_exponent} | "
              f"envelope_smooth={config.envelope_smooth_ms}ms | "
              f"norm={config.normalisation} | bits={config.output_bits}")
    print("=" * 70)

    # Load
    sv_data, depths, ping_times = load_echogram(date_str, config)

    if config.synthesis_method == 'ifft':
        print("\nPreparing amplitude grid (IFFT path)...")
        amplitude_grid, depths_crop = prepare_amplitude_grid(sv_data, depths, config)
        print("\nSynthesizing audio (IFFT synthesis)...")
        audio = synthesize_ifft(amplitude_grid, depths_crop, config)
        ifft_suffix = f"_ifft_gl{config.ifft_iter}" if config.ifft_iter > 0 else "_ifft"
        wav_path = OUTPUT_DATA / f"echogram_audio_{date_str}{ifft_suffix}.wav"
    else:
        print("\nPreparing spectrogram (additive path)...")
        spectrogram = prepare_spectrogram(sv_data, depths, config)
        freqs = generate_frequency_table(config)
        print("\nSynthesizing audio (additive synthesis)...")
        audio = synthesize_additive(spectrogram, freqs, config)
        wav_path = OUTPUT_DATA / f"echogram_audio_{date_str}.wav"

    peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    print(f"  Peak level before normalisation: {peak_db:.1f} dB")

    # Save
    save_wav(audio, config, wav_path)
    save_provenance(date_str, config, wav_path)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    duration_min = config.audio_duration_s / 60
    compression = 86400 / config.audio_duration_s
    print(f"  Audio:   {duration_min:.1f} min ({compression:.0f}× time compression)")
    print(f"  Mapping: {config.depth_min_m:.0f}m → {config.freq_max_hz:.0f} Hz, "
          f"{config.depth_max_m:.0f}m → {config.freq_min_hz:.0f} Hz")
    print(f"  Output:  {wav_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate audio (WAV) from echogram via direct mapping."
    )
    parser.add_argument(
        'date', nargs='?', default='20110126',
        help='Date in YYYYMMDD format (default: 20110126)'
    )
    parser.add_argument(
        '--method', type=str, default='additive',
        choices=['additive', 'ifft'],
        help='Synthesis method: additive (sinusoidal partials) or '
             'ifft (ISTFT-based, no AM sidebands). (default: additive)'
    )
    parser.add_argument(
        '--ifft-iter', type=int, default=0,
        help='Griffin-Lim iterations for IFFT method: 0=random phase (fast), '
             '32=iterative phase refinement (slower, more spectrally consistent). '
             '(default: 0)'
    )
    parser.add_argument(
        '--duration', type=float, default=240.0,
        help='Audio duration in seconds (default: 240 = 4 min)'
    )
    parser.add_argument(
        '--freq-min', type=float, default=50.0,
        help='Minimum frequency in Hz (default: 50)'
    )
    parser.add_argument(
        '--freq-max', type=float, default=8000.0,
        help='Maximum frequency in Hz (default: 8000)'
    )
    parser.add_argument(
        '--freq-bins', type=int, default=256,
        help='Number of frequency bins for additive method (default: 256; '
             'ignored for ifft)'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=44100,
        help='Audio sample rate (default: 44100)'
    )
    parser.add_argument(
        '--depth-min', type=float, default=50.0,
        help='Minimum depth in metres — surface exclusion zone (default: 50)'
    )
    parser.add_argument(
        '--sv-variable', type=str, default='Sv',
        choices=['Sv', 'Sv_corrected'],
        help='NetCDF Sv variable to use (default: Sv). '
             'Sv_corrected gives sparser, higher-contrast output.'
    )
    parser.add_argument(
        '--perceptual-exponent', type=float, default=2.0,
        help='Amplitude shaping exponent: 1.0=linear, 2.0=suppress background (default: 2.0)'
    )
    parser.add_argument(
        '--output-bits', type=int, default=32, choices=[16, 32],
        help='WAV output bit depth: 32=float32, 16=int16 (default: 32)'
    )
    args = parser.parse_args()

    ensure_dirs()

    config = EchogramAudioConfig(
        audio_duration_s=args.duration,
        freq_min_hz=args.freq_min,
        freq_max_hz=args.freq_max,
        n_freq_bins=args.freq_bins,
        sample_rate=args.sample_rate,
        depth_min_m=args.depth_min,
        sv_variable=args.sv_variable,
        perceptual_exponent=args.perceptual_exponent,
        output_bits=args.output_bits,
        synthesis_method=args.method,
        ifft_iter=args.ifft_iter,
    )

    main(args.date, config)
