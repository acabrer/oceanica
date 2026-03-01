#!/usr/bin/env python3
"""
================================================================================
AUDIO SPECTROGRAM — VALIDATION & COMPARISON
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya, Departament de Cultura

Generates a spectrogram from the synthesized echogram audio WAV and displays
it alongside the original 24h echogram for visual validation.

The two panels should show matching biological structures:
  - Main scattering layer (dense mid-frequency band)
  - DVM arc (rising/falling pitch over the day/night cycle)
  - Relative silence in empty water column

Because the synthesis maps depth → frequency logarithmically (50m → 8000 Hz,
1000m → 50 Hz), the audio spectrogram frequency axis corresponds directly to
the echogram depth axis. A secondary depth scale is shown on the right.

Produces:
  audio_spectrogram_{date}.png  — 2-panel comparison figure

Usage:
    python src/visualization/audio_to_spectrogram.py [YYYYMMDD]
    python src/visualization/audio_to_spectrogram.py 20110126
    python src/visualization/audio_to_spectrogram.py 20110127

Author: Oceànica (Alex Cabrer & Joan Sala)
================================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DATA, OUTPUT_VIZ, ensure_dirs

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.io import wavfile
from scipy.signal import spectrogram as scipy_spectrogram
import json
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Frequency ↔ Depth conversion (must match echogram_to_audio.py mapping)
# ---------------------------------------------------------------------------

def freq_to_depth(freq_hz, freq_min=50.0, freq_max=8000.0,
                  depth_min=50.0, depth_max=1000.0):
    """
    Invert the log-spaced depth→frequency mapping from echogram_to_audio.py.

    The synthesis maps:
        depth_min (shallow) → freq_max (high pitch)
        depth_max (deep)    → freq_min (low pitch)
    logarithmically.
    """
    log_fmax = np.log10(freq_max)
    log_fmin = np.log10(freq_min)
    # Fraction along the log frequency range (0 = freq_max, 1 = freq_min)
    frac = (log_fmax - np.log10(np.clip(freq_hz, freq_min, freq_max))) / (log_fmax - log_fmin)
    return depth_min + frac * (depth_max - depth_min)


def depth_to_freq(depth_m, freq_min=50.0, freq_max=8000.0,
                  depth_min=50.0, depth_max=1000.0):
    """Forward mapping: depth → frequency."""
    frac = (depth_m - depth_min) / (depth_max - depth_min)
    return 10 ** (np.log10(freq_max) - frac * (np.log10(freq_max) - np.log10(freq_min)))


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_audio(date_str, method='additive', ifft_iter=0):
    """Load synthesized WAV and its provenance JSON."""
    if method == 'ifft':
        suffix = f"_ifft_gl{ifft_iter}" if ifft_iter > 0 else "_ifft"
    else:
        suffix = ''
    wav_path = OUTPUT_DATA / f"echogram_audio_{date_str}{suffix}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {wav_path}\n"
            f"Run 'python src/extraction/echogram_to_audio.py {date_str}' first."
        )

    sample_rate, audio = wavfile.read(str(wav_path))

    # Handle float32 (32-bit) and int16 (16-bit) WAV formats
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    print(f"  Audio: {len(audio)/sample_rate:.1f}s @ {sample_rate} Hz "
          f"| peak={np.max(np.abs(audio)):.3f} | dtype={audio.dtype}")

    # Load provenance parameters if available
    params_path = wav_path.with_name(wav_path.stem + '_params.json')
    params = {}
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        print(f"  Params: {params_path.name}")
    else:
        print(f"  No params JSON found — using defaults")

    return audio, sample_rate, params


def load_echogram(date_str):
    """Load Sv echogram from cleaned NetCDF for comparison panel."""
    nc_path = OUTPUT_DATA / f"cleaned_Sv_24h_{date_str}.nc"
    if not nc_path.exists():
        print(f"  WARNING: echogram NetCDF not found ({nc_path.name}) — skipping comparison panel")
        return None, None, None

    ds = xr.open_dataset(nc_path)
    sv_data = ds['Sv'].values          # (n_pings, n_depth) — raw calibrated for display
    depths = ds['depth'].values
    times = ds['ping_time'].values
    ds.close()

    # Hours of day (0–24) for x-axis alignment with spectrogram
    t0 = times[0].astype('datetime64[s]').astype(object)
    hours = np.array([(t.astype('datetime64[s]').astype(object) - t0).total_seconds() / 3600
                      for t in times])

    print(f"  Echogram: {sv_data.shape[0]} pings × {sv_data.shape[1]} depth samples")
    return sv_data, depths, hours


# ---------------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------------

def compute_spectrogram(audio, sample_rate, nperseg=8192, overlap_frac=0.75):
    """
    Compute magnitude spectrogram in dB using short-time Fourier transform.

    Parameters
    ----------
    nperseg : int
        FFT window length in samples.
        8192 @ 44100 Hz → 186ms time resolution → ~67s real ocean time at 360×.
    overlap_frac : float
        Fractional overlap between windows (0.75 = 75%).

    Returns
    -------
    freqs : (n_freq,) Hz
    times : (n_time,) seconds
    S_db  : (n_freq, n_time) magnitude in dB
    """
    noverlap = int(nperseg * overlap_frac)
    freqs, times, Sxx = scipy_spectrogram(
        audio,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum'
    )

    # Magnitude in dB (floor at -120 dB to avoid log(0))
    S_db = 10.0 * np.log10(np.maximum(Sxx, 1e-12))

    print(f"  Spectrogram: {len(freqs)} freq bins × {len(times)} time frames")
    print(f"  Freq resolution: {freqs[1]-freqs[0]:.1f} Hz | "
          f"Time resolution: {(times[1]-times[0])*1000:.0f} ms per frame")
    print(f"  S_db range: {S_db.min():.1f} – {S_db.max():.1f} dB")

    return freqs, times, S_db


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(date_str, audio, sample_rate, params,
                    sv_data, depths, echo_hours, out_suffix=''):
    """
    2-panel figure: audio spectrogram (top) + echogram (bottom).
    Both share the same time axis (hours of day, 0–24).
    """
    # --- Spectrogram ---
    print("\nComputing spectrogram...")
    freqs, spec_times, S_db = compute_spectrogram(audio, sample_rate)

    # Convert spectrogram time axis to hours of day (audio covers 0–24h)
    audio_duration_h = len(audio) / sample_rate / 3600 * 24 / 24  # always maps to 24h
    spec_hours = spec_times / spec_times[-1] * 24.0  # scale 0–24h

    # Frequency range from params (fallback to defaults)
    freq_min = params.get('freq_range_hz', [50.0, 8000.0])[0]
    freq_max = params.get('freq_range_hz', [50.0, 8000.0])[1]
    depth_min = params.get('depth_range_m', [50.0, 1000.0])[0]
    depth_max = params.get('depth_range_m', [50.0, 1000.0])[1]

    # Mask spectrogram to synthesis frequency range only
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_plot = freqs[freq_mask]
    S_plot = S_db[freq_mask, :]

    # Robust colour limits: ignore the darkest 5% and brightest 1%
    vmin = np.percentile(S_plot, 5)
    vmax = np.percentile(S_plot, 99)

    # --- Layout ---
    has_echo = sv_data is not None
    n_panels = 2 if has_echo else 1
    fig_h = 10 if has_echo else 5

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, fig_h),
        sharex=True,
        gridspec_kw={'hspace': 0.08}
    )
    if n_panels == 1:
        axes = [axes]

    ax_spec = axes[0]
    ax_echo = axes[1] if has_echo else None

    # ----------------------------------------------------------------
    # Panel 1: Audio spectrogram
    # ----------------------------------------------------------------
    mesh = ax_spec.pcolormesh(
        spec_hours,
        freqs_plot,
        S_plot,
        shading='auto',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )

    ax_spec.set_yscale('log')
    ax_spec.set_ylim(freq_min, freq_max)
    ax_spec.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x)}' if x >= 100 else f'{x:.0f}'
    ))
    ax_spec.set_ylabel('Frequency (Hz)\n← deep (low)    shallow (high) →',
                        fontsize=10)
    ax_spec.set_title(
        f'MALASPINA Atlantic — {date_str[:4]}-{date_str[4:6]}-{date_str[6:]} '
        f'| Audio Spectrogram (synthesis frequency range {freq_min:.0f}–{freq_max:.0f} Hz)',
        fontsize=11, pad=6
    )

    cb1 = plt.colorbar(mesh, ax=ax_spec, pad=0.01, fraction=0.015)
    cb1.set_label('Power (dB)', fontsize=9)

    # Secondary y-axis: depth labels on the right
    ax_depth = ax_spec.twinx()
    ax_depth.set_yscale('log')
    ax_depth.set_ylim(freq_min, freq_max)

    # Tick positions at meaningful depth values
    depth_ticks = [50, 100, 200, 300, 500, 700, 1000]
    freq_ticks = [depth_to_freq(d, freq_min, freq_max, depth_min, depth_max)
                  for d in depth_ticks]
    # Only show ticks within our frequency range
    valid = [(f, d) for f, d in zip(freq_ticks, depth_ticks)
             if freq_min <= f <= freq_max]
    if valid:
        f_ticks, d_labels = zip(*valid)
        ax_depth.set_yticks(list(f_ticks))
        ax_depth.set_yticklabels([f'{d}m' for d in d_labels], fontsize=8)
    ax_depth.set_ylabel('Depth (m)', fontsize=9)

    # Annotation box: synthesis parameters
    sv_var = params.get('sv_variable', 'Sv')
    n_bins = params.get('n_freq_bins', '?')
    smooth = params.get('envelope_smooth_ms', '?')
    annot = (f"Source: {sv_var}  |  Bins: {n_bins}  |  "
             f"Envelope smooth: {smooth}ms  |  Depth: {depth_min:.0f}–{depth_max:.0f}m")
    ax_spec.text(0.01, 0.02, annot, transform=ax_spec.transAxes,
                 fontsize=8, color='white', alpha=0.85,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.4))

    # ----------------------------------------------------------------
    # Panel 2: Original echogram (for comparison)
    # ----------------------------------------------------------------
    if has_echo:
        # Crop to synthesis depth range
        depth_mask = (depths >= depth_min) & (depths <= depth_max)
        sv_plot = sv_data[:, depth_mask]
        depths_plot = depths[depth_mask]

        # Fill NaN for display
        nan_cols = np.all(np.isnan(sv_plot), axis=1)
        if np.any(nan_cols):
            valid_idx = np.where(~nan_cols)[0]
            for idx in np.where(nan_cols)[0]:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                sv_plot[idx] = sv_plot[nearest]

        sv_plot = np.nan_to_num(sv_plot, nan=-90.0)

        mesh2 = ax_echo.pcolormesh(
            echo_hours,
            depths_plot,
            sv_plot.T,
            shading='auto',
            cmap='viridis',
            vmin=-90, vmax=-50
        )
        ax_echo.invert_yaxis()
        ax_echo.set_ylabel('Depth (m)', fontsize=10)
        ax_echo.set_title(
            f'Original 38 kHz Echogram  (Sv range −90–−50 dB re 1 m⁻¹)',
            fontsize=11, pad=6
        )
        cb2 = plt.colorbar(mesh2, ax=ax_echo, pad=0.01, fraction=0.015)
        cb2.set_label('Sv (dB)', fontsize=9)

        ax_echo.set_xlabel('Time (hours of day)', fontsize=10)
        ax_echo.set_xlim(0, 24)
        ax_echo.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax_echo.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax_echo.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{int(x):02d}:00')
        )
    else:
        ax_spec.set_xlabel('Time (hours of day)', fontsize=10)
        ax_spec.set_xlim(0, 24)
        ax_spec.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax_spec.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{int(x):02d}:00')
        )

    # Shared x-axis limits
    if has_echo:
        ax_spec.set_xlim(0, 24)
        ax_spec.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax_spec.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    out_path = OUTPUT_VIZ / f"audio_spectrogram_{date_str}{out_suffix}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(date_str, method='additive', ifft_iter=0):
    print("=" * 70)
    print("AUDIO → SPECTROGRAM  (validation & echogram comparison)")
    print("=" * 70)
    print(f"Date:   {date_str}")
    print(f"Method: {method}" + (f" (GL iter={ifft_iter})" if method == 'ifft' and ifft_iter > 0 else ""))

    ensure_dirs()

    print("\nLoading audio...")
    audio, sample_rate, params = load_audio(date_str, method, ifft_iter)

    print("\nLoading echogram for comparison...")
    sv_data, depths, echo_hours = load_echogram(date_str)

    if method == 'ifft':
        out_suffix = f"_ifft_gl{ifft_iter}" if ifft_iter > 0 else "_ifft"
    else:
        out_suffix = ''

    print("\nRendering figure...")
    out_path = plot_comparison(date_str, audio, sample_rate, params,
                               sv_data, depths, echo_hours, out_suffix)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"  Output: {out_path}")
    print()
    print("  Validation guide:")
    print("  ✓ Dense mid-frequency band in spectrogram ↔ main scattering layer in echogram")
    print("  ✓ DVM arc: low frequencies during day, rising to high at dusk")
    print("  ✓ Relative silence in upper frequencies = no surface noise (50m exclusion)")
    print("  ✓ Similar brightness patterns indicate faithful Sv → amplitude mapping")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate spectrogram from echogram audio and compare with original echogram."
    )
    parser.add_argument(
        'date', nargs='?', default='20110126',
        help='Date in YYYYMMDD format (default: 20110126)'
    )
    parser.add_argument(
        '--method', type=str, default='additive',
        choices=['additive', 'ifft'],
        help='Synthesis method used to generate the WAV file (default: additive)'
    )
    parser.add_argument(
        '--ifft-iter', type=int, default=0,
        help='Griffin-Lim iterations used when generating the IFFT WAV '
             '(must match the value used in echogram_to_audio.py; default: 0)'
    )
    args = parser.parse_args()
    main(args.date, args.method, args.ifft_iter)
