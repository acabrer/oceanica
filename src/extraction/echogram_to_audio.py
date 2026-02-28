#!/usr/bin/env python3
"""
================================================================================
DIRECT ECHOGRAM → AUDIO MAPPING
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya, Departament de Cultura

Treats the echogram as an audio spectrogram and synthesizes a WAV file
via additive synthesis:

    depth axis   → frequency axis  (shallow=high pitch, deep=low pitch)
    Sv intensity → amplitude       (strong backscatter=loud, weak=quiet)
    ping time    → audio time      (24 hours compressed to ~4 minutes)

Reads cleaned 24h NetCDF data (output of run_24h_processing.py).

Usage:
    python src/extraction/echogram_to_audio.py [YYYYMMDD]
    python src/extraction/echogram_to_audio.py 20110126 --duration 120
    python src/extraction/echogram_to_audio.py 20110126 --freq-bins 256

Author: Oceànica (Alex Cabrer & Joan Sala)
================================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DATA, ensure_dirs

import numpy as np
import xarray as xr
from scipy.ndimage import zoom
from scipy.io import wavfile
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
    n_freq_bins: int = 512                # sinusoidal partials
    depth_min_m: float = 10.0             # surface exclusion
    depth_max_m: float = 1000.0           # max depth
    sv_min_db: float = -90.0              # maps to amplitude 0
    sv_max_db: float = -50.0              # maps to amplitude 1


def fill_nan_pings(sv_data):
    """
    Fill all-NaN pings (impulse noise) via nearest-neighbor interpolation
    along the time axis. Display/synthesis only — does not alter stored data.
    """
    sv_filled = sv_data.copy()
    all_nan_mask = np.all(np.isnan(sv_filled), axis=1)

    if not np.any(all_nan_mask):
        return sv_filled

    nan_indices = np.where(all_nan_mask)[0]
    valid_indices = np.where(~all_nan_mask)[0]

    if len(valid_indices) == 0:
        return sv_filled

    for idx in nan_indices:
        distances = np.abs(valid_indices - idx)
        nearest = valid_indices[np.argmin(distances)]
        sv_filled[idx, :] = sv_filled[nearest, :]

    n_filled = len(nan_indices)
    pct = 100 * n_filled / sv_data.shape[0]
    print(f"  Filled {n_filled} all-NaN pings ({pct:.1f}%) via nearest-neighbor")

    return sv_filled


def load_echogram(date_str):
    """Load cleaned 24h NetCDF and return (sv_data, depths, ping_times)."""
    nc_path = OUTPUT_DATA / f"cleaned_Sv_24h_{date_str}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Cleaned NetCDF not found: {nc_path}\n"
            f"Run 'python src/processing/run_24h_processing.py {date_str}' first."
        )

    print(f"\nLoading {nc_path.name}...")
    ds = xr.open_dataset(nc_path)
    sv_data = ds['Sv'].values
    depths = ds['depth'].values
    ping_times = ds['ping_time'].values
    ds.close()

    print(f"  {sv_data.shape[0]} pings × {sv_data.shape[1]} depth samples")

    sv_data = fill_nan_pings(sv_data)

    return sv_data, depths, ping_times


def prepare_spectrogram(sv_data, depths, config):
    """
    Crop depth range, downsample to n_freq_bins, convert dB to linear amplitude.

    Returns (n_pings, n_freq_bins) array with values in [0, 1].
    Index 0 = shallowest (highest frequency), last index = deepest (lowest frequency).
    """
    # Crop to depth range
    depth_mask = (depths >= config.depth_min_m) & (depths <= config.depth_max_m)
    sv_crop = sv_data[:, depth_mask]
    n_pings, n_depth_crop = sv_crop.shape
    print(f"\n  Depth crop: {n_depth_crop} bins ({config.depth_min_m}–{config.depth_max_m}m)")

    # Downsample depth axis to n_freq_bins
    zoom_factor = config.n_freq_bins / n_depth_crop
    # zoom along axis=1 only (depth), keep axis=0 (pings) unchanged
    spectrogram = zoom(sv_crop, (1.0, zoom_factor), order=1)
    print(f"  Downsampled: {spectrogram.shape[0]} × {spectrogram.shape[1]}")

    # Replace NaN with sv_min (silence)
    spectrogram = np.nan_to_num(spectrogram, nan=config.sv_min_db)

    # dB to linear amplitude [0, 1]
    amplitude = np.clip(
        (spectrogram - config.sv_min_db) / (config.sv_max_db - config.sv_min_db),
        0.0, 1.0
    )

    # Perceptual squaring: makes quiet regions quieter, prevents constant hum
    amplitude = amplitude ** 2

    print(f"  Amplitude range: {amplitude.min():.4f} – {amplitude.max():.4f}")
    print(f"  Non-zero bins: {np.count_nonzero(amplitude)}/{amplitude.size} "
          f"({100*np.count_nonzero(amplitude)/amplitude.size:.1f}%)")

    return amplitude


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


def synthesize_audio(spectrogram, freqs, config):
    """
    Additive synthesis: each frequency bin is a sinusoid whose amplitude
    envelope follows the corresponding column of the spectrogram.

    Processes in blocks of time and frequency to manage memory.
    """
    n_pings, n_bins = spectrogram.shape
    n_total_samples = int(config.audio_duration_s * config.sample_rate)

    print(f"\n  Synthesizing {config.audio_duration_s:.0f}s audio "
          f"({n_total_samples} samples @ {config.sample_rate} Hz)")
    print(f"  {n_bins} frequency bins, block-based processing...")

    audio = np.zeros(n_total_samples, dtype=np.float64)

    # Map ping indices to audio sample positions for interpolation
    ping_positions = np.linspace(0, n_total_samples - 1, n_pings)
    sample_indices = np.arange(n_total_samples, dtype=np.float64)

    # Block sizes for memory management
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

        # Pre-interpolate amplitude envelopes for this frequency block
        # across the full time range
        block_amps_full = np.zeros((n_total_samples, f_end - f_start))
        for i, bin_idx in enumerate(range(f_start, f_end)):
            block_amps_full[:, i] = np.interp(
                sample_indices, ping_positions, spectrogram[:, bin_idx]
            )

        for tb in range(n_time_blocks):
            t_start = tb * TIME_BLOCK
            t_end = min(t_start + TIME_BLOCK, n_total_samples)
            chunk_len = t_end - t_start

            # Time vector for this chunk (absolute time for phase continuity)
            t_chunk = (np.arange(chunk_len) + t_start).astype(np.float64) / config.sample_rate

            # Amplitude envelopes for this chunk
            block_amps = block_amps_full[t_start:t_end, :]

            # Generate sinusoids: (chunk_len, n_freqs_in_block)
            phases = 2.0 * np.pi * t_chunk[:, np.newaxis] * block_freqs[np.newaxis, :]
            sinusoids = np.sin(phases)

            # Accumulate weighted sinusoids
            audio[t_start:t_end] += np.sum(sinusoids * block_amps, axis=1)

            block_count += 1
            if block_count % 50 == 0 or block_count == total_blocks:
                print(f"    [{block_count}/{total_blocks}] blocks processed "
                      f"({100*block_count/total_blocks:.0f}%)")

    # Normalize by number of bins (theoretical max amplitude = n_bins)
    audio /= n_bins

    return audio


def save_wav(audio, config, output_path):
    """Peak-normalize to -1 dB and write as 16-bit WAV."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        target = 10 ** (-1.0 / 20.0)  # -1 dB = ~0.891
        audio = audio * (target / peak)

    # Convert to int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    wavfile.write(str(output_path), config.sample_rate, audio_int16)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved: {output_path} ({size_mb:.1f} MB)")


def main(date_str, config):
    """Full pipeline: load → prepare → synthesize → save."""
    print("=" * 70)
    print("ECHOGRAM → AUDIO DIRECT MAPPING")
    print("=" * 70)
    print(f"Date: {date_str}")
    print(f"Duration: {config.audio_duration_s:.0f}s | "
          f"Freq: {config.freq_min_hz:.0f}–{config.freq_max_hz:.0f} Hz | "
          f"Bins: {config.n_freq_bins} | "
          f"SR: {config.sample_rate}")
    print("=" * 70)

    # Load
    sv_data, depths, ping_times = load_echogram(date_str)

    # Prepare spectrogram
    print("\nPreparing spectrogram...")
    spectrogram = prepare_spectrogram(sv_data, depths, config)
    freqs = generate_frequency_table(config)

    # Synthesize
    print("\nSynthesizing audio (additive synthesis)...")
    audio = synthesize_audio(spectrogram, freqs, config)

    peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    print(f"  Peak level: {peak_db:.1f} dB")

    # Save
    wav_path = OUTPUT_DATA / f"echogram_audio_{date_str}.wav"
    save_wav(audio, config, wav_path)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    duration_min = config.audio_duration_s / 60
    compression = 86400 / config.audio_duration_s
    print(f"  Audio: {duration_min:.1f} min ({compression:.0f}x time compression)")
    print(f"  Mapping: {config.depth_min_m:.0f}m → {config.freq_max_hz:.0f} Hz, "
          f"{config.depth_max_m:.0f}m → {config.freq_min_hz:.0f} Hz")
    print(f"  Output: {wav_path}")


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
        '--freq-bins', type=int, default=512,
        help='Number of frequency bins (default: 512)'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=44100,
        help='Audio sample rate (default: 44100)'
    )
    args = parser.parse_args()

    ensure_dirs()

    config = EchogramAudioConfig(
        audio_duration_s=args.duration,
        freq_min_hz=args.freq_min,
        freq_max_hz=args.freq_max,
        n_freq_bins=args.freq_bins,
        sample_rate=args.sample_rate,
    )

    main(args.date, config)
