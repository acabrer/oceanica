#!/usr/bin/env python3
"""
================================================================================
24-HOUR ECHOGRAM VISUALIZATION FOR VALIDATION
================================================================================
Project: Sonificació Oceànica

Reads from the pre-cleaned NetCDF file (output of run_24h_processing.py)
and generates echogram visualizations.

Produces:
  1. echogram_24h_validation_{date}.png  — 6-panel analysis figure
  2. echogram_24h_{date}.png              — clean single-panel 24h echogram

Author: Oceànica (Alex Cabrer & Joan Sala)
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DATA, OUTPUT_VIZ, ensure_dirs, ECHOGRAM_PRESETS

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import json
import warnings
warnings.filterwarnings('ignore')

# Import colormaps from centralized module
from colormaps import get_colormap


def fill_nan_pings(sv_data):
    """
    Fill all-NaN pings (impulse noise gaps) via nearest-neighbor interpolation
    along the time axis for display purposes only.

    Only fills columns where ALL depth samples are NaN (= impulse-masked pings).
    Partial-NaN columns (e.g., bottom echo masking) are left unchanged.

    Parameters
    ----------
    sv_data : np.ndarray
        2D array of shape (n_pings, n_depth).

    Returns
    -------
    np.ndarray
        Copy of sv_data with all-NaN columns filled from nearest valid neighbor.
    """
    sv_filled = sv_data.copy()
    all_nan_mask = np.all(np.isnan(sv_filled), axis=1)  # shape: (n_pings,)

    if not np.any(all_nan_mask):
        return sv_filled

    nan_indices = np.where(all_nan_mask)[0]
    valid_indices = np.where(~all_nan_mask)[0]

    if len(valid_indices) == 0:
        return sv_filled

    # For each NaN ping, find the nearest valid ping
    for idx in nan_indices:
        distances = np.abs(valid_indices - idx)
        nearest = valid_indices[np.argmin(distances)]
        sv_filled[idx, :] = sv_filled[nearest, :]

    n_filled = len(nan_indices)
    pct = 100 * n_filled / sv_data.shape[0]
    print(f"  Filled {n_filled} all-NaN pings ({pct:.1f}%) via nearest-neighbor for display")

    return sv_filled


def load_cleaned_data(date_str):
    """
    Load pre-cleaned 24h Sv data from the NetCDF produced by run_24h_processing.py.

    Parameters
    ----------
    date_str : str
        Date in YYYYMMDD format.

    Returns
    -------
    sv_data : np.ndarray
        2D array of shape (n_pings, n_depth) — cleaned Sv values.
    depths : np.ndarray
        1D array of depth values in metres.
    times : np.ndarray
        1D array of ping_time timestamps (numpy datetime64).
    """
    nc_path = OUTPUT_DATA / f"cleaned_Sv_24h_{date_str}.nc"
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Cleaned NetCDF not found: {nc_path}\n"
            f"Run 'python src/processing/run_24h_processing.py {date_str}' first."
        )

    ds = xr.open_dataset(nc_path)
    sv_data = ds['Sv'].values          # (ping_time, depth)
    depths = ds['depth'].values        # (depth,)
    times = ds['ping_time'].values     # (ping_time,)
    ds.close()

    print(f"Loaded cleaned data: {sv_data.shape[0]} pings × {sv_data.shape[1]} depth samples")
    print(f"  Depth range: 0 – {depths[-1]:.0f} m")
    print(f"  Time range:  {times[0]} → {times[-1]}")

    return sv_data, depths, times


def create_validation_visualization(date_str, output_dir):
    """Create comprehensive 24-hour validation visualization from cleaned data."""

    # Load pre-cleaned data
    sv_data, depths, times = load_cleaned_data(date_str)

    # Load extracted features for overlay (from output/data/)
    features_path = OUTPUT_DATA / f"sonification_sc_v3_{date_str}.json"
    if not features_path.exists():
        print(f"  Features file not found: {features_path}")
        print(f"  Skipping 6-panel validation — generating 24h echogram only.")

        # Convert times to hours
        start_time = times[0]
        hours = (times - start_time) / np.timedelta64(1, 'h')
        hours = np.array(hours, dtype=float)

        valid_mask = np.isfinite(hours) & (hours >= 0) & (hours <= 24)
        hours = hours[valid_mask]
        sv_data = sv_data[valid_mask, :]

        # Fill impulse-noise NaN pings for display (data file unchanged)
        sv_data = fill_nan_pings(sv_data)

        depths = np.array(depths, dtype=float)
        depths = np.nan_to_num(depths, nan=0)

        create_24h_echogram(sv_data, depths, hours, None, None, output_dir, date_str)
        return

    with open(features_path, 'r') as f:
        features = json.load(f)

    feat_times = np.array(features['38kHz']['time_seconds'])
    feat_depth = np.array(features['38kHz']['depth_m'])
    feat_velocity = np.array(features['38kHz']['velocity_m_h'])
    feat_intensity = np.array(features['38kHz']['intensity_norm'])
    feat_hour = np.array(features['38kHz']['hour_of_day'])

    # Convert times to hours
    start_time = times[0]
    hours = (times - start_time) / np.timedelta64(1, 'h')
    hours = np.array(hours, dtype=float)
    feat_hours = feat_times / 3600

    # Clean up any NaN in hours or depths
    valid_mask = np.isfinite(hours) & (hours >= 0) & (hours <= 24)
    hours = hours[valid_mask]
    sv_data = sv_data[valid_mask, :]

    # Fill impulse-noise NaN pings for display (data file unchanged)
    sv_data = fill_nan_pings(sv_data)

    # Ensure depths are finite
    depths = np.array(depths, dtype=float)
    depths = np.nan_to_num(depths, nan=0)

    # Create figure
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'MALASPINA LEG2 - Atlantic Ocean - {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}\n'
                 f'24-Hour Diel Vertical Migration Analysis (38 kHz) — Cleaned Sv data (noise removed)',
                 fontsize=14, fontweight='bold')

    # Get publication preset configuration (matches Klevjer et al. 2016 Figure 1)
    config = ECHOGRAM_PRESETS['publication']
    sv_min, sv_max = config['sv_range']
    depth_min, depth_max = config['depth_range']
    cmap = get_colormap(config['colormap'])

    # =========================================================================
    # Panel 1: Full Echogram with DVM overlay
    # =========================================================================
    ax1 = fig.add_subplot(3, 2, (1, 2))

    sv_plot = np.clip(sv_data.T, sv_min, sv_max)
    sv_plot = np.where(np.isnan(sv_plot), sv_min, sv_plot)

    im = ax1.pcolormesh(hours, depths, sv_plot, cmap=cmap, shading='auto', vmin=sv_min, vmax=sv_max)

    # Overlay center of mass line
    ax1.plot(feat_hours, feat_depth, 'r-', linewidth=1.5, alpha=0.8, label='Center of Mass')

    ax1.set_ylim(depth_max, depth_min)
    ax1.set_xlim(0, 24)
    ax1.set_xlabel('Hour (UTC)', fontsize=11)
    ax1.set_ylabel('Depth (m)', fontsize=11)
    ax1.set_title('Echogram with Extracted Center of Mass', fontsize=12)
    ax1.legend(loc='lower right')

    cbar = plt.colorbar(im, ax=ax1, label='Sv (dB re 1 m\u207b\u00b9)', pad=0.02)

    # Day/night shading
    ax1.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
    ax1.axvspan(6, 18, alpha=0.05, color='yellow')
    ax1.axvspan(18, 24, alpha=0.1, color='blue')

    # =========================================================================
    # Panel 2: Depth time series (smoothed)
    # =========================================================================
    ax2 = fig.add_subplot(3, 2, 3)

    ax2.plot(feat_hours, feat_depth, 'b-', linewidth=1, alpha=0.5, label='Raw')
    depth_smooth = uniform_filter1d(feat_depth, 200, mode='nearest')
    ax2.plot(feat_hours, depth_smooth, 'b-', linewidth=2, label='Smoothed')

    ax2.set_ylim(700, 100)
    ax2.set_xlim(0, 24)
    ax2.set_xlabel('Hour (UTC)', fontsize=11)
    ax2.set_ylabel('Depth (m)', fontsize=11)
    ax2.set_title('Center of Mass Depth Over Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.annotate('Night\n(Surface)', xy=(2, 250), fontsize=9, ha='center', color='blue')
    ax2.annotate('Dawn\nDescent', xy=(6.5, 350), fontsize=9, ha='center', color='orange')
    ax2.annotate('Day\n(Deep)', xy=(12, 500), fontsize=9, ha='center', color='red')
    ax2.annotate('Dusk\nAscent', xy=(19, 400), fontsize=9, ha='center', color='orange')
    ax2.annotate('Night\n(Surface)', xy=(22, 280), fontsize=9, ha='center', color='blue')

    # =========================================================================
    # Panel 3: Velocity (migration speed)
    # =========================================================================
    ax3 = fig.add_subplot(3, 2, 4)

    vel_clipped = np.clip(feat_velocity, -100, 100)
    vel_smooth = uniform_filter1d(vel_clipped, 200, mode='nearest')

    ax3.fill_between(feat_hours, 0, vel_smooth, where=vel_smooth > 0,
                     color='green', alpha=0.5, label='Ascending')
    ax3.fill_between(feat_hours, 0, vel_smooth, where=vel_smooth < 0,
                     color='red', alpha=0.5, label='Descending')
    ax3.axhline(y=0, color='black', linewidth=0.5)

    ax3.set_xlim(0, 24)
    ax3.set_ylim(-100, 100)
    ax3.set_xlabel('Hour (UTC)', fontsize=11)
    ax3.set_ylabel('Velocity (m/h)', fontsize=11)
    ax3.set_title('Migration Velocity (+ = ascending, - = descending)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4: Intensity (acoustic backscatter)
    # =========================================================================
    ax4 = fig.add_subplot(3, 2, 5)

    intensity_db = np.array(features['38kHz']['intensity_db'])
    int_smooth = uniform_filter1d(intensity_db, 200, mode='nearest')

    ax4.plot(feat_hours, int_smooth, 'purple', linewidth=1.5)
    ax4.fill_between(feat_hours, -50, int_smooth, alpha=0.3, color='purple')

    ax4.set_xlim(0, 24)
    ax4.set_ylim(-50, -25)
    ax4.set_xlabel('Hour (UTC)', fontsize=11)
    ax4.set_ylabel('Sv (dB)', fontsize=11)
    ax4.set_title('Total Acoustic Intensity (proxy for biomass)', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 5: Sonification Parameter Summary
    # =========================================================================
    ax5 = fig.add_subplot(3, 2, 6)

    depth_norm = np.array(features['38kHz']['depth_norm'])
    velocity_norm = np.array(features['38kHz']['velocity_norm'])
    spread_norm = np.array(features['38kHz']['spread_norm'])
    layers_norm = np.array(features['38kHz']['layers_norm'])

    ax5.plot(feat_hours, uniform_filter1d(depth_norm, 100), label='Depth (shallow=high)', linewidth=1.5)
    ax5.plot(feat_hours, uniform_filter1d(velocity_norm, 100), label='Velocity (up=high)', linewidth=1.5)
    ax5.plot(feat_hours, uniform_filter1d(spread_norm, 100), label='Spread', linewidth=1.5)
    ax5.plot(feat_hours, uniform_filter1d(layers_norm, 100), label='Layers', linewidth=1.5)

    ax5.set_xlim(0, 24)
    ax5.set_ylim(0, 1)
    ax5.set_xlabel('Hour (UTC)', fontsize=11)
    ax5.set_ylabel('Normalized Value', fontsize=11)
    ax5.set_title('Sonification Parameters (all normalized 0-1)', fontsize=12)
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = output_dir / f"echogram_24h_validation_{date_str}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    # Also create single-panel 24h echogram
    create_24h_echogram(sv_data, depths, hours, feat_hours, feat_depth, output_dir, date_str)

    plt.close()


def create_24h_echogram(sv_data, depths, hours, feat_hours, feat_depth, output_dir, date_str):
    """Create a single-panel 24h echogram from cleaned data."""
    config = ECHOGRAM_PRESETS['publication']
    sv_min, sv_max = config['sv_range']
    depth_min, depth_max = config['depth_range']
    cmap = get_colormap(config['colormap'])

    fig, ax = plt.subplots(figsize=config['figsize'])

    # Fill impulse-noise NaN pings for display (data file unchanged)
    sv_data = fill_nan_pings(sv_data)

    sv_plot = np.clip(sv_data.T, sv_min, sv_max)
    sv_plot = np.where(np.isnan(sv_plot), sv_min, sv_plot)

    im = ax.pcolormesh(hours, depths, sv_plot, cmap=cmap, shading='auto', vmin=sv_min, vmax=sv_max)

    # Overlay center of mass (if available)
    if feat_hours is not None and feat_depth is not None:
        ax.plot(feat_hours, feat_depth, 'r-', linewidth=2, alpha=0.9)

    ax.set_ylim(depth_max, depth_min)
    ax.set_xlim(0, 24)
    ax.set_xlabel('Time (local)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)

    # Format date for title
    date_title = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    dvm_label = " with DVM" if feat_hours is not None else ""
    ax.set_title(f'MALASPINA Atlantic - {date_title} - 38 kHz Echogram{dvm_label}',
                 fontsize=13)

    cbar = plt.colorbar(im, ax=ax, label='Sv (dB)', pad=0.02)
    cbar.set_ticks([-90, -80, -70, -60, -50])

    # Time axis formatting (matching Klevjer et al.)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '00:00'])

    plt.tight_layout()

    output_path = output_dir / f"echogram_24h_{date_str}.png"
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate 24-hour echogram visualization from cleaned data."
    )
    parser.add_argument(
        'date', nargs='?', default='20110126',
        help='Date string in YYYYMMDD format (default: 20110126)'
    )
    args = parser.parse_args()

    ensure_dirs()
    output_dir = OUTPUT_VIZ
    date_str = args.date

    create_validation_visualization(date_str, output_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Date: {date_str}")
    print("Check output/visualizations/ for generated echograms.")
