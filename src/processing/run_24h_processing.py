#!/usr/bin/env python3
"""
================================================================================
24-HOUR CLEANING PIPELINE
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)

Orchestrates the echopype_main.py cleaning pipeline across all raw files for a
given date, then concatenates the cleaned Sv data into a single 24-hour NetCDF.

This is the entry point for the cleaned data pipeline:
    raw files -> run_24h_processing.py -> cleaned_Sv_24h_{date}.nc

Downstream scripts (visualization, extraction) should read from the cleaned
NetCDF rather than re-opening raw files independently.

Usage:
    python src/processing/run_24h_processing.py              # default date
    python src/processing/run_24h_processing.py 20110126     # explicit date
================================================================================
"""

import sys
from pathlib import Path
import argparse
import logging
import tempfile
import time

# Add project root to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MALASPINA_LEG2, OUTPUT_DATA, ENV_PARAMS, NOISE_PARAMS, ensure_dirs

import numpy as np
import xarray as xr

from echopype_main import MalaspinaProcessor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run_24h_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_38khz_channel(ds_Sv):
    """Return the integer index of the 38 kHz channel in a calibrated dataset."""
    channels = ds_Sv.channel.values
    for idx, ch in enumerate(channels):
        if '38' in str(ch).lower():
            return idx
    return None


def process_single_file(raw_file, output_dir_tmp):
    """
    Run calibration + noise removal on a single raw file and return the
    cleaned Sv for the 38 kHz channel, along with its depth axis.

    Returns
    -------
    ds_clean_38 : xr.Dataset or None
        Cleaned Sv dataset for 38 kHz (single channel), or None on failure.
    """
    processor = MalaspinaProcessor(str(raw_file), output_dir=str(output_dir_tmp))

    # 1. Load raw data
    processor.load_raw_data()

    # 2. Calibrate Sv
    processor.calibrate_data(custom_env_params=ENV_PARAMS)

    # 3. Find 38 kHz channel index
    ch_idx = find_38khz_channel(processor.ds_Sv)
    if ch_idx is None:
        raise ValueError(f"No 38 kHz channel found in {raw_file.name}")

    # 4. Noise removal (background + impulse)
    processor.detect_and_remove_artifacts()

    # 5. Extract cleaned Sv for 38 kHz only (no MVBS, no save)
    ds_clean_38 = processor.get_cleaned_sv(channel_idx=ch_idx)

    return ds_clean_38


def build_cleaned_24h(date_str='20110126', data_dir=None):
    """
    Process all raw files for *date_str*, concatenate cleaned 38 kHz Sv,
    and save as a single NetCDF file.

    Parameters
    ----------
    date_str : str
        Date in YYYYMMDD format.
    data_dir : Path, optional
        Directory containing .raw files. Defaults to MALASPINA_LEG2.

    Returns
    -------
    Path
        Path to the saved NetCDF file.
    """
    if data_dir is None:
        data_dir = MALASPINA_LEG2

    ensure_dirs()

    # Discover raw files for the given date, sorted chronologically
    pattern = f"D{date_str}-T*.raw"
    raw_files = sorted(data_dir.glob(pattern))

    if not raw_files:
        logger.error(f"No files found matching {pattern} in {data_dir}")
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")

    logger.info("=" * 70)
    logger.info("24-HOUR CLEANING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Date:       {date_str}")
    logger.info(f"Data dir:   {data_dir}")
    logger.info(f"Files:      {len(raw_files)}")
    logger.info(f"Env params: T={ENV_PARAMS['temperature']}°C, "
                f"S={ENV_PARAMS['salinity']} PSU, P={ENV_PARAMS['pressure']} dbar")
    logger.info("=" * 70)

    cleaned_datasets = []
    failed_files = []
    t_start = time.time()

    # Use a temporary directory for per-file processor artefacts (calibration
    # JSONs written by MalaspinaProcessor). Automatically cleaned up after
    # processing — the calibration values are already applied in the NetCDF.
    with tempfile.TemporaryDirectory(prefix="oceanica_") as tmp_dir:
        for i, raw_file in enumerate(raw_files):
            # Progress indicator every 10 files
            if i % 10 == 0:
                logger.info(f"[{i+1}/{len(raw_files)}] Processing {raw_file.name} ...")

            try:
                ds_clean = process_single_file(raw_file, tmp_dir)
                if ds_clean is not None:
                    cleaned_datasets.append(ds_clean)
            except Exception as e:
                logger.error(f"FAILED {raw_file.name}: {e}")
                failed_files.append((raw_file.name, str(e)))
                continue

    if not cleaned_datasets:
        raise RuntimeError("All files failed processing — no cleaned data produced.")

    # ------------------------------------------------------------------
    # Concatenate along ping_time
    # ------------------------------------------------------------------
    logger.info("Concatenating cleaned datasets along ping_time ...")

    # Check which Sv variables are available
    has_sv_corrected = 'Sv_corrected' in cleaned_datasets[0].data_vars

    # Build a uniform depth axis from the first valid dataset.
    # Each per-file dataset may have a slightly different echo_range/depth shape;
    # we interpolate to a common depth grid.
    # Determine depth variable
    if 'depth' in cleaned_datasets[0].data_vars or 'depth' in cleaned_datasets[0].coords:
        depth_var = 'depth'
    else:
        depth_var = 'echo_range'

    # Extract a representative depth axis (first file, first valid ping)
    ref_ds = cleaned_datasets[0]
    if depth_var in ref_ds.data_vars:
        depth_raw = ref_ds[depth_var].values
        if depth_raw.ndim == 2:  # (ping_time, range_sample)
            # Take first ping with non-NaN values
            for p in range(depth_raw.shape[0]):
                if not np.all(np.isnan(depth_raw[p])):
                    depth_axis = depth_raw[p]
                    break
            else:
                depth_axis = np.nanmean(depth_raw, axis=0)
        else:
            depth_axis = depth_raw
    else:
        depth_raw = ref_ds[depth_var].values
        if depth_raw.ndim == 2:
            for p in range(depth_raw.shape[0]):
                if not np.all(np.isnan(depth_raw[p])):
                    depth_axis = depth_raw[p]
                    break
            else:
                depth_axis = np.nanmean(depth_raw, axis=0)
        else:
            depth_axis = depth_raw

    # Trim NaN tail from depth axis
    valid_mask = ~np.isnan(depth_axis)
    if valid_mask.any():
        last_valid = np.where(valid_mask)[0][-1]
        depth_axis = depth_axis[:last_valid + 1]
    n_depth = len(depth_axis)

    # Collect BOTH Sv (raw calibrated, impulse-masked) and Sv_corrected
    # (background noise subtracted + SNR thresholded) from each file.
    # Visualization uses Sv (shows full water column structure like reference papers),
    # while quantitative analysis can use Sv_corrected.
    all_sv_raw = []
    all_sv_corrected = []
    all_times = []

    for ds in cleaned_datasets:
        # Raw calibrated Sv (impulse pings already masked as NaN)
        sv_raw = ds['Sv'].values  # (ping_time, range_sample)
        # Noise-corrected Sv (background subtracted + SNR thresholded)
        if has_sv_corrected:
            sv_corr = ds['Sv_corrected'].values
        else:
            sv_corr = sv_raw  # Fallback: identical

        times = ds.ping_time.values

        # Handle single-ping edge case
        if sv_raw.ndim == 1:
            sv_raw = sv_raw[np.newaxis, :]
            sv_corr = sv_corr[np.newaxis, :]

        all_sv_raw.append(sv_raw[:, :n_depth])
        all_sv_corrected.append(sv_corr[:, :n_depth])
        all_times.append(times)

    sv_raw_combined = np.vstack(all_sv_raw)
    sv_corr_combined = np.vstack(all_sv_corrected)
    times_combined = np.concatenate(all_times)

    # Sort by time (should already be sorted, but be safe)
    sort_idx = np.argsort(times_combined)
    sv_raw_combined = sv_raw_combined[sort_idx]
    sv_corr_combined = sv_corr_combined[sort_idx]
    times_combined = times_combined[sort_idx]

    # ------------------------------------------------------------------
    # Build output Dataset
    # ------------------------------------------------------------------
    ds_out = xr.Dataset(
        {
            'Sv': (['ping_time', 'depth'], sv_raw_combined),
            'Sv_corrected': (['ping_time', 'depth'], sv_corr_combined),
        },
        coords={
            'ping_time': times_combined,
            'depth': depth_axis,
        },
        attrs={
            'description': f'Cleaned 24h Sv data for {date_str} (38 kHz)',
            'date': date_str,
            'frequency_hz': 38000,
            'num_files_processed': len(cleaned_datasets),
            'num_files_failed': len(failed_files),
            'total_pings': len(times_combined),
            'depth_samples': n_depth,
            'env_temperature_C': ENV_PARAMS['temperature'],
            'env_salinity_PSU': ENV_PARAMS['salinity'],
            'env_pressure_dbar': ENV_PARAMS['pressure'],
            'noise_removal': 'De Robertis & Higginbottom (2007) background + Ryan et al. (2015) impulse',
            'noise_ping_num': NOISE_PARAMS['ping_num'],
            'noise_range_sample_num': NOISE_PARAMS['range_sample_num'],
            'noise_SNR_threshold_dB': NOISE_PARAMS['SNR_threshold'],
            'created': str(np.datetime64('now')),
        }
    )

    ds_out['Sv'].attrs = {
        'long_name': 'Volume backscattering strength (calibrated, impulse-masked)',
        'units': 'dB re 1 m^-1',
        'description': 'Raw calibrated Sv with impulse noise pings set to NaN. '
                       'Use for echogram visualization (shows full water column structure).',
    }
    ds_out['Sv_corrected'].attrs = {
        'long_name': 'Volume backscattering strength (noise-corrected)',
        'units': 'dB re 1 m^-1',
        'description': 'Background noise subtracted + SNR thresholded Sv. '
                       'Values below SNR threshold set to NaN. '
                       'Use for quantitative analysis (biomass estimation).',
    }
    ds_out['depth'].attrs = {
        'long_name': 'Depth',
        'units': 'm',
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = OUTPUT_DATA / f"cleaned_Sv_24h_{date_str}.nc"
    ds_out.to_netcdf(output_path)
    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Files processed:  {len(cleaned_datasets)} / {len(raw_files)}")
    logger.info(f"Files failed:     {len(failed_files)}")
    logger.info(f"Total pings:      {len(times_combined)}")
    logger.info(f"Time range:       {times_combined[0]} → {times_combined[-1]}")
    logger.info(f"Depth range:      0 – {depth_axis[-1]:.1f} m  ({n_depth} samples)")
    logger.info(f"Output file:      {output_path}")
    logger.info(f"File size:        {output_path.stat().st_size / 1e6:.1f} MB")
    logger.info(f"Elapsed:          {elapsed:.0f} s")

    if failed_files:
        logger.warning("Failed files:")
        for name, err in failed_files:
            logger.warning(f"  {name}: {err}")

    logger.info("=" * 70)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 24-hour cleaning pipeline on MALASPINA raw data."
    )
    parser.add_argument(
        'date', nargs='?', default='20110126',
        help='Date string in YYYYMMDD format (default: 20110126)'
    )
    args = parser.parse_args()

    build_cleaned_24h(date_str=args.date)
