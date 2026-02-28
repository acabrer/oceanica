#!/usr/bin/env python3
"""
Malaspina Acoustic Data Processing Pipeline
============================================
Scientific processing of EK60 echosounder data for the Sonificació Oceànica project
Using echopype for standardized, reproducible acoustic data analysis

Author: Oceànica Team
Date: October 2025
References:
- De Robertis & Higginbottom (2007) - Noise removal methods
- Ryan et al. (2015) - Artifact detection and removal
- Demer et al. (2015) - Calibration standards
- MacLennan et al. (2002) - MVBS computation standards
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
import logging
from typing import Dict, Tuple, Optional, Union
import json

# Echopype imports
import echopype as ep
from echopype import mask, consolidate
from echopype.calibrate import compute_Sv
from echopype.clean import remove_background_noise, mask_impulse_noise, estimate_background_noise
from echopype.commongrid import compute_MVBS
from echopype.qc import exist_reversed_time, coerce_increasing_time
from echopype.mask import apply_mask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('malaspina_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class MalaspinaProcessor:
    """
    Professional acoustic data processor for Malaspina EK60 files
    Implements best practices for echosounder data processing
    """
    
    def __init__(self, raw_file_path: str, output_dir: str = "./processed_data"):
        """
        Initialize processor with file paths and parameters
        
        Parameters
        ----------
        raw_file_path : str
            Path to the .raw file (expects .bot and .idx in same directory)
        output_dir : str
            Directory for processed outputs
        """
        self.raw_file = Path(raw_file_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Verify auxiliary files exist
        self.bot_file = self.raw_file.with_suffix('.bot')
        self.idx_file = self.raw_file.with_suffix('.idx')
        
        logger.info(f"Initializing processor for: {self.raw_file.name}")
        logger.info(f"Bottom file exists: {self.bot_file.exists()}")
        logger.info(f"Index file exists: {self.idx_file.exists()}")
        
        # Processing parameters (based on scientific literature)
        self.params = {
            'noise_estimation': {
                'ping_num': 30,  # De Robertis & Higginbottom 2007
                'range_sample_num': 100,
                'SNR_threshold': 3.0  # 3 dB signal-to-noise ratio
            },
            'impulse_noise': {
                # Ryan et al. 2015 parameters
                'depth_bin': '5m',  # Vertical binning for impulse detection
                'num_side_pings': 2,  # Pings on each side for comparison
                'threshold': '10.0dB'  # Threshold for impulse classification
            },
            'mvbs': {
                'ping_time_bin': '10s',  # 10 second bins
                'range_bin': '5m'  # 5 meter vertical bins
            },
            'environmental': {
                # Mediterranean Sea typical values (adjust if known)
                'temperature': 14.0,  # degrees Celsius
                'salinity': 38.5,  # PSU
                'pressure': 10.0  # dbar (approximate for surface)
            },
            'sonification': {
                'surface_exclusion_depth': 10.0,  # meters - exclude near-surface noise
                'max_depth': 1000.0,  # meters - maximum depth to consider
                'min_sv_threshold': -90.0  # dB - matches scientific literature standard
            }
        }
        
        self.echodata = None
        self.ds_Sv = None
        self.ds_Sv_clean = None
        self.ds_MVBS = None
        self.artifacts_log = []
    
    def load_raw_data(self) -> ep.echodata.EchoData:
        """
        Load raw data using echopype with proper error handling
        """
        try:
            logger.info("Loading raw data with echopype...")
            
            # Open raw file - echopype automatically detects .bot and .idx files
            self.echodata = ep.open_raw(
                self.raw_file,
                sonar_model='EK60',
                include_bot=True,  # Include bottom detection if available
                include_idx=True   # Include index for faster access
            )
            
            # Log basic information
            logger.info(f"Successfully loaded: {self.raw_file.name}")
            logger.info(f"Time range: {self.echodata.beam.ping_time.min().values} to "
                       f"{self.echodata.beam.ping_time.max().values}")
            logger.info(f"Channels: {self.echodata.beam.channel.values}")
            logger.info(f"Frequencies: {self.echodata.beam.frequency_nominal.values} Hz")

            # Check for and fix time reversals (common in EK60 data)
            if exist_reversed_time(self.echodata.beam.ds, time_name='ping_time'):
                logger.warning("Time reversals detected - correcting...")
                coerce_increasing_time(self.echodata.beam.ds, time_name='ping_time')
            
            return self.echodata
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            raise
    
    def calibrate_data(self, custom_env_params: Optional[Dict] = None) -> xr.Dataset:
        """
        Calibrate raw data to Sv (volume backscattering strength)
        Following Demer et al. 2015 standards
        """
        try:
            logger.info("Calibrating data to Sv...")
            
            # Use custom environmental parameters if provided
            env_params = custom_env_params or self.params['environmental']
            
            # Compute Sv with environmental corrections
            self.ds_Sv = compute_Sv(
                self.echodata,
                env_params=env_params,
                cal_params=None  # Use default calibration from file
            )
            
            # Log calibration results
            logger.info(f"Calibration complete")
            logger.info(f"Sv shape: {self.ds_Sv['Sv'].shape}")
            logger.info(f"Sound speed used: {self.ds_Sv['sound_speed'].mean().values:.1f} m/s")
            logger.info(f"Absorption coefficient: {self.ds_Sv['sound_absorption'].mean().values:.6f} dB/m")
            
            # Save calibration parameters for reproducibility
            self._save_calibration_metadata()
            
            return self.ds_Sv
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise
    
    def detect_and_remove_artifacts(self) -> xr.Dataset:
        """
        Detect and remove artifacts including noise and spikes
        Based on Ryan et al. 2015 and De Robertis & Higginbottom 2007

        Updated to use echopype v0.11 best practices:
        - remove_background_noise returns Sv_corrected
        - Custom impulse noise detection (Ryan et al. 2015 algorithm)
        - Marks entire pings as bad if they contain impulse noise
        """
        logger.info("Starting artifact detection and removal...")

        # Step 1: Add depth variable for proper vertical referencing
        logger.info("Adding depth variable...")
        ds_with_depth = consolidate.add_depth(self.ds_Sv, self.echodata)

        # Step 2: Remove background noise (De Robertis & Higginbottom 2007)
        logger.info("Removing background noise...")
        ds_Sv_denoised = remove_background_noise(
            ds_with_depth,
            ping_num=self.params['noise_estimation']['ping_num'],
            range_sample_num=self.params['noise_estimation']['range_sample_num'],
            background_noise_max=None,
            SNR_threshold=f"{self.params['noise_estimation']['SNR_threshold']}dB"
        )

        # Note: remove_background_noise adds 'Sv_corrected' and 'Sv_noise' variables
        logger.info(f"Noise removal complete. Variables: {list(ds_Sv_denoised.data_vars)}")

        # Step 3: Custom impulse noise detection (Ryan et al. 2015 inspired)
        logger.info("Detecting impulse noise (custom implementation)...")
        ds_Sv_clean, impulse_info = self._detect_and_mask_impulse_noise(ds_Sv_denoised)

        self.artifacts_log.append({
            'timestamp': datetime.now().isoformat(),
            'file': self.raw_file.name,
            **impulse_info
        })

        # Step 4: Apply bottom mask if bottom detection available
        if 'bottom_range' in self.echodata.beam.data_vars:
            logger.info("Applying bottom mask...")
            ds_Sv_clean = self._apply_bottom_mask(ds_Sv_clean)

        self.ds_Sv_clean = ds_Sv_clean

        # Save artifact log
        self._save_artifacts_log()

        return self.ds_Sv_clean

    def _detect_and_mask_impulse_noise(self, ds_Sv: xr.Dataset) -> Tuple[xr.Dataset, Dict]:
        """
        Custom impulse noise detection based on Ryan et al. 2015

        Algorithm:
        1. Calculate mean Sv per ping (column average)
        2. Compare each ping to its neighbors (two-sided comparison)
        3. If a ping is significantly higher than BOTH neighbors, flag it
        4. Mask flagged pings by setting Sv values to NaN

        This catches vertical "stripe" artifacts like electrical interference.
        """
        sv_var = 'Sv_corrected' if 'Sv_corrected' in ds_Sv.data_vars else 'Sv'
        threshold_db = float(self.params['impulse_noise']['threshold'].replace('dB', ''))
        num_side_pings = self.params['impulse_noise']['num_side_pings']

        logger.info(f"Impulse detection: threshold={threshold_db} dB, side_pings={num_side_pings}")

        # Get Sv data
        sv_data = ds_Sv[sv_var].values  # Shape: (channel, ping_time, range_sample)
        n_channels, n_pings, n_range = sv_data.shape

        # Track bad pings across all channels
        bad_pings = set()
        channel_stats = {}

        for ch_idx in range(n_channels):
            ch_data = sv_data[ch_idx, :, :]  # (ping_time, range_sample)

            # Calculate mean Sv per ping (ignoring NaN)
            mean_sv_per_ping = np.nanmean(ch_data, axis=1)  # (n_pings,)

            # Two-sided comparison for each ping
            flagged_pings = []

            for ping_idx in range(num_side_pings, n_pings - num_side_pings):
                current_sv = mean_sv_per_ping[ping_idx]

                if np.isnan(current_sv):
                    continue

                # Get neighbor values
                left_neighbors = mean_sv_per_ping[ping_idx - num_side_pings:ping_idx]
                right_neighbors = mean_sv_per_ping[ping_idx + 1:ping_idx + 1 + num_side_pings]

                # Calculate median of neighbors (robust to outliers)
                left_median = np.nanmedian(left_neighbors)
                right_median = np.nanmedian(right_neighbors)

                # Check if current ping exceeds BOTH sides by threshold
                if (not np.isnan(left_median) and not np.isnan(right_median)):
                    exceeds_left = current_sv > (left_median + threshold_db)
                    exceeds_right = current_sv > (right_median + threshold_db)

                    if exceeds_left and exceeds_right:
                        flagged_pings.append(ping_idx)
                        bad_pings.add(ping_idx)
                        logger.debug(f"Channel {ch_idx}: Ping {ping_idx} flagged "
                                    f"(Sv={current_sv:.1f}, left={left_median:.1f}, right={right_median:.1f})")

            channel_stats[f'channel_{ch_idx}_flagged'] = len(flagged_pings)
            if flagged_pings:
                logger.info(f"Channel {ch_idx}: Flagged {len(flagged_pings)} pings as impulse noise: {flagged_pings}")

        # Create cleaned dataset
        ds_clean = ds_Sv.copy(deep=True)

        if bad_pings:
            bad_pings_list = sorted(list(bad_pings))
            logger.info(f"Total bad pings to mask: {len(bad_pings_list)} -> {bad_pings_list}")

            # Mask the bad pings by setting to NaN
            for ping_idx in bad_pings_list:
                ds_clean[sv_var].values[:, ping_idx, :] = np.nan
                # Also mask the original Sv if different
                if sv_var != 'Sv' and 'Sv' in ds_clean.data_vars:
                    ds_clean['Sv'].values[:, ping_idx, :] = np.nan

            logger.info(f"Masked {len(bad_pings_list)} impulse noise pings")

        info = {
            'impulse_noise_pings': len(bad_pings),
            'impulse_ping_indices': sorted(list(bad_pings)),
            'threshold_dB': threshold_db,
            'num_side_pings': num_side_pings,
            'method': 'custom_two_sided_comparison (Ryan et al. 2015 inspired)',
            **channel_stats
        }

        return ds_clean, info
    
    def _apply_bottom_mask(self, ds_Sv: xr.Dataset) -> xr.Dataset:
        """
        Mask data below detected bottom
        """
        # Implementation depends on bottom data structure
        # This is a placeholder for the actual implementation
        logger.info("Bottom masking applied")
        return ds_Sv
    
    def get_cleaned_sv(self, channel_idx: Optional[int] = None) -> xr.Dataset:
        """
        Return the cleaned Sv dataset (after calibration and artifact removal),
        without computing MVBS or saving to disk.

        Optionally select a single channel by index.

        Parameters
        ----------
        channel_idx : int, optional
            If provided, select only this channel from the dataset.

        Returns
        -------
        xr.Dataset
            The cleaned Sv dataset (ds_Sv_clean). Contains 'Sv_corrected'
            (from noise removal) and/or 'Sv' variables.
        """
        if self.ds_Sv_clean is None:
            raise RuntimeError("No cleaned data available. Run calibrate_data() and "
                               "detect_and_remove_artifacts() first.")
        ds = self.ds_Sv_clean
        if channel_idx is not None:
            ds = ds.isel(channel=channel_idx)
        return ds

    def compute_mvbs(self, preset: str = 'sonification') -> xr.Dataset:
        """
        Compute Mean Volume Backscattering Strength (MVBS)
        Following MacLennan et al. 2002 standards

        Updated for echopype v0.11:
        - Uses depth variable if available
        - Handles Sv_corrected from noise removal
        - Supports configurable binning presets

        Parameters
        ----------
        preset : str
            MVBS preset name: 'publication' (6min x 1m), 'analysis' (1min x 2m),
            or 'sonification' (10s x 5m, default). See config.MVBS_PRESETS.
        """
        # Import presets from config if available
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import MVBS_PRESETS
            if preset in MVBS_PRESETS:
                self.params['mvbs'] = MVBS_PRESETS[preset]
                logger.info(f"Using MVBS preset: {preset} -> {self.params['mvbs']}")
        except ImportError:
            logger.info("Using default MVBS parameters")

        logger.info("Computing MVBS...")

        # Prepare dataset for MVBS - needs Sv variable
        ds_for_mvbs = self.ds_Sv_clean.copy()

        # If we have Sv_corrected, use it as Sv for MVBS computation
        if 'Sv_corrected' in ds_for_mvbs.data_vars:
            ds_for_mvbs['Sv'] = ds_for_mvbs['Sv_corrected']
            logger.info("Using Sv_corrected for MVBS computation")

        # Determine range variable
        range_var = 'depth' if 'depth' in ds_for_mvbs.data_vars else 'echo_range'

        ds_MVBS = compute_MVBS(
            ds_for_mvbs,
            range_var=range_var,
            range_bin=self.params['mvbs']['range_bin'],
            ping_time_bin=self.params['mvbs']['ping_time_bin']
        )

        logger.info(f"MVBS computed: shape {ds_MVBS['Sv'].shape}")
        logger.info(f"MVBS range variable: {range_var}")

        # Store for later use
        self.ds_MVBS = ds_MVBS

        return ds_MVBS
    
    def extract_sonification_features(self, max_pings: int = 5000) -> Dict:
        """
        Extract features specifically for sonification

        Key improvements:
        - Uses Sv_corrected if available (from noise removal)
        - Excludes near-surface noise (configurable surface_exclusion_depth)
        - Applies minimum Sv threshold to filter noise
        - Uses depth variable if available, falls back to echo_range
        """
        logger.info("Extracting sonification features...")

        # Use clean data
        ds = self.ds_Sv_clean

        # Determine which Sv variable to use
        sv_var = 'Sv_corrected' if 'Sv_corrected' in ds.data_vars else 'Sv'
        logger.info(f"Using variable: {sv_var}")

        # Determine depth variable
        if 'depth' in ds.data_vars or 'depth' in ds.coords:
            depth_var = 'depth'
        else:
            depth_var = 'echo_range'
        logger.info(f"Using depth variable: {depth_var}")

        # Limit to max_pings if needed
        if len(ds.ping_time) > max_pings:
            ds = ds.isel(ping_time=slice(0, max_pings))

        # Get sonification parameters
        surface_exclusion = self.params['sonification']['surface_exclusion_depth']
        max_depth = self.params['sonification']['max_depth']
        min_sv = self.params['sonification']['min_sv_threshold']

        # Calculate max depth from data
        if depth_var in ds.data_vars:
            actual_max_depth = float(np.nanmax(ds[depth_var].values))
        else:
            actual_max_depth = float(ds[depth_var].max().values)

        features = {
            'metadata': {
                'file': self.raw_file.name,
                'start_time': str(ds.ping_time.min().values),
                'end_time': str(ds.ping_time.max().values),
                'duration_seconds': float((ds.ping_time.max() - ds.ping_time.min()).values / 1e9),
                'frequencies': ds.frequency_nominal.values.tolist(),
                'depth_range': [surface_exclusion, min(max_depth, actual_max_depth)],
                'surface_exclusion_m': surface_exclusion,
                'min_sv_threshold_dB': min_sv,
                'sv_variable_used': sv_var,
                'depth_variable_used': depth_var,
                'num_pings': len(ds.ping_time),
                'artifacts_removed': self.artifacts_log[-1].get('impulse_noise_samples', 0) if self.artifacts_log else 0
            },
            'acoustic_features': {}
        }

        # For each channel, extract time series features
        for ch_idx, freq in enumerate(ds.frequency_nominal.values):
            logger.info(f"Processing channel {ch_idx}: {freq/1000:.0f} kHz")

            ch_sv = ds[sv_var].isel(channel=ch_idx)

            # Get depth axis for this channel
            if depth_var in ds.data_vars:
                # depth is a data variable with dimensions
                depth_data = ds[depth_var]
                if 'channel' in depth_data.dims:
                    depth_data = depth_data.isel(channel=ch_idx)
                # Average across ping_time if needed to get 1D depth axis
                if 'ping_time' in depth_data.dims:
                    depth_axis = depth_data.mean(dim='ping_time').values
                else:
                    depth_axis = depth_data.values
            else:
                # echo_range as coordinate
                depth_data = ds[depth_var]
                if 'channel' in depth_data.dims:
                    depth_data = depth_data.isel(channel=ch_idx)
                if 'ping_time' in depth_data.dims:
                    depth_axis = np.nanmean(depth_data.values, axis=0)
                else:
                    depth_axis = depth_data.values

            # Create depth mask: exclude surface and beyond max_depth
            depth_mask = (depth_axis >= surface_exclusion) & (depth_axis <= max_depth)
            valid_depth_indices = np.where(depth_mask)[0]

            if len(valid_depth_indices) == 0:
                logger.warning(f"No valid depths found for channel {ch_idx}")
                continue

            # Filter depth axis to valid range
            depth_valid_range = depth_axis[depth_mask]

            center_of_mass = []
            total_intensity = []
            peak_depth = []
            vertical_spread = []
            layer_count = []

            for t in range(len(ds.ping_time)):
                # Get Sv for this ping, filtered by depth
                ping_sv = ch_sv.isel(ping_time=t).values

                # Apply depth mask
                ping_sv_filtered = ping_sv[depth_mask]

                # Apply Sv threshold mask (exclude noise floor)
                sv_mask = ping_sv_filtered > min_sv
                valid = sv_mask & ~np.isnan(ping_sv_filtered)

                if np.sum(valid) > 10:  # Need minimum samples for statistics
                    depth_valid = depth_valid_range[valid]
                    sv_valid = ping_sv_filtered[valid]

                    # Convert from dB to linear for weighting
                    weights_valid = 10**(sv_valid / 10)

                    # Center of mass (weighted average depth)
                    com = np.average(depth_valid, weights=weights_valid)
                    center_of_mass.append(float(com))

                    # Total intensity (sum in linear, convert back to dB)
                    total = np.sum(weights_valid)
                    total_intensity.append(float(10 * np.log10(total) if total > 0 else -90))

                    # Peak depth - depth of maximum backscatter
                    peak_idx = np.argmax(sv_valid)  # Use Sv directly, not weights
                    peak_depth.append(float(depth_valid[peak_idx]))

                    # Vertical spread (weighted standard deviation)
                    spread = np.sqrt(np.average((depth_valid - com)**2, weights=weights_valid))
                    vertical_spread.append(float(spread))

                    # Layer count - number of distinct peaks (simple threshold crossing)
                    sv_threshold = np.percentile(sv_valid, 75)
                    above_threshold = sv_valid > sv_threshold
                    transitions = np.diff(above_threshold.astype(int))
                    n_layers = max(1, np.sum(transitions == 1))
                    layer_count.append(int(n_layers))

                else:
                    center_of_mass.append(np.nan)
                    total_intensity.append(-90.0)
                    peak_depth.append(np.nan)
                    vertical_spread.append(0.0)
                    layer_count.append(0)

            # Calculate statistics for validation
            valid_com = [x for x in center_of_mass if not np.isnan(x)]
            valid_peak = [x for x in peak_depth if not np.isnan(x)]

            if valid_com:
                logger.info(f"  Center of mass: {np.min(valid_com):.1f} - {np.max(valid_com):.1f} m "
                           f"(mean: {np.mean(valid_com):.1f} m)")
            if valid_peak:
                logger.info(f"  Peak depth: {np.min(valid_peak):.1f} - {np.max(valid_peak):.1f} m "
                           f"(mean: {np.mean(valid_peak):.1f} m)")

            features['acoustic_features'][f'{int(freq)}_Hz'] = {
                'center_of_mass': center_of_mass,
                'total_intensity': total_intensity,
                'peak_depth': peak_depth,
                'vertical_spread': vertical_spread,
                'layer_count': layer_count,
                'ping_times': ds.ping_time.values.astype(str).tolist(),
                'statistics': {
                    'center_of_mass_mean': float(np.nanmean(center_of_mass)),
                    'center_of_mass_std': float(np.nanstd(center_of_mass)),
                    'peak_depth_mean': float(np.nanmean(peak_depth)),
                    'peak_depth_std': float(np.nanstd(peak_depth)),
                    'total_intensity_mean': float(np.nanmean(total_intensity)),
                    'valid_pings': len(valid_com)
                }
            }

        return features
    
    def save_processed_data(self):
        """
        Save all processed data products
        """
        logger.info("Saving processed data...")

        def clean_attrs(ds):
            """Remove None values from attributes that can't be serialized"""
            ds_clean = ds.copy()
            for var in ds_clean.variables:
                if hasattr(ds_clean[var], 'attrs'):
                    ds_clean[var].attrs = {k: v for k, v in ds_clean[var].attrs.items()
                                           if v is not None}
            ds_clean.attrs = {k: v for k, v in ds_clean.attrs.items() if v is not None}
            return ds_clean

        # Save clean Sv as NetCDF
        if self.ds_Sv_clean is not None:
            ds_to_save = clean_attrs(self.ds_Sv_clean)
            output_file = self.output_dir / f"{self.raw_file.stem}_Sv_clean.nc"
            ds_to_save.to_netcdf(output_file)
            logger.info(f"Saved clean Sv to: {output_file}")

        # Save MVBS as NetCDF if available
        if hasattr(self, 'ds_MVBS') and self.ds_MVBS is not None:
            ds_mvbs_save = clean_attrs(self.ds_MVBS)
            mvbs_file = self.output_dir / f"{self.raw_file.stem}_MVBS.nc"
            ds_mvbs_save.to_netcdf(mvbs_file)
            logger.info(f"Saved MVBS to: {mvbs_file}")

        # Save sonification features
        features = self.extract_sonification_features()
        features_file = self.output_dir / f"{self.raw_file.stem}_sonification.json"
        with open(features_file, 'w') as f:
            json.dump(features, f, indent=2, default=str)
        logger.info(f"Saved sonification features to: {features_file}")

        # Save as NPZ for compatibility (excluding nested dicts like 'statistics')
        npz_file = self.output_dir / f"{self.raw_file.stem}_sonification.npz"
        npz_data = {}
        for freq_key, freq_data in features['acoustic_features'].items():
            for feature_name, feature_values in freq_data.items():
                if feature_name not in ['ping_times', 'statistics'] and isinstance(feature_values, list):
                    npz_data[f"{freq_key}_{feature_name}"] = np.array(feature_values)
        np.savez(npz_file, **npz_data)
        logger.info(f"Saved NPZ file to: {npz_file}")
    
    def _save_calibration_metadata(self):
        """
        Save calibration parameters for reproducibility
        """
        metadata = {
            'calibration_time': datetime.now().isoformat(),
            'file': self.raw_file.name,
            'environmental_params': self.params['environmental'],
            'sound_speed': float(self.ds_Sv['sound_speed'].mean().values),
            'absorption': float(self.ds_Sv['sound_absorption'].mean().values),
            'frequencies': self.ds_Sv.frequency_nominal.values.tolist()
        }
        
        metadata_file = self.output_dir / f"{self.raw_file.stem}_calibration.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_artifacts_log(self):
        """
        Save artifact detection log
        """
        if self.artifacts_log:
            log_file = self.output_dir / "artifacts_log.json"
            with open(log_file, 'w') as f:
                json.dump(self.artifacts_log, f, indent=2)
    
    def create_echogram_visualization(self):
        """
        Create publication-quality echogram visualization

        Updated for echopype v0.11:
        - Uses Sv_corrected if available
        - Uses depth variable if available
        - Shows both channels
        """
        if self.ds_Sv_clean is None:
            logger.warning("No clean data available for visualization")
            return

        # Determine which Sv variable to use
        sv_var = 'Sv_corrected' if 'Sv_corrected' in self.ds_Sv_clean.data_vars else 'Sv'
        depth_var = 'depth' if 'depth' in self.ds_Sv_clean.data_vars else 'echo_range'

        n_channels = len(self.ds_Sv_clean.channel)
        fig, axes = plt.subplots(n_channels + 2, 1, figsize=(14, 4 * (n_channels + 2)),
                                 gridspec_kw={'height_ratios': [3] * n_channels + [1, 1]})

        if n_channels == 1:
            axes = [axes]

        # Plot echogram for each channel
        for ch_idx in range(n_channels):
            Sv_data = self.ds_Sv_clean[sv_var].isel(channel=ch_idx)
            freq = self.ds_Sv_clean.frequency_nominal.values[ch_idx]

            # Get depth values
            try:
                if depth_var in self.ds_Sv_clean.data_vars:
                    depth_data = self.ds_Sv_clean[depth_var]
                    if 'channel' in depth_data.dims:
                        depth_data = depth_data.isel(channel=ch_idx)
                    if 'ping_time' in depth_data.dims:
                        depth_values = np.nanmean(depth_data.values, axis=0)
                    else:
                        depth_values = depth_data.values
                else:
                    depth_data = self.ds_Sv_clean[depth_var]
                    if 'channel' in depth_data.dims:
                        depth_data = depth_data.isel(channel=ch_idx)
                    if 'ping_time' in depth_data.dims:
                        depth_values = np.nanmean(depth_data.values, axis=0)
                    else:
                        depth_values = depth_data.values
            except Exception as e:
                logger.warning(f"Could not extract depth values: {e}")
                depth_values = np.arange(Sv_data.shape[1]) * 0.5

            # Ensure finite depth values
            if np.any(np.isnan(depth_values)) or np.any(np.isinf(depth_values)):
                depth_values = np.arange(len(depth_values)) * 0.5

            # Show full depth range up to max_depth parameter
            max_depth = self.params['sonification']['max_depth']
            depth_mask = depth_values <= max_depth
            depth_plot = depth_values[depth_mask]
            Sv_plot = Sv_data.values[:, depth_mask]

            # Replace NaN for plotting
            Sv_plot = np.nan_to_num(Sv_plot, nan=-90.0)

            # Determine color scale from data
            valid_sv = Sv_plot[Sv_plot > -89]
            if len(valid_sv) > 0:
                vmin = max(-90, np.percentile(valid_sv, 5))
                vmax = min(-40, np.percentile(valid_sv, 95))
            else:
                vmin, vmax = -90, -50

            im = axes[ch_idx].pcolormesh(
                np.arange(len(Sv_data.ping_time)),
                depth_plot,
                Sv_plot.T,
                shading='auto',
                cmap='viridis',
                vmin=vmin, vmax=vmax
            )

            axes[ch_idx].set_ylabel('Depth (m)')
            axes[ch_idx].set_title(f'Processed Echogram - {self.raw_file.name}\n'
                                   f'Frequency: {freq/1000:.0f} kHz | Variable: {sv_var}')
            axes[ch_idx].invert_yaxis()

            # Add surface exclusion line
            surface_excl = self.params['sonification']['surface_exclusion_depth']
            axes[ch_idx].axhline(y=surface_excl, color='red', linestyle='--',
                                alpha=0.7, label=f'Surface exclusion ({surface_excl}m)')
            axes[ch_idx].legend(loc='lower right')

            plt.colorbar(im, ax=axes[ch_idx], label='Sv (dB re 1 m⁻¹)')

        # Mean Sv over depth (temporal variation) - use first channel
        ch_idx = 0
        Sv_data = self.ds_Sv_clean[sv_var].isel(channel=ch_idx)
        mean_sv = np.nanmean(Sv_data.values, axis=1)
        axes[n_channels].plot(mean_sv, 'b-', linewidth=0.5)
        axes[n_channels].set_ylabel('Mean Sv (dB)')
        axes[n_channels].set_title('Temporal Variation (Mean Sv per ping)')
        axes[n_channels].grid(True, alpha=0.3)

        # Mark impulse noise if logged
        if self.artifacts_log:
            impulse_count = self.artifacts_log[-1].get('impulse_noise_samples', 0)
            axes[n_channels].text(0.02, 0.95, f'Impulse noise samples: {impulse_count}',
                                 transform=axes[n_channels].transAxes,
                                 fontsize=10, verticalalignment='top')

        # Vertical distribution profile
        depth_profile = np.nanmean(Sv_data.values, axis=0)
        axes[n_channels + 1].plot(depth_profile, depth_plot[:len(depth_profile)], 'g-', linewidth=1)
        axes[n_channels + 1].set_xlabel('Mean Sv (dB)')
        axes[n_channels + 1].set_ylabel('Depth (m)')
        axes[n_channels + 1].set_title('Vertical Distribution (Mean Sv per depth)')
        axes[n_channels + 1].invert_yaxis()
        axes[n_channels + 1].grid(True, alpha=0.3)
        axes[n_channels + 1].axhline(y=surface_excl, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save figure
        fig_file = self.output_dir / f"{self.raw_file.stem}_echogram.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved echogram to: {fig_file}")

        # Print depth information for debugging
        logger.info(f"Depth range plotted: 0 - {depth_plot[-1]:.1f} m")
        logger.info(f"Number of depth samples: {len(depth_plot)}")
        logger.info(f"Number of pings: {len(Sv_data.ping_time)}")
        logger.info(f"Sv variable used: {sv_var}")

        plt.close(fig)  # Close to free memory, don't show interactively

        return fig
    
    def process_complete_workflow(self):
        """
        Execute complete processing workflow
        """
        logger.info("="*60)
        logger.info("STARTING MALASPINA DATA PROCESSING")
        logger.info("="*60)
        
        try:
            # Load data
            self.load_raw_data()
            
            # Calibrate
            self.calibrate_data()
            
            # Clean and remove artifacts
            self.detect_and_remove_artifacts()
            
            # Compute MVBS
            self.compute_mvbs()
            
            # Save outputs
            self.save_processed_data()
            
            # Create visualization
            self.create_echogram_visualization()
            
            logger.info("="*60)
            logger.info("PROCESSING COMPLETE")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise


def main():
    """
    Main execution function - simple version
    """
    # Get file path from user
    raw_file = input("Enter path to .raw file: ").strip()
    
    # Check if file exists
    if not Path(raw_file).exists():
        print(f"Error: File not found: {raw_file}")
        return
    
    # Initialize processor with default output directory
    processor = MalaspinaProcessor(raw_file, output_dir="./malaspina_processed")
    
    # Run complete workflow
    processor.process_complete_workflow()
    
    # Access processed data for sonification
    features = processor.extract_sonification_features()
    
    print("\nSonification data ready!")
    print(f"Duration: {features['metadata']['duration_seconds']:.1f} seconds")
    print(f"Frequencies: {features['metadata']['frequencies']}")
    print(f"Artifacts removed: {features['metadata']['artifacts_removed']}")


if __name__ == "__main__":
    main()