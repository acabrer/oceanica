# src/extraction/features/__init__.py
"""
Feature extraction modules for sonification.

Each module handles an independent feature domain:
  config.py           — SonificationConfigV3 dataclass
  normalization.py    — Shared normalization utilities
  ping_features.py    — Per-ping extraction (CoM, entropy, peaks, histogram)
  derived_features.py — Time-series derived features (velocity, anomaly, onset)
  histogram.py        — 8-band oceanographic zone energy + differentials
  dvm.py              — DVM depth tracking (corridor center-of-mass)
  events.py           — Layer events, regime changepoints, layer tracking
  formatter_v8.py     — v8 JSON builder for SuperCollider
"""

from .config import SonificationConfigV3
