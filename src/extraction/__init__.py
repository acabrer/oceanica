# src/extraction/__init__.py
"""
Sonification feature extraction package.

Modules:
  sonification_extractor  — CLI orchestrator (run as script or import load_and_extract)
  features/               — Modular feature extraction (config, ping, derived, histogram, dvm, events)

The original monolith is preserved as sonification_extractor_monolith.py
and can still generate v5/v6 JSON if needed.
"""
