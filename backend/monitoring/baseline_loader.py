"""
Baseline Loader (Drift Monitoring)
==================================

This module loads "baseline feature distributions" used by drift detection.

Why baselines exist:
--------------------
Drift tests like KS and PSI compare a *current batch* of feature values
against a *baseline distribution* saved at training time.

Where baselines live:
---------------------
Preferred (current convention):
  artifacts/baseline/<feature_name>.npy

Backwards-compatible fallback (older convention):
  models/saved_models/baseline_dist_<feature_name>.npy

If baselines are missing:
-------------------------
We raise a clear error telling you to run:
  python models/train_model.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _repo_root() -> Path:
    """
    Get repo root reliably regardless of where uvicorn is launched.
    baseline_loader.py lives at:
      backend/monitoring/baseline_loader.py
    so repo root is 3 parents up from this file.
    """
    return Path(__file__).resolve().parents[2]


def load_baselines() -> Dict[str, np.ndarray]:
    """
    Loads baseline distributions for every feature.

    Returns:
        Dict[str, np.ndarray]: Mapping of feature_name -> 1D array of baseline values

    Raises:
        FileNotFoundError: If no baselines are found in either location.
    """
    root = _repo_root()

    preferred_dir = root / "artifacts" / "baseline"
    legacy_dir = root / "models" / "saved_models"

    baselines: Dict[str, np.ndarray] = {}

    # -----------------------------
    # 1) Preferred: artifacts/baseline/*.npy
    # -----------------------------
    if preferred_dir.exists():
        for fp in preferred_dir.glob("*.npy"):
            feature_name = fp.stem
            baselines[feature_name] = np.load(fp, allow_pickle=False)

    # -----------------------------
    # 2) Fallback: models/saved_models/baseline_dist_*.npy
    # -----------------------------
    if not baselines and legacy_dir.exists():
        for fp in legacy_dir.glob("baseline_dist_*.npy"):
            # baseline_dist_V1.npy -> V1
            feature_name = fp.stem.replace("baseline_dist_", "")
            baselines[feature_name] = np.load(fp, allow_pickle=False)

    if not baselines:
        raise FileNotFoundError(
            "No baseline distributions found.\n"
            "Expected either:\n"
            "  artifacts/baseline/*.npy\n"
            "or (legacy):\n"
            "  models/saved_models/baseline_dist_*.npy\n\n"
            "Fix: run `python models/train_model.py` to generate baseline files."
        )

    return baselines
