"""Local benchmark inputs and generated results."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = Path(os.environ.get("GBCI_RESULTS_DIR", PROJECT_ROOT / "results")).expanduser().resolve()
