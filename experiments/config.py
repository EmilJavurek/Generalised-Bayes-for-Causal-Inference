"""Load the JSON configuration selected by an experiment's command line."""

import argparse
import json
from pathlib import Path

from gbci.paths import PROJECT_ROOT


def load_config(experiment: str) -> dict:
    default = PROJECT_ROOT / "experiments" / "configs" / f"{experiment}.json"
    parser = argparse.ArgumentParser(description=f"Run the {experiment} experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default,
        help=f"JSON configuration file (default: experiments/configs/{experiment}.json)",
    )
    path = parser.parse_args().config.expanduser()
    with path.open(encoding="utf-8") as source:
        settings = json.load(source)
    if not isinstance(settings, dict):
        parser.error(f"Configuration must be a JSON object: {path}")
    return settings
