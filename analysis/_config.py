"""Loads config/real_estate_demo.yml once and exposes it to every analytics module.

Every threshold, weight, and target that would realistically differ
from one real-estate client to the next lives in that YAML file, not
scattered across analysis/*.py as hardcoded constants -- see the
file's own header comment for why. This module's only job is reading
it exactly once (module-level cache) and handing back a plain dict, so
every analytics module does `from analysis._config import CONFIG`
instead of re-parsing YAML on every import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "real_estate_demo.yml"


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Real estate demo config not found at {path}. This platform requires "
            "config/real_estate_demo.yml for company name, currency, targets, health-score "
            "weights, and thresholds -- see that file's header comment."
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping; check its YAML syntax.")
    return data


CONFIG: dict[str, Any] = _load_config(CONFIG_PATH)
