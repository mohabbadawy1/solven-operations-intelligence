"""Generic analytics primitives shared by every analysis engine.

Everything here is a pure, domain-agnostic utility -- percentage/
rounding math, z-scoring, JSON sanitization, and small structural
patterns (building rate columns, converting a ranked DataFrame to
records, sorting anomalies by severity) that show up identically in
analysis/delivery.py and analysis/complaints.py. Domain logic (what a
"warehouse" or a "complaint" is) does not belong in this module --
only mechanics reused across engines belong here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SEVERITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _percentage(numerator: float, denominator: float, ndigits: int = 1) -> float:
    """Compute a percentage, returning 0.0 for an empty denominator."""
    if not denominator:
        return 0.0
    return round(100 * numerator / denominator, ndigits)


def _safe_round(value: Any, ndigits: int = 2) -> float | None:
    """Round a numeric value, returning None for missing/NaN input."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), ndigits)


def _zscores(series: pd.Series) -> pd.Series:
    """Population z-scores for a series; zeros out if there's no spread."""
    std = series.std(ddof=0)
    if not std or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars into plain JSON-safe Python types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        return None if np.isnan(as_float) else as_float
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def validate_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    """Raise a clear ValueError if `df` is missing any column in `required`.

    Args:
        df: The DataFrame to check.
        required: Column names the caller depends on.
        context: A short prefix identifying the caller and dataset,
            e.g. "analyze_complaints: complaints", used to make the
            error message self-explanatory without a traceback.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context} data is missing required columns: {missing}")


def to_records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Convert selected columns of a DataFrame to a list of dicts, rounding floats.

    Centralizes the "select columns, round every float column, emit
    plain dicts" pattern used by every ranked scorecard in the
    analytics engines.
    """
    return [
        {k: _safe_round(v) if isinstance(v, float) else v for k, v in row.items()}
        for row in df[columns].to_dict("records")
    ]


def rate_against_population(counts: pd.Series, population: pd.Series) -> pd.Series:
    """Compute `counts / population` as a percentage, aligned by index.

    Both series are grouped totals indexed by the same category (e.g.
    complaint counts and shipment counts, both indexed by warehouse).
    Categories present in only one series are treated as zero in the
    other, so a segment with events but no population (or vice versa)
    doesn't silently disappear from the result.
    """
    combined = pd.DataFrame({"count": counts, "population": population}).fillna(0)
    return combined.apply(lambda r: _percentage(r["count"], r["population"]), axis=1)


def sort_and_limit_anomalies(
    anomalies: list[dict[str, Any]], max_count: int
) -> list[dict[str, Any]]:
    """Sort anomalies most-severe-first and cap the list length."""
    return sorted(anomalies, key=lambda a: SEVERITY_RANK.get(a["severity"], len(SEVERITY_RANK)))[:max_count]
