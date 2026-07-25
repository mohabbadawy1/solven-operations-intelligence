"""Generic analytics primitives shared by every analysis engine.

Everything here is a pure, domain-agnostic utility -- percentage/
rounding math, z-scoring, JSON sanitization, and small structural
patterns (building rate columns, converting a ranked DataFrame to
records, sorting anomalies by severity, formatting currency/months)
that show up identically across every analysis/*.py engine. Domain
logic (what a "project" or a "broker" is) does not belong in this
module -- only mechanics reused across engines belong here.
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


def sort_and_limit_anomalies(
    anomalies: list[dict[str, Any]], max_count: int
) -> list[dict[str, Any]]:
    """Sort anomalies most-severe-first and cap the list length."""
    return sorted(anomalies, key=lambda a: SEVERITY_RANK.get(a["severity"], len(SEVERITY_RANK)))[:max_count]


def period_over_period(
    current: float | None, prior: float | None, higher_is_better: bool = True
) -> dict[str, Any]:
    """Compare a current-period value to a prior-period value.

    Every real-estate KPI module uses this one function to build its
    period comparisons, so "favorable"/"unfavorable" is always decided
    by an explicit `higher_is_better` flag the caller states (higher
    sales is favorable; higher cancellations is not) rather than a
    naive "up is good" assumption baked into formatting code.

    Returns:
        {"current", "prior", "absolute_change", "percentage_change",
         "direction" ("up"/"down"/"flat"), "favorable" (bool | None)}.
        `favorable` is None when either value is missing, since no
        judgement can be made without both numbers.
    """
    if current is None or prior is None:
        return {
            "current": current, "prior": prior, "absolute_change": None,
            "percentage_change": None, "direction": None, "favorable": None,
        }
    absolute_change = round(current - prior, 2)
    percentage_change = round((absolute_change / prior) * 100, 1) if prior else None
    if absolute_change > 0:
        direction = "up"
    elif absolute_change < 0:
        direction = "down"
    else:
        direction = "flat"
    favorable = None if direction == "flat" else (
        (direction == "up") == higher_is_better
    )
    return {
        "current": current, "prior": prior, "absolute_change": absolute_change,
        "percentage_change": percentage_change, "direction": direction, "favorable": favorable,
    }


# Canonical aging-bucket order shared by every module that ages a
# receivable, an inventory unit, or any other dated exposure -- so a
# ranking or chart never accidentally sorts buckets alphabetically.
AGING_BUCKET_ORDER = ["0-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"]
RECEIVABLES_AGING_ORDER = ["Current", "1-30 days", "31-60 days", "61-90 days", "91-180 days", "180+ days"]


def bucket_rank(bucket: str | None, order: list[str]) -> int:
    """Sort key placing a named bucket in its canonical order; unknowns sort last."""
    try:
        return order.index(bucket)
    except ValueError:
        return len(order)


MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def fmt_currency_compact(value: float | None, currency: str = "EGP") -> str:
    """8_400_000 -> "EGP 8.4M"; 1_260_000_000 -> "EGP 1.26B".

    The one canonical compact-currency formatter for the platform --
    used both by analysis/correlations.py (so evidence sentences handed
    to the AI narrative layer never contain a raw, un-formatted float
    for it to copy verbatim) and by ai/html_report_renderer.py (so a
    structured currency value and a currency value quoted inside a
    sentence always render identically). Never exposes more than two
    decimal places or an un-abbreviated 9/10-digit number.
    """
    if value is None:
        return "N/A"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}{currency} {value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{sign}{currency} {value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}{currency} {value / 1_000:.0f}K"
    return f"{sign}{currency} {value:,.0f}"


def fmt_month_names(month_numbers: list[int]) -> str:
    """[5, 6, 11] -> "May, June, November". Never exposes a raw list literal in report prose."""
    names = [MONTH_NAMES[m - 1] for m in month_numbers if 1 <= m <= 12]
    return ", ".join(names)
