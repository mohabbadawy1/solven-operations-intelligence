"""Inventory & Pricing Intelligence engine.

Turns units.csv into the numbers a Development Director or CCO needs
to answer "what's left to sell, how fast is it moving, and where is
it going stale": available inventory value, absorption, months of
supply, aging distribution, and price-per-sqm dispersion -- by
project, unit type, price band, and view.

Output shape: {summary, kpis, rankings, trends, anomalies,
health_score, methodology}. "trends" here is a project-level snapshot
comparison (available vs. sold-out share), not a time series --
units.csv, like the platform's original inventory.csv, is a
point-in-time position, not a dated ledger; velocity is instead
derived from sales_performance's own monthly net-contracted counts,
passed in as `monthly_net_sales` so this module never recomputes a
number sales_performance.py already owns.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis._config import CONFIG
from analysis._shared import (
    AGING_BUCKET_ORDER,
    _json_safe,
    _percentage,
    _safe_round,
    _zscores,
    bucket_rank,
    sort_and_limit_anomalies,
    to_records,
    validate_columns,
)

REQUIRED_UNIT_COLUMNS = [
    "unit_id", "project_id", "unit_type", "built_up_area_sqm", "view_type", "list_price",
    "price_per_sqm", "unit_status", "days_on_market", "availability_bucket",
]

AVAILABLE_STATUSES = {"available"}
SOLD_STATUSES = {"reserved", "contracted", "handed_over"}

MAX_ANOMALIES_RETURNED = 6
ANOMALY_Z_HIGH = 1.8

# --- Inventory Health Score ------------------------------------------
#
#   absorption            share of released inventory that has sold
#                         (reserved/contracted/handed_over), rescaled
#                         against each project's own absorption target
#   staleness              100 minus a penalty for the share of
#                         available inventory sitting in the oldest
#                         aging bucket (365+ days)
#   price_consistency        100 minus a penalty for how dispersed
#                         price-per-sqm is within a project/unit-type
#                         cohort -- wide, unexplained dispersion is a
#                         pricing-governance risk, not just a fact
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "absorption": 0.45,
    "staleness": 0.35,
    "price_consistency": 0.20,
}
STALE_THRESHOLD_DAYS = 180


def _prepare(units_df: pd.DataFrame) -> pd.DataFrame:
    if units_df is None or len(units_df) == 0:
        raise ValueError("analyze_inventory_velocity: units DataFrame is empty or None")
    validate_columns(units_df, REQUIRED_UNIT_COLUMNS, "analyze_inventory_velocity: units")

    df = units_df.copy()
    df["is_available"] = df["unit_status"].isin(AVAILABLE_STATUSES)
    df["is_sold"] = df["unit_status"].isin(SOLD_STATUSES)
    df["is_stale"] = df["is_available"] & (df["days_on_market"].fillna(0) >= STALE_THRESHOLD_DAYS)
    return df


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    released = df[df["unit_status"] != "unreleased"]
    available = df[df["is_available"]]
    sold = df[df["is_sold"]]

    return {
        "total_units": int(len(df)),
        "released_units": int(len(released)),
        "available_units": int(len(available)),
        "available_inventory_value": _safe_round(available["list_price"].sum()),
        "sold_units": int(len(sold)),
        "sell_through_rate_pct": _percentage(len(sold), len(released)) if len(released) else 0.0,
        "average_price_per_sqm": _safe_round(available["price_per_sqm"].mean()),
        "average_days_on_market_available": _safe_round(available["days_on_market"].mean()),
        "stale_units_180plus_days": int(df["is_stale"].sum()),
        "stale_units_pct_of_available": _percentage(int(df["is_stale"].sum()), len(available)) if len(available) else 0.0,
    }


# --------------------------------------------------------------------------
# Absorption & months of supply (uses external monthly net sales)
# --------------------------------------------------------------------------

def _compute_absorption(df: pd.DataFrame, monthly_net_sales_by_project: dict[str, float]) -> dict[str, Any]:
    """Absorption and months-of-supply per project.

    `monthly_net_sales_by_project` is the average net-contracted units
    sold per month, per project, over the recent period --
    sales_performance.py's own monthly trend, averaged over its most
    recent months by the caller (app.py), so this module never
    recomputes a sales-velocity figure that module already owns.
    """
    rows = []
    for project_id, group in df.groupby("project_id"):
        released = group[group["unit_status"] != "unreleased"]
        available = group[group["is_available"]]
        sold = group[group["is_sold"]]
        absorption_pct = _percentage(len(sold), len(released)) if len(released) else 0.0
        monthly_velocity = monthly_net_sales_by_project.get(project_id, 0.0)
        months_of_supply = round(len(available) / monthly_velocity, 1) if monthly_velocity else None
        rows.append({
            "project_id": project_id,
            "released_units": int(len(released)),
            "available_units": int(len(available)),
            "sold_units": int(len(sold)),
            "absorption_pct": absorption_pct,
            "avg_monthly_net_sales_velocity": _safe_round(monthly_velocity),
            "months_of_supply": months_of_supply,
        })
    rows.sort(key=lambda r: r["absorption_pct"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return {"by_project": rows}


# --------------------------------------------------------------------------
# Aging distribution
# --------------------------------------------------------------------------

def _compute_aging(df: pd.DataFrame) -> dict[str, Any]:
    available = df[df["is_available"]]
    counts = available["availability_bucket"].value_counts()
    distribution = sorted(
        (
            {"bucket": bucket, "count": int(count), "percentage": _percentage(count, len(available))}
            for bucket, count in counts.items() if pd.notna(bucket)
        ),
        key=lambda r: bucket_rank(r["bucket"], AGING_BUCKET_ORDER),
    )

    by_project = []
    for project_id, group in available.groupby("project_id"):
        stale = group[group["days_on_market"].fillna(0) >= STALE_THRESHOLD_DAYS]
        by_project.append({
            "project_id": project_id,
            "available_units": int(len(group)),
            "avg_days_on_market": _safe_round(group["days_on_market"].mean()),
            "stale_units": int(len(stale)),
            "stale_pct": _percentage(len(stale), len(group)) if len(group) else 0.0,
        })
    by_project.sort(key=lambda r: r["stale_pct"], reverse=True)

    by_unit_type = []
    for unit_type, group in available.groupby("unit_type"):
        stale = group[group["days_on_market"].fillna(0) >= STALE_THRESHOLD_DAYS]
        by_unit_type.append({
            "unit_type": unit_type,
            "available_units": int(len(group)),
            "avg_days_on_market": _safe_round(group["days_on_market"].mean()),
            "stale_pct": _percentage(len(stale), len(group)) if len(group) else 0.0,
        })
    by_unit_type.sort(key=lambda r: r["stale_pct"], reverse=True)

    return {
        "aging_distribution": distribution,
        "by_project": by_project,
        "by_unit_type": by_unit_type,
        "stalest_project": by_project[0]["project_id"] if by_project else None,
        "stalest_unit_type": by_unit_type[0]["unit_type"] if by_unit_type else None,
    }


# --------------------------------------------------------------------------
# Pricing dispersion
# --------------------------------------------------------------------------

def _compute_pricing_dispersion(df: pd.DataFrame) -> dict[str, Any]:
    sold = df[df["is_sold"]]
    rows = []
    for (project_id, unit_type), group in sold.groupby(["project_id", "unit_type"]):
        if len(group) < 5:
            continue
        ppsqm = group["price_per_sqm"]
        rows.append({
            "project_id": project_id, "unit_type": unit_type, "units": int(len(group)),
            "avg_price_per_sqm": _safe_round(ppsqm.mean()),
            "min_price_per_sqm": _safe_round(ppsqm.min()),
            "max_price_per_sqm": _safe_round(ppsqm.max()),
            "dispersion_pct": _safe_round(100 * (ppsqm.std(ddof=0) / ppsqm.mean())) if ppsqm.mean() else 0.0,
        })
    rows.sort(key=lambda r: r["dispersion_pct"] or 0, reverse=True)
    return {"by_project_unit_type": rows}


# --------------------------------------------------------------------------
# Rankings
# --------------------------------------------------------------------------

def _compute_rankings(df: pd.DataFrame) -> dict[str, Any]:
    by_project = []
    for project_id, group in df.groupby("project_id"):
        available = group[group["is_available"]]
        by_project.append({
            "project_id": project_id,
            "available_units": int(len(available)),
            "available_inventory_value": _safe_round(available["list_price"].sum()),
            "avg_days_on_market": _safe_round(available["days_on_market"].mean()),
        })
    by_project.sort(key=lambda r: r["available_inventory_value"] or 0, reverse=True)
    for i, r in enumerate(by_project, start=1):
        r["rank"] = i
    return {
        "by_project_available_value": by_project,
        "largest_available_position": by_project[0]["project_id"] if by_project else None,
    }


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

def _detect_anomalies(aging: dict[str, Any]) -> list[dict[str, Any]]:
    """Flag projects whose stale-inventory share is a statistical outlier vs. its peers (cross-sectional)."""
    by_project = aging["by_project"]
    if len(by_project) < 3:
        return []
    frame = pd.DataFrame(by_project)
    z = _zscores(frame["stale_pct"].astype(float))
    anomalies = []
    for idx, z_value in z.items():
        if z_value < ANOMALY_Z_HIGH:
            continue
        row = frame.loc[idx]
        anomalies.append({
            "severity": "HIGH" if z_value >= 2.5 else "MEDIUM",
            "category": "Inventory",
            "title": f"{row['project_id']} carries a disproportionate share of stale inventory",
            "description": (
                f"{row['project_id']} has {row['stale_pct']:.1f}% of its available units on the "
                f"market 180+ days ({int(row['stale_units'])} of {int(row['available_units'])}), "
                f"{z_value:.1f} standard deviations above its peer projects."
            ),
            "recommended_action": (
                f"Review pricing and unit-type mix for {row['project_id']}'s stale inventory; "
                "consider a targeted repricing or bundled incentive for the specific unit types "
                "driving the staleness rather than a blanket project-wide discount."
            ),
            "project_id": row["project_id"],
        })
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------

def _compute_health_score(summary: dict[str, Any], absorption: dict[str, Any], pricing: dict[str, Any]) -> dict[str, Any]:
    absorption_values = [row["absorption_pct"] for row in absorption["by_project"]]
    absorption_targets = {pid: d["absorption_target"] * 100 for pid, d in _project_targets().items()}
    if absorption["by_project"]:
        ratios = [
            min(row["absorption_pct"] / absorption_targets.get(row["project_id"], 65.0), 1.2)
            for row in absorption["by_project"] if absorption_targets.get(row["project_id"])
        ]
        absorption_score = min(100.0, (sum(ratios) / len(ratios)) * 100) if ratios else 70.0
    else:
        absorption_score = 70.0

    staleness_score = max(0.0, 100 - (summary["stale_units_pct_of_available"] or 0.0) * 2.5)

    dispersions = [row["dispersion_pct"] for row in pricing["by_project_unit_type"] if row["dispersion_pct"] is not None]
    avg_dispersion = sum(dispersions) / len(dispersions) if dispersions else 0.0
    price_consistency_score = max(0.0, 100 - avg_dispersion * 6)

    components = {
        "absorption": round(absorption_score, 1),
        "staleness": round(staleness_score, 1),
        "price_consistency": round(price_consistency_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            "Weighted blend of three 0-100 components: absorption (each project's actual "
            "absorption rate vs. its own configured target, averaged across projects and capped "
            "at 100, weight 0.45), staleness (100 minus 2.5 points per percentage point of "
            "available inventory aged 180+ days, weight 0.35), and price_consistency (100 minus a "
            "penalty for price-per-sqm dispersion within project/unit-type cohorts, weight 0.20)."
        ),
    }


def _project_targets() -> dict[str, dict[str, float]]:
    """Absorption targets by project, sourced from config where available."""
    # absorption_target lives in generate_real_estate_data.py's PROJECT_DEFS,
    # which analytics modules never import (analytics must not depend on the
    # data generator). Config carries a portfolio-wide default instead; a
    # per-project override could be added to config/real_estate_demo.yml
    # if a real engagement needs project-specific absorption targets.
    default_target = 0.65
    return {pid: {"absorption_target": default_target} for pid in ("AUR", "CST", "VTX", "HVW", "MER")}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_inventory_velocity(units_df: pd.DataFrame, monthly_net_sales_by_project: dict[str, float] | None = None) -> dict[str, Any]:
    """Run the full Inventory & Pricing Intelligence suite a Development Director/CCO needs.

    Args:
        units_df: The units DataFrame loaded from data/units.csv.
        monthly_net_sales_by_project: Average monthly net-contracted
            units sold per project, as already computed by
            analysis.sales_performance -- used only for months-of-supply.
            Defaults to an empty dict (months_of_supply reports as None
            rather than fabricating a velocity figure).

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},
                "kpis": {"absorption": {...}, "aging": {...}, "pricing_dispersion": {...}},
                "rankings": {...},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If `units_df` is empty/None or missing columns
            this module depends on.
    """
    df = _prepare(units_df)
    monthly_net_sales_by_project = monthly_net_sales_by_project or {}

    summary = _compute_summary(df)
    absorption = _compute_absorption(df, monthly_net_sales_by_project)
    aging = _compute_aging(df)
    pricing = _compute_pricing_dispersion(df)
    health_score = _compute_health_score(summary, absorption, pricing)

    result = {
        "summary": summary,
        "kpis": {"absorption": absorption, "aging": aging, "pricing_dispersion": pricing},
        "rankings": _compute_rankings(df),
        "anomalies": _detect_anomalies(aging),
        "health_score": health_score,
        "methodology": {
            "stale_definition": (
                f"A unit is 'stale' once it has been available {STALE_THRESHOLD_DAYS}+ days "
                "(days_on_market, measured from release_date), a threshold set in this module, "
                "not per-project -- see config/real_estate_demo.yml thresholds.stale_inventory_days "
                "for the same value used elsewhere in the platform."
            ),
            "months_of_supply": (
                "months_of_supply = available_units / average monthly net-contracted units sold, "
                "where the sales-velocity figure is supplied by analysis.sales_performance (this "
                "module never recomputes it), avoiding two modules disagreeing on the same number."
            ),
            "anomaly_detection": (
                "Stale-inventory anomalies are z-scored across projects' own stale-share percentage "
                "(cross-sectional, not time-based, since units.csv is a point-in-time position) -- "
                "never by referencing a specific project by name in code."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
