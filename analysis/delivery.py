"""Enterprise delivery analytics engine.

Turns a raw shipments DataFrame (as exported from the TMS) into the
structured findings an Operations Director actually asks for: fleet
KPIs, warehouse/driver/regional/customer/priority scorecards, monthly
trends, data-driven anomaly detection, and a single Operations Health
Score. The output is a plain, JSON-serializable dictionary intended
to be handed directly to the AI reporting layer (ai/report_generator.py).

Design notes
------------
- "On-time" is defined structurally: a shipment is on-time when it
  was actually delivered (status == "Delivered") rather than merely
  "not late", because the data generator only assigns that status
  when the shipment cleared its own promised_delivery_date. Lost and
  Cancelled shipments never reached the customer, so they are
  excluded from transit-time averages (a shipment that was lost has
  no meaningful "delivery time") but are still counted in volume and
  rate denominators.
- Anomaly detection never references a specific warehouse, driver, or
  month by name. It works by comparing every warehouse's own monthly
  metrics against that warehouse's own 12-month baseline (a z-score
  against its own mean/std), and every driver's first-half-of-year
  average rating against their own second-half average. A real
  operational problem shows up as a statistical outlier regardless of
  where in the data it occurs.
- Operations Health Score methodology is documented next to
  HEALTH_SCORE_WEIGHTS below and echoed in the returned
  "methodology" field so the score is never a black box.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis._shared import (
    _json_safe,
    _percentage,
    _safe_round,
    _zscores,
    sort_and_limit_anomalies,
    to_records,
    validate_columns,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Columns this module relies on. Validated up front so a malformed or
# out-of-date export fails fast with a clear message instead of a
# cryptic KeyError halfway through analysis.
REQUIRED_COLUMNS = [
    "shipment_id", "order_date", "region", "warehouse", "status",
    "delivery_days", "delivery_cost", "fuel_cost", "driver_rating",
    "driver_id", "driver_name", "loading_time_minutes",
    "unloading_time_minutes", "promised_delivery_date",
    "actual_delivery_date", "customer_type", "priority",
]

# A shipment counts toward transit-time metrics only if it actually
# reached the customer. Lost/Cancelled shipments are excluded from
# "average delivery days" but still counted in volume and delay rate.
COMPLETED_STATUSES = {"Delivered", "Delayed"}

# --- Driver Intelligence ---------------------------------------------------

# Drivers with fewer deliveries than this are excluded from ranking,
# top-performer, and coaching-candidate lists so that a driver with
# (say) two deliveries and one bad day doesn't get flagged for
# coaching alongside drivers with a real track record.
MIN_SHIPMENTS_FOR_DRIVER_RANKING = 15
TOP_PERFORMER_COUNT = 5
COACHING_CANDIDATE_COUNT = 5

# --- Anomaly detection -------------------------------------------------

# A warehouse-month bucket needs at least this many shipments before
# its metrics are trusted enough to be compared against baseline.
ANOMALY_MIN_MONTHLY_SHIPMENTS = 15
# A warehouse needs at least this many qualifying months before a
# same-warehouse baseline mean/std is considered meaningful.
MIN_MONTHS_FOR_BASELINE = 3
ANOMALY_Z_SCORE_MEDIUM = 2.0
ANOMALY_Z_SCORE_HIGH = 3.0

DRIVER_DEGRADATION_MIN_SHIPMENTS_PER_HALF = 5
DRIVER_DEGRADATION_RATING_DROP_THRESHOLD = 0.3
DRIVER_DEGRADATION_HIGH_THRESHOLD = 0.6

MAX_ANOMALIES_RETURNED = 10

# Metrics scanned for warehouse-month anomalies: whether an increase
# ("high") or a decrease ("low") in that metric is the bad direction,
# and the human-readable label used in generated anomaly text.
WAREHOUSE_ANOMALY_METRICS: dict[str, tuple[str, str]] = {
    "delay_rate": ("high", "delay rate"),
    "avg_delivery_days": ("high", "average delivery time"),
    "avg_loading_time": ("high", "average loading time"),
    "avg_driver_rating": ("low", "average driver rating"),
}

# --- Operations Health Score ------------------------------------------

# The health score is a weighted blend of four 0-100 sub-scores:
#
#   on_time_rate           the network's overall on-time delivery %,
#                           taken directly as its score (already 0-100)
#   delay_severity          100 minus a penalty for how many days late
#                           shipments run on average across the fleet
#   driver_rating            average driver rating rescaled from a
#                           1-5 star scale to 0-100
#   warehouse_consistency    100 minus a penalty for the spread
#                           between the best and worst warehouse's
#                           annual delay rate -- a network with one
#                           badly-run warehouse scores worse here
#                           even if the fleet-wide average looks fine
#
# Weights (sum to 1.0) reflect that whether the customer got their
# order on time (on_time_rate) matters most, service quality
# (driver_rating) matters nearly as much, and the other two act as
# secondary quality/equity checks.
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "on_time_rate": 0.35,
    "delay_severity": 0.20,
    "driver_rating": 0.25,
    "warehouse_consistency": 0.20,
}
SEVERITY_POINTS_PER_DELAY_DAY = 8.0
CONSISTENCY_PENALTY_PER_POINT = 1.2
MAX_DRIVER_RATING = 5.0


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
#
# Domain-agnostic primitives (_percentage, _safe_round, _zscores,
# _json_safe, to_records, validate_columns, sort_and_limit_anomalies)
# live in analysis._shared and are imported above, so this module and
# analysis/complaints.py share one implementation of each.

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and enrich a raw shipments DataFrame for analysis.

    Coerces date columns (defensively, in case the caller loaded the
    CSV without parse_dates), and derives two working columns reused
    across every section below: `month` (a "YYYY-MM" label) and
    `completed_delivery_days` (delivery_days for shipments that
    actually reached the customer, NaN otherwise, so group means
    automatically exclude Lost/Cancelled shipments).
    """
    if df is None or len(df) == 0:
        raise ValueError("analyze_deliveries: shipments DataFrame is empty or None")
    validate_columns(df, REQUIRED_COLUMNS, "analyze_deliveries: shipments")

    df = df.copy()
    for col in ("order_date", "promised_delivery_date", "actual_delivery_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["month"] = df["order_date"].dt.strftime("%Y-%m")
    is_completed = df["status"].isin(COMPLETED_STATUSES)
    df["completed_delivery_days"] = df["delivery_days"].where(is_completed)

    delay_days = (df["actual_delivery_date"] - df["promised_delivery_date"]).dt.days
    df["delay_days"] = delay_days.clip(lower=0).fillna(0)
    return df


# --------------------------------------------------------------------------
# Executive KPIs
# --------------------------------------------------------------------------

def _compute_executive_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Fleet-wide headline numbers for an executive summary."""
    total = len(df)
    status_counts = df["status"].value_counts()
    delivered = int(status_counts.get("Delivered", 0))
    delayed = int(status_counts.get("Delayed", 0))
    cancelled = int(status_counts.get("Cancelled", 0))
    lost = int(status_counts.get("Lost", 0))

    return {
        "total_shipments": total,
        "delivered_shipments": delivered,
        "delayed_shipments": delayed,
        "cancelled_shipments": cancelled,
        "lost_shipments": lost,
        "on_time_delivery_percentage": _percentage(delivered, total),
        "delay_rate_percentage": _percentage(delayed, total),
        "average_delivery_days": _safe_round(df["completed_delivery_days"].mean()),
        "average_delivery_cost": _safe_round(df["delivery_cost"].mean()),
        "average_fuel_cost": _safe_round(df["fuel_cost"].mean()),
        "average_driver_rating": _safe_round(df["driver_rating"].mean()),
    }


# --------------------------------------------------------------------------
# Warehouse Intelligence
# --------------------------------------------------------------------------

def _compute_warehouse_intelligence(df: pd.DataFrame) -> dict[str, Any]:
    """Per-warehouse scorecard, ranked best to worst."""
    grouped = df.groupby("warehouse").agg(
        shipment_volume=("shipment_id", "count"),
        delayed=("status", lambda s: (s == "Delayed").sum()),
        delivered=("status", lambda s: (s == "Delivered").sum()),
        avg_delivery_days=("completed_delivery_days", "mean"),
        avg_driver_rating=("driver_rating", "mean"),
        avg_loading_time=("loading_time_minutes", "mean"),
        avg_unloading_time=("unloading_time_minutes", "mean"),
    ).reset_index()

    grouped["delay_rate_percentage"] = grouped.apply(
        lambda r: _percentage(r["delayed"], r["shipment_volume"]), axis=1
    )
    grouped["on_time_rate_percentage"] = grouped.apply(
        lambda r: _percentage(r["delivered"], r["shipment_volume"]), axis=1
    )

    ranked = grouped.sort_values(
        ["on_time_rate_percentage", "avg_driver_rating"], ascending=[False, False]
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)

    columns = [
        "rank", "warehouse", "shipment_volume", "on_time_rate_percentage",
        "delay_rate_percentage", "avg_delivery_days", "avg_driver_rating",
        "avg_loading_time", "avg_unloading_time",
    ]
    records = to_records(ranked, columns)

    return {
        "ranked_warehouses": records,
        "best_warehouse": records[0]["warehouse"] if records else None,
        "worst_warehouse": records[-1]["warehouse"] if records else None,
    }


# --------------------------------------------------------------------------
# Driver Intelligence
# --------------------------------------------------------------------------

def _compute_driver_intelligence(df: pd.DataFrame) -> dict[str, Any]:
    """Per-driver scorecard, ranked best to worst among drivers with enough volume."""
    grouped = df.groupby(["driver_id", "driver_name"]).agg(
        deliveries_completed=("shipment_id", "count"),
        delayed=("status", lambda s: (s == "Delayed").sum()),
        avg_delivery_days=("completed_delivery_days", "mean"),
        avg_customer_rating=("driver_rating", "mean"),
    ).reset_index()
    grouped["delay_rate_percentage"] = grouped.apply(
        lambda r: _percentage(r["delayed"], r["deliveries_completed"]), axis=1
    )

    eligible_mask = grouped["deliveries_completed"] >= MIN_SHIPMENTS_FOR_DRIVER_RANKING
    eligible = grouped[eligible_mask].sort_values(
        ["avg_customer_rating", "delay_rate_percentage"], ascending=[False, True]
    ).reset_index(drop=True)
    eligible.insert(0, "rank", eligible.index + 1)

    columns = [
        "rank", "driver_id", "driver_name", "deliveries_completed",
        "avg_customer_rating", "delay_rate_percentage", "avg_delivery_days",
    ]
    records = to_records(eligible, columns)

    return {
        "ranked_drivers": records,
        "top_performers": records[:TOP_PERFORMER_COUNT],
        "drivers_needing_coaching": list(reversed(records[-COACHING_CANDIDATE_COUNT:])),
        "minimum_shipments_threshold": MIN_SHIPMENTS_FOR_DRIVER_RANKING,
        "excluded_low_volume_driver_count": int((~eligible_mask).sum()),
    }


# --------------------------------------------------------------------------
# Regional Intelligence
# --------------------------------------------------------------------------

def _compute_regional_intelligence(df: pd.DataFrame) -> dict[str, Any]:
    """Per-region scorecard, ranked best to worst, with the highest-risk region flagged."""
    grouped = df.groupby("region").agg(
        shipment_count=("shipment_id", "count"),
        delayed=("status", lambda s: (s == "Delayed").sum()),
        avg_delivery_days=("completed_delivery_days", "mean"),
    ).reset_index()
    grouped["delay_rate_percentage"] = grouped.apply(
        lambda r: _percentage(r["delayed"], r["shipment_count"]), axis=1
    )

    ranked = grouped.sort_values("delay_rate_percentage", ascending=True).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)

    columns = ["rank", "region", "shipment_count", "delay_rate_percentage", "avg_delivery_days"]
    records = to_records(ranked, columns)

    return {
        "ranked_regions": records,
        "highest_risk_region": records[-1]["region"] if records else None,
    }


# --------------------------------------------------------------------------
# Customer Intelligence
# --------------------------------------------------------------------------

def _compute_customer_intelligence(df: pd.DataFrame) -> dict[str, Any]:
    """Delay rate and delivery speed compared across customer tiers."""
    grouped = df.groupby("customer_type").agg(
        shipment_count=("shipment_id", "count"),
        delayed=("status", lambda s: (s == "Delayed").sum()),
        avg_delivery_days=("completed_delivery_days", "mean"),
    )

    result: dict[str, Any] = {}
    for customer_type, row in grouped.iterrows():
        result[customer_type] = {
            "shipment_count": int(row["shipment_count"]),
            "delay_rate_percentage": _percentage(row["delayed"], row["shipment_count"]),
            "average_delivery_days": _safe_round(row["avg_delivery_days"]),
        }
    return result


# --------------------------------------------------------------------------
# Priority Intelligence
# --------------------------------------------------------------------------

def _compute_priority_intelligence(df: pd.DataFrame) -> dict[str, Any]:
    """SLA performance compared across priority tiers.

    Since higher-priority shipments carry a tighter promised date by
    design, on-time percentage is the fair way to compare tiers (it
    already accounts for each shipment's own SLA rather than a single
    fixed cutoff).
    """
    grouped = df.groupby("priority").agg(
        shipment_count=("shipment_id", "count"),
        delivered=("status", lambda s: (s == "Delivered").sum()),
        avg_delivery_days=("completed_delivery_days", "mean"),
    )

    tiers: dict[str, Any] = {}
    for priority, row in grouped.iterrows():
        tiers[priority] = {
            "shipment_count": int(row["shipment_count"]),
            "on_time_rate_percentage": _percentage(row["delivered"], row["shipment_count"]),
            "average_delivery_days": _safe_round(row["avg_delivery_days"]),
        }

    standard_on_time = tiers.get("Standard", {}).get("on_time_rate_percentage", 0.0)
    premium_tiers = [t for t in ("Express", "Critical") if t in tiers]
    premium_meets_expectations = all(
        tiers[t]["on_time_rate_percentage"] >= standard_on_time for t in premium_tiers
    )

    return {
        "tiers": tiers,
        "premium_tiers_meeting_expectations": premium_meets_expectations,
        "notes": (
            "Express/Critical shipments are held to the same on-time bar as Standard "
            "shipments' own (looser) promised date. 'meeting expectations' means paying "
            "for priority handling is not, on average, delivering worse reliability."
        ),
    }


# --------------------------------------------------------------------------
# Monthly Trends
# --------------------------------------------------------------------------

def _compute_monthly_trends(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Month-over-month trend series suitable for a dashboard line chart."""
    grouped = df.groupby("month").agg(
        shipment_count=("shipment_id", "count"),
        delayed=("status", lambda s: (s == "Delayed").sum()),
        avg_delivery_days=("completed_delivery_days", "mean"),
        avg_driver_rating=("driver_rating", "mean"),
    ).reset_index().sort_values("month")

    return [
        {
            "month": row["month"],
            "shipment_count": int(row["shipment_count"]),
            "delay_rate_percentage": _percentage(row["delayed"], row["shipment_count"]),
            "average_delivery_days": _safe_round(row["avg_delivery_days"]),
            "average_driver_rating": _safe_round(row["avg_driver_rating"]),
        }
        for _, row in grouped.iterrows()
    ]


# --------------------------------------------------------------------------
# Operational Anomaly Detection
# --------------------------------------------------------------------------

def _detect_warehouse_monthly_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag warehouse-months that are statistical outliers vs. that warehouse's own year.

    For each warehouse, computes delay rate, average delivery time,
    average loading time, and average driver rating per month, then
    z-scores each metric against that warehouse's own 12-month
    distribution. A month where a metric moves several standard
    deviations in the "bad" direction is reported as an anomaly. This
    is deliberately warehouse-relative (not compared across
    warehouses) so a warehouse with a naturally longer baseline
    transit time isn't penalized for being far away -- only for
    deviating from its own normal.
    """
    metrics = df.groupby(["warehouse", "month"]).agg(
        shipment_count=("shipment_id", "count"),
        delayed=("status", lambda s: (s == "Delayed").sum()),
        avg_delivery_days=("completed_delivery_days", "mean"),
        avg_driver_rating=("driver_rating", "mean"),
        avg_loading_time=("loading_time_minutes", "mean"),
    ).reset_index()
    metrics["delay_rate"] = metrics["delayed"] / metrics["shipment_count"]
    metrics = metrics[metrics["shipment_count"] >= ANOMALY_MIN_MONTHLY_SHIPMENTS]

    anomalies: list[dict[str, Any]] = []
    for warehouse, group in metrics.groupby("warehouse"):
        if len(group) < MIN_MONTHS_FOR_BASELINE:
            continue
        group = group.sort_values("month").reset_index(drop=True)

        fired: dict[int, list[tuple[str, float]]] = {}
        for metric, (direction, _label) in WAREHOUSE_ANOMALY_METRICS.items():
            z = _zscores(group[metric])
            for idx, z_value in z.items():
                triggered = (
                    (direction == "high" and z_value >= ANOMALY_Z_SCORE_MEDIUM)
                    or (direction == "low" and z_value <= -ANOMALY_Z_SCORE_MEDIUM)
                )
                if triggered:
                    fired.setdefault(idx, []).append((metric, z_value))

        for idx, hits in fired.items():
            row = group.loc[idx]
            worst_z = max(abs(z) for _metric, z in hits)
            severity = "HIGH" if worst_z >= ANOMALY_Z_SCORE_HIGH else "MEDIUM"
            metric_summary = "; ".join(
                f"{WAREHOUSE_ANOMALY_METRICS[metric][1]} at {row[metric]:.2f}" for metric, _z in hits
            )
            anomalies.append({
                "severity": severity,
                "category": "Warehouse",
                "title": f"{warehouse} operational degradation in {row['month']}",
                "description": (
                    f"{warehouse} deviates sharply from its own 12-month baseline in "
                    f"{row['month']}: {metric_summary} across {int(row['shipment_count'])} shipments."
                ),
                "recommended_action": (
                    f"Investigate {warehouse}'s order volume, staffing, and capacity for "
                    f"{row['month']} to confirm an overload event, and consider temporarily "
                    "rebalancing routes to neighboring warehouses."
                ),
                "warehouse": warehouse,
                "month": row["month"],
            })

    return anomalies


def _detect_driver_degradation(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag drivers whose average rating dropped significantly from H1 to H2.

    Compares each driver's own first-half-of-year average rating
    against their own second-half average, requiring a minimum
    delivery count in both halves so the comparison isn't driven by a
    handful of data points.
    """
    working = df.copy()
    working["half"] = np.where(working["order_date"].dt.month <= 6, "H1", "H2")

    grouped = working.groupby(["driver_id", "driver_name", "half"]).agg(
        shipments=("shipment_id", "count"),
        avg_rating=("driver_rating", "mean"),
    ).reset_index()

    pivot = grouped.pivot(index=["driver_id", "driver_name"], columns="half", values=["shipments", "avg_rating"])
    pivot.columns = [f"{value}_{half}" for value, half in pivot.columns]
    pivot = pivot.reset_index().dropna(subset=["avg_rating_H1", "avg_rating_H2"])

    eligible = pivot[
        (pivot["shipments_H1"] >= DRIVER_DEGRADATION_MIN_SHIPMENTS_PER_HALF)
        & (pivot["shipments_H2"] >= DRIVER_DEGRADATION_MIN_SHIPMENTS_PER_HALF)
    ].copy()
    eligible["rating_drop"] = eligible["avg_rating_H1"] - eligible["avg_rating_H2"]
    flagged = eligible[eligible["rating_drop"] >= DRIVER_DEGRADATION_RATING_DROP_THRESHOLD]

    anomalies: list[dict[str, Any]] = []
    for _, row in flagged.iterrows():
        severity = "HIGH" if row["rating_drop"] >= DRIVER_DEGRADATION_HIGH_THRESHOLD else "MEDIUM"
        anomalies.append({
            "severity": severity,
            "category": "Driver",
            "title": f"{row['driver_name']} ({row['driver_id']}) rating decline",
            "description": (
                f"{row['driver_name']}'s average customer rating fell from "
                f"{row['avg_rating_H1']:.2f} (H1) to {row['avg_rating_H2']:.2f} (H2) "
                f"across {int(row['shipments_H1'])} and {int(row['shipments_H2'])} deliveries respectively."
            ),
            "recommended_action": (
                f"Review {row['driver_name']}'s recent routes and workload; a rating drop of "
                "this size usually reflects route overload, vehicle issues, or an unaddressed "
                "customer complaint pattern."
            ),
            "driver_id": row["driver_id"],
        })
    return anomalies


def _detect_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run every anomaly detector and return the most severe findings first."""
    anomalies = _detect_warehouse_monthly_anomalies(df) + _detect_driver_degradation(df)
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Operations Health Score
# --------------------------------------------------------------------------

def _compute_health_score(
    df: pd.DataFrame, kpis: dict[str, Any], warehouse_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Blend fleet KPIs into a single 0-100 Operations Health Score.

    See HEALTH_SCORE_WEIGHTS above for the full methodology. Every
    component score and the weights used are returned alongside the
    final number so the score is always auditable, never a black box.
    """
    on_time_score = kpis["on_time_delivery_percentage"]

    avg_delay_days = float(df["delay_days"].mean())
    delay_severity_score = max(0.0, 100 - avg_delay_days * SEVERITY_POINTS_PER_DELAY_DAY)

    avg_rating = kpis["average_driver_rating"] or 0.0
    driver_rating_score = min(100.0, (avg_rating / MAX_DRIVER_RATING) * 100)

    delay_rates = [w["delay_rate_percentage"] for w in warehouse_records]
    spread = (max(delay_rates) - min(delay_rates)) if delay_rates else 0.0
    warehouse_consistency_score = max(0.0, 100 - spread * CONSISTENCY_PENALTY_PER_POINT)

    components = {
        "on_time_rate": round(on_time_score, 1),
        "delay_severity": round(delay_severity_score, 1),
        "driver_rating": round(driver_rating_score, 1),
        "warehouse_consistency": round(warehouse_consistency_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            "Weighted blend of four 0-100 components: on_time_rate (network on-time %, "
            "weight 0.35), delay_severity (100 minus 8 points per average day late across "
            "the fleet, weight 0.20), driver_rating (average driver rating rescaled from a "
            "5-star to a 100-point scale, weight 0.25), and warehouse_consistency (100 "
            "minus 1.2x the percentage-point gap between the best and worst warehouse's "
            "annual delay rate, weight 0.20). This is an annual figure: an acute single-month "
            "event (see 'anomalies') can be severe without moving this score much, by design "
            "-- it measures sustained network health, not a single incident."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_deliveries(df: pd.DataFrame) -> dict[str, Any]:
    """Run the full delivery analytics suite an Operations Director needs.

    Computes executive KPIs, warehouse/driver/regional/customer/
    priority scorecards, monthly trend series, statistically-detected
    operational anomalies, and an overall Operations Health Score.

    Args:
        df: The shipments DataFrame loaded from data/shipments.csv
            (parsed or unparsed dates are both accepted).

    Returns:
        A hierarchical, JSON-serializable dictionary suitable for
        rendering in a dashboard or passed directly to an LLM as
        report-generation context.

    Raises:
        ValueError: If `df` is empty/None or missing columns this
            module depends on.
    """
    prepared = _prepare(df)

    kpis = _compute_executive_kpis(prepared)
    warehouse_intelligence = _compute_warehouse_intelligence(prepared)

    result = {
        "executive_kpis": kpis,
        "warehouse_intelligence": warehouse_intelligence,
        "driver_intelligence": _compute_driver_intelligence(prepared),
        "regional_intelligence": _compute_regional_intelligence(prepared),
        "customer_intelligence": _compute_customer_intelligence(prepared),
        "priority_intelligence": _compute_priority_intelligence(prepared),
        "monthly_trends": _compute_monthly_trends(prepared),
        "anomalies": _detect_anomalies(prepared),
        "operations_health_score": _compute_health_score(
            prepared, kpis, warehouse_intelligence["ranked_warehouses"]
        ),
    }
    return _json_safe(result)
