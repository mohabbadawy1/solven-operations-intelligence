"""Customer Experience Intelligence engine.

Turns raw complaint tickets (as exported from the CRM) into the
findings a Chief Customer Officer, COO, or CEO actually asks for:
where dissatisfaction concentrates, whether the support organization
is closing tickets fast enough to keep customers, and whether
operational problems documented in the TMS (delays, specific
warehouses, priority tiers) are the ones actually driving complaints.

complaints.csv does not carry warehouse, delivery status, priority,
customer tier, or vehicle type -- those live on the shipment. Every
cross-dataset section in this module joins complaints back to
shipments on `shipment_id` first, which is what makes those questions
answerable at all (see `_prepare`).

The output mirrors the structure of analysis/delivery.py's
`analyze_deliveries`: a hierarchical, JSON-serializable dictionary
meant to be handed directly to the AI reporting layer, plus a single
Customer Health Score. Its own top-level shape --
{summary, kpis, rankings, trends, anomalies, health_score,
methodology} -- groups the many requested intelligence sections under
those seven keys; see analyze_complaints()'s docstring for the full
map.

Design notes
------------
- "Complaint rate" is always complaints as a percentage of that
  segment's total *shipments* (from shipments.csv), never complaints
  as a percentage of other complaints. A warehouse that ships far
  more volume should not look artificially worse just for being
  bigger; rate, not raw count, is the fair unit of comparison. Where
  raw share of total complaints is actually the more useful question
  (e.g. "which warehouse contributes the most unhappy customers in
  absolute terms"), that is reported separately and labeled as a
  share, not a rate.
- Anomaly detection never references a specific warehouse or month by
  name. It z-scores each month's network-wide metrics against the
  network's own historical baseline, and each warehouse's monthly
  complaint rate against that same warehouse's own baseline -- the
  same statistical approach used in analysis/delivery.py. Note that
  `month` here is a complaint's own filing month, which can trail its
  shipment's order month by up to ~11 days (and occasionally spill
  into the following calendar year), so the baseline isn't always
  exactly 12 buckets.
- Customer Health Score methodology is documented next to
  HEALTH_SCORE_WEIGHTS below and echoed in the returned dictionary so
  the score is never a black box.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis._shared import (
    _json_safe,
    _percentage,
    _safe_round,
    _zscores,
    rate_against_population,
    sort_and_limit_anomalies,
    to_records,
    validate_columns,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

REQUIRED_COMPLAINT_COLUMNS = [
    "complaint_id", "shipment_id", "complaint_date", "category", "sentiment",
    "resolution_time_hours", "assigned_team", "escalation_level", "resolved",
    "customer_satisfaction_after_resolution",
]
# Shipment attributes complaints are joined against for cross-dataset
# intelligence; complaints.csv does not carry any of these natively.
REQUIRED_SHIPMENT_COLUMNS = [
    "shipment_id", "order_date", "warehouse", "status", "priority",
    "customer_type", "vehicle_type",
]
SHIPMENT_JOIN_COLUMNS = [
    "shipment_id", "warehouse", "status", "priority", "customer_type", "vehicle_type",
]

# A complaint is considered "negative" for sentiment-rate purposes
# when its sentiment label is exactly this.
NEGATIVE_SENTIMENT_LABEL = "Negative"
MAX_SATISFACTION_RATING = 5.0

# escalation_level starts at 1 (front-line handling) in the source
# data; a ticket is considered genuinely "escalated" once it moves
# past that baseline.
ESCALATED_LEVEL_THRESHOLD = 2

# --- Cross-dataset intelligence -----------------------------------------

CROSS_DATASET_DIMENSIONS = ["warehouse", "status", "priority", "customer_type", "vehicle_type"]

# --- Anomaly detection -------------------------------------------------

MIN_MONTHS_FOR_BASELINE = 3
ANOMALY_Z_SCORE_MEDIUM = 2.0
ANOMALY_Z_SCORE_HIGH = 3.0
# A warehouse-month needs at least this many complaints before its
# rate is trusted enough to compare against that warehouse's baseline.
WAREHOUSE_SURGE_MIN_MONTHLY_COMPLAINTS = 5
MAX_ANOMALIES_RETURNED = 10

# Network-wide monthly metrics scanned for anomalies: whether an
# increase ("high") or decrease ("low") is the bad direction, and the
# human-readable label used in generated anomaly text.
NETWORK_ANOMALY_METRICS: dict[str, tuple[str, str]] = {
    "complaint_rate_percentage": ("high", "complaint rate"),
    "negative_sentiment_percentage": ("high", "negative sentiment rate"),
    "average_resolution_time_hours": ("high", "average resolution time"),
    "average_satisfaction": ("low", "average post-resolution satisfaction"),
    "escalation_rate_percentage": ("high", "escalation rate"),
}

# --- Customer Health Score ------------------------------------------

# Five 0-100 sub-scores, weighted (sum to 1.0):
#
#   low_complaint_rate   100 minus a penalty for complaints as a share
#                        of shipments -- the fewer customers who
#                        complain at all, the healthier the experience
#   resolution_rate      the % of complaints actually resolved, taken
#                        directly as its score (already 0-100)
#   resolution_speed     100 minus a penalty for how many hours the
#                        average ticket takes to close
#   satisfaction         average post-resolution CSAT rescaled from a
#                        1-5 scale to 0-100
#   sentiment_quality    100 minus the network's negative-sentiment
#                        rate -- captures dissatisfaction even where
#                        no formal escalation happened
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "low_complaint_rate": 0.25,
    "resolution_rate": 0.20,
    "resolution_speed": 0.15,
    "satisfaction": 0.25,
    "sentiment_quality": 0.15,
}
COMPLAINT_RATE_PENALTY_MULTIPLIER = 3.0
RESOLUTION_TIME_PENALTY_PER_HOUR = 1.2


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------

def _prepare(complaints_df: pd.DataFrame, shipments_df: pd.DataFrame) -> pd.DataFrame:
    """Validate both inputs and join complaints to their originating shipments.

    Coerces date columns defensively, derives a `month` label from
    `complaint_date`, and left-joins each complaint to its shipment's
    warehouse/status/priority/customer_type/vehicle_type -- the
    attributes complaints.csv doesn't carry natively but every
    cross-dataset section below depends on.
    """
    if complaints_df is None or len(complaints_df) == 0:
        raise ValueError("analyze_complaints: complaints DataFrame is empty or None")
    if shipments_df is None or len(shipments_df) == 0:
        raise ValueError("analyze_complaints: shipments DataFrame is empty or None")
    validate_columns(complaints_df, REQUIRED_COMPLAINT_COLUMNS, "analyze_complaints: complaints")
    validate_columns(shipments_df, REQUIRED_SHIPMENT_COLUMNS, "analyze_complaints: shipments")

    complaints = complaints_df.copy()
    complaints["complaint_date"] = pd.to_datetime(complaints["complaint_date"], errors="coerce")
    complaints["resolved"] = complaints["resolved"].astype(bool)
    complaints["month"] = complaints["complaint_date"].dt.strftime("%Y-%m")

    merged = complaints.merge(shipments_df[SHIPMENT_JOIN_COLUMNS], on="shipment_id", how="left")
    return merged


def _shipment_month_counts(shipments_df: pd.DataFrame) -> pd.Series:
    """Shipment volume per "YYYY-MM" order month, used as a trend-rate denominator."""
    order_month = pd.to_datetime(shipments_df["order_date"], errors="coerce").dt.strftime("%Y-%m")
    return order_month.value_counts()


def _satisfaction_by(merged: pd.DataFrame, group_col: str) -> pd.Series:
    """Average post-resolution satisfaction per group, resolved tickets only."""
    return merged.loc[merged["resolved"]].groupby(group_col)["customer_satisfaction_after_resolution"].mean()


def _reindex_to_population(
    grouped: pd.DataFrame, population: pd.Series, count_columns: list[str]
) -> pd.DataFrame:
    """Expand a complaints-only groupby to include every category with zero complaints.

    `merged.groupby(dimension)` only contains categories that received
    at least one complaint. A warehouse/region/segment with zero
    complaints is the *best* possible outcome and must still appear
    in the ranking, not silently vanish -- so every ranking function
    reindexes onto the full population (e.g. every warehouse that
    shipped at least one order) before computing rates.
    """
    full = grouped.reindex(population.index)
    for column in count_columns:
        full[column] = full[column].fillna(0)
    return full


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(merged: pd.DataFrame, shipments_df: pd.DataFrame) -> dict[str, Any]:
    """Fleet-wide headline numbers for an executive snapshot of customer experience."""
    total_complaints = len(merged)
    total_shipments = len(shipments_df)
    resolved_count = int(merged["resolved"].sum())
    unresolved_count = total_complaints - resolved_count
    negative_count = int((merged["sentiment"] == NEGATIVE_SENTIMENT_LABEL).sum())
    resolved_satisfaction = merged.loc[merged["resolved"], "customer_satisfaction_after_resolution"]

    return {
        "total_complaints": total_complaints,
        "total_shipments": total_shipments,
        "complaint_rate_percentage": _percentage(total_complaints, total_shipments),
        "resolved_complaints": resolved_count,
        "unresolved_complaints": unresolved_count,
        "resolution_rate_percentage": _percentage(resolved_count, total_complaints),
        "average_resolution_time_hours": _safe_round(merged["resolution_time_hours"].mean()),
        "average_satisfaction_after_resolution": _safe_round(resolved_satisfaction.mean()),
        "negative_sentiment_percentage": _percentage(negative_count, total_complaints),
    }


# --------------------------------------------------------------------------
# Complaint Intelligence
# --------------------------------------------------------------------------

def _compute_complaint_intelligence(merged: pd.DataFrame) -> dict[str, Any]:
    """Category distribution: what customers are actually complaining about."""
    total = len(merged)
    counts = merged["category"].value_counts()
    distribution = [
        {"category": category, "count": int(count), "percentage": _percentage(count, total)}
        for category, count in counts.items()
    ]

    return {
        "category_distribution": distribution,
        "most_common_category": distribution[0]["category"] if distribution else None,
        "least_common_category": distribution[-1]["category"] if distribution else None,
    }


# --------------------------------------------------------------------------
# Escalation Intelligence
# --------------------------------------------------------------------------

def _compute_escalation_intelligence(merged: pd.DataFrame) -> dict[str, Any]:
    """Escalation level distribution and whether escalation correlates with dissatisfaction."""
    total = len(merged)
    level_counts = merged["escalation_level"].value_counts().sort_index()
    distribution = [
        {"escalation_level": int(level), "count": int(count), "percentage": _percentage(count, total)}
        for level, count in level_counts.items()
    ]

    by_level = merged.groupby("escalation_level").agg(
        total=("complaint_id", "count"),
        resolved_count=("resolved", "sum"),
    )
    by_level["resolution_rate_percentage"] = by_level.apply(
        lambda r: _percentage(r["resolved_count"], r["total"]), axis=1
    )
    by_level["avg_satisfaction"] = _satisfaction_by(merged, "escalation_level")
    by_level = by_level.reset_index().sort_values("escalation_level")

    performance_by_level = to_records(
        by_level, ["escalation_level", "total", "resolution_rate_percentage", "avg_satisfaction"]
    )

    satisfaction_sequence = [
        row["avg_satisfaction"] for row in performance_by_level if row["avg_satisfaction"] is not None
    ]
    higher_escalation_reduces_satisfaction = (
        len(satisfaction_sequence) > 1
        and all(a >= b for a, b in zip(satisfaction_sequence, satisfaction_sequence[1:]))
    )

    return {
        "escalation_level_distribution": distribution,
        "performance_by_level": performance_by_level,
        "higher_escalation_reduces_satisfaction": higher_escalation_reduces_satisfaction,
    }


# --------------------------------------------------------------------------
# Rankings: Warehouse / Regional / Customer Segment / Resolution Team
# --------------------------------------------------------------------------

def _compute_warehouse_ranking(merged: pd.DataFrame, shipments_df: pd.DataFrame) -> dict[str, Any]:
    """Per-warehouse complaint scorecard, ranked best (lowest complaint rate) to worst."""
    population = shipments_df.groupby("warehouse")["shipment_id"].count()
    grouped = merged.groupby("warehouse").agg(
        total_complaints=("complaint_id", "count"),
        avg_resolution_time_hours=("resolution_time_hours", "mean"),
        negative_count=("sentiment", lambda s: (s == NEGATIVE_SENTIMENT_LABEL).sum()),
    )
    grouped = _reindex_to_population(grouped, population, ["total_complaints", "negative_count"])
    grouped["avg_satisfaction"] = _satisfaction_by(merged, "warehouse")
    grouped["complaint_rate_percentage"] = rate_against_population(grouped["total_complaints"], population)
    grouped["negative_sentiment_percentage"] = grouped.apply(
        lambda r: _percentage(r["negative_count"], r["total_complaints"]), axis=1
    )
    grouped = grouped.reset_index()

    ranked = grouped.sort_values(
        ["complaint_rate_percentage", "negative_sentiment_percentage"], ascending=[True, True]
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)

    columns = [
        "rank", "warehouse", "total_complaints", "complaint_rate_percentage",
        "avg_resolution_time_hours", "avg_satisfaction", "negative_sentiment_percentage",
    ]
    records = to_records(ranked, columns)

    return {
        "ranked_warehouses": records,
        "best_warehouse": records[0]["warehouse"] if records else None,
        "worst_warehouse": records[-1]["warehouse"] if records else None,
    }


def _compute_regional_ranking(merged: pd.DataFrame, shipments_df: pd.DataFrame) -> dict[str, Any]:
    """Per-region complaint scorecard, ranked best to worst, with the highest-risk region flagged."""
    population = shipments_df.groupby("region")["shipment_id"].count()
    grouped = merged.groupby("region").agg(
        complaint_count=("complaint_id", "count"),
        avg_resolution_time_hours=("resolution_time_hours", "mean"),
    )
    grouped = _reindex_to_population(grouped, population, ["complaint_count"])
    grouped["avg_satisfaction"] = _satisfaction_by(merged, "region")
    grouped["complaint_rate_percentage"] = rate_against_population(grouped["complaint_count"], population)
    grouped = grouped.reset_index()

    ranked = grouped.sort_values("complaint_rate_percentage", ascending=True).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)

    columns = ["rank", "region", "complaint_count", "complaint_rate_percentage", "avg_resolution_time_hours", "avg_satisfaction"]
    records = to_records(ranked, columns)

    return {
        "ranked_regions": records,
        "highest_risk_region": records[-1]["region"] if records else None,
    }


def _compute_customer_segment_ranking(merged: pd.DataFrame, shipments_df: pd.DataFrame) -> dict[str, Any]:
    """Per-tier complaint scorecard (Retail/SME/Enterprise), ranked to find the best-served tier."""
    population = shipments_df.groupby("customer_type")["shipment_id"].count()
    grouped = merged.groupby("customer_type").agg(
        total_complaints=("complaint_id", "count"),
        avg_resolution_time_hours=("resolution_time_hours", "mean"),
        escalated=("escalation_level", lambda s: (s >= ESCALATED_LEVEL_THRESHOLD).sum()),
    )
    grouped = _reindex_to_population(grouped, population, ["total_complaints", "escalated"])
    grouped["avg_satisfaction"] = _satisfaction_by(merged, "customer_type")
    grouped["complaint_rate_percentage"] = rate_against_population(grouped["total_complaints"], population)
    grouped["escalation_percentage"] = grouped.apply(
        lambda r: _percentage(r["escalated"], r["total_complaints"]), axis=1
    )
    grouped = grouped.reset_index()

    ranked = grouped.sort_values(
        ["complaint_rate_percentage", "avg_satisfaction"], ascending=[True, False]
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)

    columns = [
        "rank", "customer_type", "total_complaints", "complaint_rate_percentage",
        "avg_resolution_time_hours", "avg_satisfaction", "escalation_percentage",
    ]
    records = to_records(ranked, columns)

    return {
        "ranked_segments": records,
        "best_served_segment": records[0]["customer_type"] if records else None,
        "worst_served_segment": records[-1]["customer_type"] if records else None,
    }


def _compute_resolution_team_ranking(merged: pd.DataFrame) -> dict[str, Any]:
    """Per-team scorecard, ranked by satisfaction delivered then resolution speed.

    Unlike warehouse/region/segment, teams have no shipment population
    to rate against -- a team's workload is entirely defined by which
    tickets get routed to it -- so this ranks on service quality
    (satisfaction, speed, escalation) rather than a volume-based rate.
    """
    grouped = merged.groupby("assigned_team").agg(
        tickets_handled=("complaint_id", "count"),
        avg_resolution_time_hours=("resolution_time_hours", "mean"),
        escalated=("escalation_level", lambda s: (s >= ESCALATED_LEVEL_THRESHOLD).sum()),
    )
    grouped["avg_satisfaction"] = _satisfaction_by(merged, "assigned_team")
    grouped["escalation_rate_percentage"] = grouped.apply(
        lambda r: _percentage(r["escalated"], r["tickets_handled"]), axis=1
    )
    grouped = grouped.reset_index()

    ranked = grouped.sort_values(
        ["avg_satisfaction", "avg_resolution_time_hours"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)

    columns = [
        "rank", "assigned_team", "tickets_handled", "avg_resolution_time_hours",
        "avg_satisfaction", "escalation_rate_percentage",
    ]
    records = to_records(ranked, columns)

    return {
        "ranked_teams": records,
        "highest_performing_team": records[0]["assigned_team"] if records else None,
        "lowest_performing_team": records[-1]["assigned_team"] if records else None,
    }


# --------------------------------------------------------------------------
# Cross-Dataset Intelligence
# --------------------------------------------------------------------------

def _compute_cross_dataset_intelligence(merged: pd.DataFrame, shipments_df: pd.DataFrame) -> dict[str, Any]:
    """Complaint rate by every shipment attribute, plus the business questions it answers.

    This is the section that only exists because complaints were
    joined to shipments: complaints.csv alone cannot say anything
    about delivery status, priority, customer tier, or vehicle type.
    """
    by_dimension: dict[str, list[dict[str, Any]]] = {}
    for dimension in CROSS_DATASET_DIMENSIONS:
        counts = merged.groupby(dimension)["complaint_id"].count()
        population = shipments_df.groupby(dimension)["shipment_id"].count()
        rate = rate_against_population(counts, population)

        dim_df = pd.DataFrame({"complaint_count": counts, "complaint_rate_percentage": rate})
        dim_df["complaint_count"] = dim_df["complaint_count"].fillna(0)
        dim_df = dim_df.reset_index().rename(columns={dimension: "value"})
        dim_df = dim_df.sort_values("complaint_rate_percentage", ascending=False).reset_index(drop=True)
        by_dimension[dimension] = to_records(dim_df, ["value", "complaint_count", "complaint_rate_percentage"])

    status_rates = {row["value"]: row["complaint_rate_percentage"] for row in by_dimension["status"]}
    delayed_rate = status_rates.get("Delayed", 0.0)
    delivered_rate = status_rates.get("Delivered", 0.0)
    delay_complaint_rate_multiplier = round(delayed_rate / delivered_rate, 1) if delivered_rate else None

    customer_rates = [row["complaint_rate_percentage"] for row in by_dimension["customer_type"]]
    customer_tier_spread = round(max(customer_rates) - min(customer_rates), 1) if customer_rates else 0.0

    warehouse_counts = {row["value"]: row["complaint_count"] for row in by_dimension["warehouse"]}
    total_complaints = sum(warehouse_counts.values())
    largest_warehouse = max(warehouse_counts, key=warehouse_counts.get) if warehouse_counts else None
    largest_warehouse_share = (
        _percentage(warehouse_counts[largest_warehouse], total_complaints) if largest_warehouse else None
    )

    return {
        "complaint_rate_by_warehouse": by_dimension["warehouse"],
        "complaint_rate_by_status": by_dimension["status"],
        "complaint_rate_by_priority": by_dimension["priority"],
        "complaint_rate_by_customer_type": by_dimension["customer_type"],
        "complaint_rate_by_vehicle_type": by_dimension["vehicle_type"],
        "insights": {
            "highest_complaint_status": by_dimension["status"][0]["value"] if by_dimension["status"] else None,
            "delayed_vs_delivered_complaint_rate_multiplier": delay_complaint_rate_multiplier,
            "customer_tier_complaint_rate_spread_percentage_points": customer_tier_spread,
            "warehouse_with_largest_complaint_share": largest_warehouse,
            "largest_complaint_share_percentage": largest_warehouse_share,
        },
    }


# --------------------------------------------------------------------------
# Monthly Trends
# --------------------------------------------------------------------------

def _compute_monthly_trends(merged: pd.DataFrame, shipments_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Month-over-month customer experience trend, suitable for a dashboard line chart."""
    grouped = merged.groupby("month").agg(
        complaint_volume=("complaint_id", "count"),
        avg_resolution_time_hours=("resolution_time_hours", "mean"),
        negative_count=("sentiment", lambda s: (s == NEGATIVE_SENTIMENT_LABEL).sum()),
    )
    grouped["avg_satisfaction"] = _satisfaction_by(merged, "month")
    grouped["shipment_volume"] = _shipment_month_counts(shipments_df)
    grouped = grouped.reset_index().sort_values("month")

    trends = []
    for _, row in grouped.iterrows():
        shipment_volume = row["shipment_volume"]
        trends.append({
            "month": row["month"],
            "complaint_volume": int(row["complaint_volume"]),
            "complaint_rate_percentage": (
                _percentage(row["complaint_volume"], shipment_volume) if pd.notna(shipment_volume) else None
            ),
            "average_resolution_time_hours": _safe_round(row["avg_resolution_time_hours"]),
            "average_satisfaction": _safe_round(row["avg_satisfaction"]),
            "negative_sentiment_percentage": _percentage(row["negative_count"], row["complaint_volume"]),
        })
    return trends


# --------------------------------------------------------------------------
# Anomaly Detection
# --------------------------------------------------------------------------

def _detect_monthly_network_anomalies(
    monthly_trends: list[dict[str, Any]], merged: pd.DataFrame
) -> list[dict[str, Any]]:
    """Flag months where network-wide customer experience is a statistical outlier.

    Reuses the already-computed monthly trend series and adds
    escalation rate, then z-scores every metric against the network's
    own historical distribution -- the same approach
    analysis/delivery.py uses for warehouse-month anomalies, applied
    network-wide here because complaint volume is too sparse to
    reliably split by warehouse and month at the same time.
    """
    if len(monthly_trends) < MIN_MONTHS_FOR_BASELINE:
        return []

    trend_df = pd.DataFrame(monthly_trends)
    escalation_counts = merged.groupby("month")["escalation_level"].apply(
        lambda s: (s >= ESCALATED_LEVEL_THRESHOLD).sum()
    )
    complaint_counts = merged.groupby("month")["complaint_id"].count()
    escalation_rate = rate_against_population(escalation_counts, complaint_counts)
    trend_df["escalation_rate_percentage"] = trend_df["month"].map(escalation_rate)
    trend_df = trend_df.sort_values("month").reset_index(drop=True)

    fired: dict[int, list[tuple[str, float]]] = {}
    for metric, (direction, _label) in NETWORK_ANOMALY_METRICS.items():
        z = _zscores(trend_df[metric].astype(float))
        for idx, z_value in z.items():
            if pd.isna(z_value):
                continue
            triggered = (
                (direction == "high" and z_value >= ANOMALY_Z_SCORE_MEDIUM)
                or (direction == "low" and z_value <= -ANOMALY_Z_SCORE_MEDIUM)
            )
            if triggered:
                fired.setdefault(idx, []).append((metric, z_value))

    anomalies: list[dict[str, Any]] = []
    for idx, hits in fired.items():
        row = trend_df.loc[idx]
        worst_z = max(abs(z) for _metric, z in hits)
        severity = "HIGH" if worst_z >= ANOMALY_Z_SCORE_HIGH else "MEDIUM"
        metric_summary = "; ".join(
            f"{NETWORK_ANOMALY_METRICS[metric][1]} at {row[metric]:.2f}" for metric, _z in hits
        )
        anomalies.append({
            "severity": severity,
            "category": "Customer Experience",
            "title": f"Customer experience deterioration in {row['month']}",
            "description": (
                f"Network-wide customer experience deviates sharply from its own historical "
                f"baseline in {row['month']}: {metric_summary}, across {int(row['complaint_volume'])} complaints."
            ),
            "recommended_action": (
                f"Cross-reference delivery performance for {row['month']} (see analyze_deliveries's "
                "anomalies and monthly_trends) to identify whether a specific warehouse, carrier, "
                "or product issue is driving the shift."
            ),
            "month": row["month"],
        })
    return anomalies


def _detect_warehouse_complaint_surges(merged: pd.DataFrame, shipments_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag warehouse-months where complaint rate is a statistical outlier vs. that warehouse's own year."""
    complaint_counts = merged.groupby(["warehouse", "month"])["complaint_id"].count().rename("complaint_count")
    shipments_with_month = shipments_df.assign(
        month=pd.to_datetime(shipments_df["order_date"], errors="coerce").dt.strftime("%Y-%m")
    )
    shipment_counts = shipments_with_month.groupby(["warehouse", "month"])["shipment_id"].count().rename("shipment_count")

    combined = pd.concat([complaint_counts, shipment_counts], axis=1).fillna(0).reset_index()
    combined["complaint_rate_percentage"] = combined.apply(
        lambda r: _percentage(r["complaint_count"], r["shipment_count"]), axis=1
    )
    combined = combined[combined["complaint_count"] >= WAREHOUSE_SURGE_MIN_MONTHLY_COMPLAINTS]

    anomalies: list[dict[str, Any]] = []
    for warehouse, group in combined.groupby("warehouse"):
        if len(group) < MIN_MONTHS_FOR_BASELINE:
            continue
        group = group.sort_values("month").reset_index(drop=True)
        z = _zscores(group["complaint_rate_percentage"])
        for idx, z_value in z.items():
            if z_value < ANOMALY_Z_SCORE_MEDIUM:
                continue
            row = group.loc[idx]
            severity = "HIGH" if z_value >= ANOMALY_Z_SCORE_HIGH else "MEDIUM"
            anomalies.append({
                "severity": severity,
                "category": "Warehouse",
                "title": f"{warehouse} complaint surge in {row['month']}",
                "description": (
                    f"{warehouse}'s complaint rate hit {row['complaint_rate_percentage']:.1f}% of shipments "
                    f"in {row['month']} ({z_value:.1f} standard deviations above its own historical average), "
                    f"from {int(row['complaint_count'])} complaints."
                ),
                "recommended_action": (
                    f"Cross-reference {warehouse}'s delivery performance for {row['month']} in the delivery "
                    "analytics engine to confirm whether an operational event is driving the surge."
                ),
                "warehouse": warehouse,
                "month": row["month"],
            })
    return anomalies


def _detect_anomalies(
    monthly_trends: list[dict[str, Any]], merged: pd.DataFrame, shipments_df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Run every anomaly detector and return the most severe findings first."""
    anomalies = _detect_monthly_network_anomalies(monthly_trends, merged) + _detect_warehouse_complaint_surges(
        merged, shipments_df
    )
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Customer Health Score
# --------------------------------------------------------------------------

def _compute_health_score(summary: dict[str, Any]) -> dict[str, Any]:
    """Blend the executive summary into a single 0-100 Customer Health Score.

    See HEALTH_SCORE_WEIGHTS above for the full methodology. Every
    component score and the weights used are returned alongside the
    final number so the score is always auditable, never a black box.
    """
    complaint_rate = summary["complaint_rate_percentage"]
    low_complaint_rate_score = max(0.0, 100 - complaint_rate * COMPLAINT_RATE_PENALTY_MULTIPLIER)

    resolution_rate_score = summary["resolution_rate_percentage"]

    avg_hours = summary["average_resolution_time_hours"] or 0.0
    resolution_speed_score = max(0.0, 100 - avg_hours * RESOLUTION_TIME_PENALTY_PER_HOUR)

    avg_satisfaction = summary["average_satisfaction_after_resolution"] or 0.0
    satisfaction_score = min(100.0, (avg_satisfaction / MAX_SATISFACTION_RATING) * 100)

    sentiment_quality_score = max(0.0, 100 - summary["negative_sentiment_percentage"])

    components = {
        "low_complaint_rate": round(low_complaint_rate_score, 1),
        "resolution_rate": round(resolution_rate_score, 1),
        "resolution_speed": round(resolution_speed_score, 1),
        "satisfaction": round(satisfaction_score, 1),
        "sentiment_quality": round(sentiment_quality_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            "Weighted blend of five 0-100 components: low_complaint_rate (100 minus 3 points per "
            "percentage point of shipments that generate a complaint, weight 0.25), resolution_rate "
            "(% of complaints resolved, weight 0.20), resolution_speed (100 minus 1.2 points per "
            "average hour to resolve a ticket, weight 0.15), satisfaction (average post-resolution "
            "CSAT rescaled from a 5-point to a 100-point scale, weight 0.25), and sentiment_quality "
            "(100 minus the network's negative-sentiment percentage, weight 0.15). This is an annual "
            "figure measuring sustained customer experience health, not a single incident -- see "
            "'anomalies' for acute, dated problems."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_complaints(complaints_df: pd.DataFrame, shipments_df: pd.DataFrame) -> dict[str, Any]:
    """Run the full Customer Experience Intelligence suite a CCO/COO/CEO needs.

    Joins complaints to their originating shipments on `shipment_id`
    to answer cross-dataset questions complaints.csv cannot answer
    alone (e.g. "does a delayed delivery make a complaint more
    likely?"), then computes an executive summary, complaint-category
    and escalation-level breakdowns, ranked warehouse/regional/
    customer-segment/support-team scorecards, monthly trends,
    statistically-detected anomalies, and an overall Customer Health
    Score.

    Args:
        complaints_df: The complaints DataFrame loaded from
            data/complaints.csv (parsed or unparsed dates accepted).
        shipments_df: The shipments DataFrame loaded from
            data/shipments.csv, used both as the join target and as
            the population denominator for every complaint rate.

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},           # Executive Summary
                "kpis": {
                    "complaint_intelligence": {...},
                    "escalation_intelligence": {...},
                    "cross_dataset_intelligence": {...},
                },
                "rankings": {
                    "warehouse": {...},
                    "regional": {...},
                    "customer_segment": {...},
                    "resolution_team": {...},
                },
                "trends": {"monthly": [...]},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

        suitable for rendering in a dashboard or passed directly to an
        LLM as report-generation context.

    Raises:
        ValueError: If either DataFrame is empty/None or missing
            columns this module depends on.
    """
    merged = _prepare(complaints_df, shipments_df)

    summary = _compute_summary(merged, shipments_df)
    monthly_trends = _compute_monthly_trends(merged, shipments_df)
    health_score = _compute_health_score(summary)

    result = {
        "summary": summary,
        "kpis": {
            "complaint_intelligence": _compute_complaint_intelligence(merged),
            "escalation_intelligence": _compute_escalation_intelligence(merged),
            "cross_dataset_intelligence": _compute_cross_dataset_intelligence(merged, shipments_df),
        },
        "rankings": {
            "warehouse": _compute_warehouse_ranking(merged, shipments_df),
            "regional": _compute_regional_ranking(merged, shipments_df),
            "customer_segment": _compute_customer_segment_ranking(merged, shipments_df),
            "resolution_team": _compute_resolution_team_ranking(merged),
        },
        "trends": {
            "monthly": monthly_trends,
        },
        "anomalies": _detect_anomalies(monthly_trends, merged, shipments_df),
        "health_score": health_score,
        "methodology": {
            "complaint_rate_definition": (
                "Complaint rate is always complaints divided by total shipments for the same "
                "segment (warehouse, region, customer tier, priority, vehicle type), sourced from "
                "shipments.csv via a shipment_id join -- never complaints as a share of other "
                "complaints. Where an absolute contribution matters more than a fair per-shipment "
                "rate (e.g. which warehouse produces the most unhappy customers overall), that is "
                "reported separately as a 'share', not a 'rate'."
            ),
            "join_strategy": (
                "Every cross-dataset metric joins complaints.csv to shipments.csv on shipment_id "
                "to attribute each complaint to its warehouse, delivery status, priority, customer "
                "tier, and vehicle type, since complaints.csv does not carry those fields natively."
            ),
            "anomaly_detection": (
                "Anomalies are detected by z-scoring monthly metrics against their own historical "
                "baseline -- network-wide for overall customer-experience metrics, and per-warehouse "
                "for complaint surges -- never by referencing a specific warehouse or month in code."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
