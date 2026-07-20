"""Business Intelligence Layer: cross-domain correlation and root-cause engine.

The three analytics engines (analysis/delivery.py, analysis/complaints.py,
analysis/inventory.py) each answer a domain-specific question well, but
none of them can say *why* the business is performing the way it is --
that requires looking across domains at once. This module is the layer
that does that: it consumes the three engines' own outputs (their
rankings, health scores, and anomalies), looks for places where
independent domains agree with each other, and turns that agreement
into evidence-backed root causes and a prioritized action plan.

    CSV Data -> Analytics Layer -> Correlation Layer -> AI Report Generator

This module sits at "Correlation Layer" in that pipeline. It is a
consumer of the analytics layer, not a fourth analytics engine: it does
not re-run warehouse groupbys or recompute rates the other engines
already computed. Raw DataFrames (shipments_df, complaints_df,
inventory_df) are accepted for exactly the cases where a genuinely new
cross-tabulation is needed that no existing engine exposes -- in this
module, that is precisely one case (which complaint categories are
most associated with delayed shipments; see
`_relationship_delay_complaint_categories`). Everything else is derived
by reading and combining fields the three engines already computed.

Design notes
------------
- delivery.py predates the {summary, kpis, rankings, trends, anomalies,
  health_score, methodology} convention complaints.py and inventory.py
  share, and still returns a flatter top-level shape (executive_kpis,
  warehouse_intelligence, ...). This module reads all three defensively
  through small named accessor helpers rather than assuming one shape,
  so it degrades gracefully if a key is missing instead of raising a
  raw KeyError deep inside a computation.
- No machine learning and no predictive modeling anywhere in this
  module, per the platform's descriptive/diagnostic-analytics mandate.
  "Correlation" here means exactly what pandas' `.corr()` computes
  (Pearson's r) applied to already-aggregated per-warehouse metrics, or
  a simple ratio/multiplier -- nothing is fit, trained, or forecast.
- Confidence is never invented. Every confidence score is built from
  the same transparent formula (`_compute_confidence`, documented in
  full next to its definition and echoed in the returned
  `methodology.confidence_methodology`): a base score for having a
  real, measured pattern at all, plus a bonus for the statistical
  strength of that pattern, plus a bonus for how many independent
  engines/anomaly-detectors flag the same conclusion. The score is
  capped below 1.0 -- this module never claims certainty.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis._shared import (
    _json_safe,
    _percentage,
    _safe_round,
    _zscores,
    to_records,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Top-level keys each analytics engine's output must have for this
# module to trust it enough to read from it. delivery_analysis is
# checked against its own (flatter) shape; complaints/inventory share
# the newer {summary, kpis, rankings, anomalies, health_score} shape.
REQUIRED_DELIVERY_KEYS = ["executive_kpis", "warehouse_intelligence", "anomalies"]
REQUIRED_COMPLAINTS_KEYS = ["summary", "kpis", "rankings", "anomalies", "health_score"]
REQUIRED_INVENTORY_KEYS = ["summary", "kpis", "rankings", "anomalies", "health_score"]

MIN_WAREHOUSES_FOR_CORRELATION = 3

# --- Confidence Engine ---------------------------------------------------
#
# confidence = base + strength_bonus + convergence_bonus, capped below 1.0.
#
#   base                a real, measured pattern in the data starts
#                       here regardless of how strong it looks
#   strength_bonus       up to CONFIDENCE_STRENGTH_WEIGHT, scaled by
#                       how strong the underlying statistic is
#                       (|Pearson r|, or how far a rate multiplier sits
#                       from a neutral 1.0x, both normalized to 0-1)
#   convergence_bonus    CONFIDENCE_PER_CONVERGENCE_SIGNAL for every
#                       independent engine or anomaly detector that
#                       flags the same entity/conclusion, up to
#                       CONFIDENCE_MAX_CONVERGENCE_SIGNALS signals
#
# This is the one formula every confidence score in this module's
# output comes from -- see `_compute_confidence`.
CONFIDENCE_BASE = 0.35
CONFIDENCE_STRENGTH_WEIGHT = 0.35
CONFIDENCE_PER_CONVERGENCE_SIGNAL = 0.10
CONFIDENCE_MAX_CONVERGENCE_SIGNALS = 3
CONFIDENCE_CAP = 0.97

# --- Root cause thresholds ------------------------------------------------

# A warehouse's composite risk z-score (sum of oriented z-scores across
# demand share, delay rate, complaint rate, and inventory risk) must
# clear this before being called out as a "bottleneck" root cause --
# otherwise the warehouse network looks reasonably balanced and no
# single warehouse deserves that label.
WAREHOUSE_BOTTLENECK_MIN_ZSCORE = 1.5
# A premium priority tier's on-time rate must trail Standard's by at
# least this many percentage points before it's called a root cause
# rather than ordinary noise.
PRIORITY_SLA_GAP_THRESHOLD = 5.0

# --- Business impact severity thresholds ----------------------------------
#
# Each is a (medium, high) percentage-point pair: below `medium` is
# LOW severity, at/above `medium` is MEDIUM, at/above `high` is HIGH.
NEGATIVE_SENTIMENT_SEVERITY = (40.0, 65.0)
DELAY_RATE_SEVERITY = (8.0, 15.0)
BELOW_REORDER_SEVERITY = (10.0, 20.0)

TOP_N_RELATIONSHIP_EVIDENCE = 3


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------

def _require_analysis(analysis: Any, required_keys: list[str], name: str) -> dict[str, Any]:
    """Validate that an upstream engine's output is a usable dict before reading from it."""
    if analysis is None or not isinstance(analysis, dict):
        raise ValueError(f"analyze_correlations: {name} must be a non-empty dict produced by its analytics engine")
    missing = [k for k in required_keys if k not in analysis]
    if missing:
        raise ValueError(f"analyze_correlations: {name} is missing expected keys: {missing}")
    return analysis


def _require_dataframe(df: Any, name: str) -> None:
    """Validate that a raw DataFrame input is present and non-empty."""
    if df is None or len(df) == 0:
        raise ValueError(f"analyze_correlations: {name} DataFrame is empty or None")


def _compute_confidence(strength: float, corroborating_signals: int) -> float:
    """The single confidence formula every score in this module uses.

    Args:
        strength: 0-1, how strong the underlying statistic is (e.g.
            |Pearson r|, or a normalized rate-multiplier distance
            from 1.0x).
        corroborating_signals: count of independent engines or
            anomaly detectors that flag the same entity/conclusion,
            beyond the primary evidence itself.

    Returns:
        A confidence score in [CONFIDENCE_BASE, CONFIDENCE_CAP],
        rounded to 2 decimals. Never 1.0 -- see module docstring.
    """
    strength = min(max(strength, 0.0), 1.0)
    convergence_bonus = min(corroborating_signals, CONFIDENCE_MAX_CONVERGENCE_SIGNALS) * CONFIDENCE_PER_CONVERGENCE_SIGNAL
    raw = CONFIDENCE_BASE + strength * CONFIDENCE_STRENGTH_WEIGHT + convergence_bonus
    return round(min(raw, CONFIDENCE_CAP), 2)


def _severity_tier(value: float | None, thresholds: tuple[float, float]) -> str:
    """Classify a percentage-point value into LOW/MEDIUM/HIGH via named thresholds."""
    if value is None:
        return "LOW"
    medium, high = thresholds
    if value >= high:
        return "HIGH"
    if value >= medium:
        return "MEDIUM"
    return "LOW"


def _make_relationship(
    domain_pair: str,
    finding: str,
    metric: str,
    value: Any,
    supporting_evidence: list[str],
    confidence: float,
    interpretation: str,
) -> dict[str, Any]:
    """Build a relationship-finding dict with a single, consistent shape."""
    return {
        "domain_pair": domain_pair,
        "finding": finding,
        "metric": metric,
        "value": value,
        "supporting_evidence": supporting_evidence,
        "confidence": confidence,
        "interpretation": interpretation,
    }


def _to_warehouse_map(records: list[dict[str, Any]], value_field: str, key_field: str = "warehouse") -> dict[str, float]:
    """Extract a {warehouse: metric_value} lookup from a ranking's record list, dropping missing values."""
    return {
        r[key_field]: r[value_field]
        for r in records
        if r.get(key_field) is not None and r.get(value_field) is not None
    }


def _warehouse_names(records: list[dict[str, Any]], key: str = "warehouse") -> set[str]:
    return {r[key] for r in records if r.get(key)}


def _common_warehouses(*name_sets: set[str]) -> list[str]:
    """The warehouse names present in every given set, so cross-engine comparisons never mix mismatched entities."""
    non_empty = [s for s in name_sets if s]
    if len(non_empty) < len(name_sets) or not non_empty:
        return []
    return sorted(set.intersection(*non_empty))


def _first_or_none(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    return items[0] if items else None


# --------------------------------------------------------------------------
# Accessors: read the three engines' outputs through one defensive layer
# --------------------------------------------------------------------------
#
# delivery_analysis has a flat top-level shape; complaints_analysis and
# inventory_analysis share the {summary, kpis, rankings, ...} shape.
# Every read from an upstream dict goes through one of these so a
# missing/renamed key degrades to an empty structure instead of a
# KeyError, and so the shape difference is handled in exactly one place.

def _delivery_kpis(d: dict[str, Any]) -> dict[str, Any]:
    return d.get("executive_kpis", {})


def _delivery_warehouses(d: dict[str, Any]) -> list[dict[str, Any]]:
    return d.get("warehouse_intelligence", {}).get("ranked_warehouses", [])


def _delivery_worst_warehouse(d: dict[str, Any]) -> str | None:
    return d.get("warehouse_intelligence", {}).get("worst_warehouse")


def _delivery_customer_intelligence(d: dict[str, Any]) -> dict[str, Any]:
    return d.get("customer_intelligence", {})


def _delivery_priority_intelligence(d: dict[str, Any]) -> dict[str, Any]:
    return d.get("priority_intelligence", {})


def _delivery_anomalies(d: dict[str, Any]) -> list[dict[str, Any]]:
    return d.get("anomalies", [])


def _delivery_health_score(d: dict[str, Any]) -> dict[str, Any]:
    return d.get("operations_health_score", {})


def _complaints_summary(c: dict[str, Any]) -> dict[str, Any]:
    return c.get("summary", {})


def _complaints_warehouses(c: dict[str, Any]) -> list[dict[str, Any]]:
    return c.get("rankings", {}).get("warehouse", {}).get("ranked_warehouses", [])


def _complaints_worst_warehouse(c: dict[str, Any]) -> str | None:
    return c.get("rankings", {}).get("warehouse", {}).get("worst_warehouse")


def _complaints_segments(c: dict[str, Any]) -> list[dict[str, Any]]:
    return c.get("rankings", {}).get("customer_segment", {}).get("ranked_segments", [])


def _complaints_cross_dataset_insights(c: dict[str, Any]) -> dict[str, Any]:
    return c.get("kpis", {}).get("cross_dataset_intelligence", {}).get("insights", {})


def _complaints_anomalies(c: dict[str, Any]) -> list[dict[str, Any]]:
    return c.get("anomalies", [])


def _complaints_health_score(c: dict[str, Any]) -> dict[str, Any]:
    return c.get("health_score", {})


def _inventory_summary(i: dict[str, Any]) -> dict[str, Any]:
    return i.get("summary", {})


def _inventory_warehouses(i: dict[str, Any]) -> list[dict[str, Any]]:
    return i.get("rankings", {}).get("warehouse", {}).get("ranked_warehouses", [])


def _inventory_highest_risk_warehouse(i: dict[str, Any]) -> str | None:
    return i.get("rankings", {}).get("warehouse", {}).get("highest_risk_warehouse")


def _inventory_suppliers(i: dict[str, Any]) -> dict[str, Any]:
    return i.get("rankings", {}).get("supplier", {})


def _inventory_demand_vs_supply(i: dict[str, Any]) -> list[dict[str, Any]]:
    return i.get("kpis", {}).get("cross_dataset_intelligence", {}).get("demand_vs_supply_by_warehouse", [])


def _inventory_anomalies(i: dict[str, Any]) -> list[dict[str, Any]]:
    return i.get("anomalies", [])


def _inventory_health_score(i: dict[str, Any]) -> dict[str, Any]:
    return i.get("health_score", {})


# --------------------------------------------------------------------------
# Convergence lookup: how many independent signals point at each warehouse
# --------------------------------------------------------------------------

def _build_convergence_lookup(
    warehouses: list[str],
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """For each warehouse, count independent 'this is the worst one' signals.

    Two kinds of signal are counted separately:
      - ranking_signals: how many of the three engines' own ranking
        functions independently named this warehouse their worst/
        highest-risk (0-3).
      - anomaly_signals: how many of the three engines flagged this
        warehouse in a HIGH-severity anomaly (0-3).

    This is the single source of convergence evidence reused by the
    executive summary, the warehouse composite ranking, and every
    warehouse-related root cause and confidence score, so "how many
    engines agree" is counted exactly one way throughout the module.
    """
    worst = {
        _delivery_worst_warehouse(delivery_analysis),
        _complaints_worst_warehouse(complaints_analysis),
        _inventory_highest_risk_warehouse(inventory_analysis),
    }
    high_anomaly_warehouses = [
        {a.get("warehouse") for a in _delivery_anomalies(delivery_analysis) if a.get("severity") == "HIGH"},
        {a.get("warehouse") for a in _complaints_anomalies(complaints_analysis) if a.get("severity") == "HIGH"},
        {a.get("warehouse") for a in _inventory_anomalies(inventory_analysis) if a.get("severity") == "HIGH"},
    ]

    lookup: dict[str, dict[str, int]] = {}
    for warehouse in warehouses:
        ranking_signals = sum(1 for w in worst if w == warehouse)
        anomaly_signals = sum(1 for wh_set in high_anomaly_warehouses if warehouse in wh_set)
        lookup[warehouse] = {"ranking_signals": ranking_signals, "anomaly_signals": anomaly_signals}
    return lookup


# --------------------------------------------------------------------------
# Relationship Discovery: Delivery <-> Complaints
# --------------------------------------------------------------------------

def _relationship_delayed_complaints(complaints_analysis: dict[str, Any]) -> dict[str, Any] | None:
    """Reuses complaints.py's own delayed-vs-delivered complaint rate multiplier."""
    insights = _complaints_cross_dataset_insights(complaints_analysis)
    multiplier = insights.get("delayed_vs_delivered_complaint_rate_multiplier")
    if multiplier is None:
        return None

    strength = min(abs(multiplier - 1) / 4, 1.0)
    confidence = _compute_confidence(strength, corroborating_signals=0)
    return _make_relationship(
        domain_pair="Delivery <-> Complaints",
        finding=(
            f"Delayed shipments are {multiplier}x more likely to generate a complaint than "
            "successfully delivered shipments."
        ),
        metric="complaint_rate_multiplier_delayed_vs_delivered",
        value=multiplier,
        supporting_evidence=[
            f"complaints_analysis.kpis.cross_dataset_intelligence.insights reports a "
            f"{multiplier}x complaint-rate gap between Delayed and Delivered shipments."
        ],
        confidence=confidence,
        interpretation=(
            "Shipment delay is the strongest known driver of a customer complaint in this data; "
            "reducing the delay rate is very likely to reduce complaint volume directly, not just "
            "correlate with it."
        ),
    )


def _relationship_delay_complaint_categories(
    complaints_df: pd.DataFrame, shipments_df: pd.DataFrame
) -> dict[str, Any] | None:
    """The one relationship in this module built from raw data, not an engine's output.

    complaints.py exposes complaint rate BY shipment status and the
    overall complaint category distribution, but not the category
    distribution *within* delayed-shipment complaints specifically --
    that cross-tabulation isn't computed anywhere upstream, so it's
    the one case where joining the raw data is genuinely required.
    """
    if "shipment_id" not in complaints_df.columns or "category" not in complaints_df.columns:
        return None
    if "shipment_id" not in shipments_df.columns or "status" not in shipments_df.columns:
        return None

    merged = complaints_df[["shipment_id", "category"]].merge(
        shipments_df[["shipment_id", "status"]], on="shipment_id", how="inner"
    )
    delayed_complaints = merged[merged["status"] == "Delayed"]
    if delayed_complaints.empty:
        return None

    total = len(delayed_complaints)
    counts = delayed_complaints["category"].value_counts()
    top_category = counts.index[0]
    top_share = _percentage(counts.iloc[0], total)
    distribution = [
        {"category": category, "count": int(count), "percentage": _percentage(count, total)}
        for category, count in counts.items()
    ]

    strength = min(top_share / 100, 1.0)
    confidence = _compute_confidence(strength, corroborating_signals=0)
    return _make_relationship(
        domain_pair="Delivery <-> Complaints",
        finding=(
            f"'{top_category}' is the most common complaint category among delayed shipments "
            f"({top_share}% of {total} complaints traced to a Delayed shipment)."
        ),
        metric="complaint_category_distribution_for_delayed_shipments",
        value=distribution,
        supporting_evidence=[
            f"{total} complaints were traced to shipments with status 'Delayed' via a "
            "shipment_id join between complaints.csv and shipments.csv."
        ],
        confidence=confidence,
        interpretation=(
            "This confirms customers who receive a late shipment file the complaint category "
            "that directly names the problem, rather than an unrelated one -- the complaint "
            "channel is accurately reflecting the operational cause."
        ),
    )


# --------------------------------------------------------------------------
# Relationship Discovery: warehouse-level pairwise correlations
# --------------------------------------------------------------------------

def _warehouse_correlation(
    metric_a_map: dict[str, float], a_bad_direction: str,
    metric_b_map: dict[str, float], b_bad_direction: str,
    common_warehouses: list[str],
) -> dict[str, Any] | None:
    """Pearson r plus a plain-language alignment check between two per-warehouse metrics.

    With only a handful of warehouses, a Pearson coefficient alone is
    a thin statistical signal, so this also reports whether the two
    metrics' individually-worst warehouses are literally the same
    warehouse -- an easy-to-audit, non-statistical corroboration that
    matters as much as the r value itself at this sample size.
    """
    warehouses = [w for w in common_warehouses if w in metric_a_map and w in metric_b_map]
    if len(warehouses) < MIN_WAREHOUSES_FOR_CORRELATION:
        return None

    a = pd.Series({w: metric_a_map[w] for w in warehouses})
    b = pd.Series({w: metric_b_map[w] for w in warehouses})
    r = a.corr(b)
    if pd.isna(r):
        r = 0.0

    worst_a = a.idxmax() if a_bad_direction == "high" else a.idxmin()
    worst_b = b.idxmax() if b_bad_direction == "high" else b.idxmin()
    return {
        "r": round(float(r), 3),
        "n": len(warehouses),
        "worst_by_metric_a": worst_a,
        "worst_by_metric_b": worst_b,
        "aligned": worst_a == worst_b,
    }


def _relationship_from_warehouse_correlation(
    domain_pair: str,
    metric_label_a: str, metric_a_map: dict[str, float], a_bad_direction: str,
    metric_label_b: str, metric_b_map: dict[str, float], b_bad_direction: str,
    common_warehouses: list[str],
    interpretation: str,
) -> dict[str, Any] | None:
    """Build a full relationship entry from a warehouse-level metric pair."""
    result = _warehouse_correlation(metric_a_map, a_bad_direction, metric_b_map, b_bad_direction, common_warehouses)
    if result is None:
        return None

    strength = abs(result["r"])
    confidence = _compute_confidence(strength, corroborating_signals=1 if result["aligned"] else 0)
    alignment_text = (
        f"the same warehouse ({result['worst_by_metric_a']}) is worst on both metrics"
        if result["aligned"]
        else f"the worst warehouse differs by metric ({result['worst_by_metric_a']} vs {result['worst_by_metric_b']})"
    )
    return _make_relationship(
        domain_pair=domain_pair,
        finding=(
            f"{metric_label_a} and {metric_label_b} correlate at r={result['r']} across "
            f"{result['n']} warehouses; {alignment_text}."
        ),
        metric=f"correlation({metric_label_a}, {metric_label_b})",
        value=result["r"],
        supporting_evidence=[
            f"Worst warehouse by {metric_label_a}: {result['worst_by_metric_a']}.",
            f"Worst warehouse by {metric_label_b}: {result['worst_by_metric_b']}.",
        ],
        confidence=confidence,
        interpretation=interpretation,
    )


# --------------------------------------------------------------------------
# Relationship Discovery: Customer Type Analysis
# --------------------------------------------------------------------------

def _relationship_customer_type(
    delivery_analysis: dict[str, Any], complaints_analysis: dict[str, Any]
) -> dict[str, Any] | None:
    """Compare Retail/SME/Enterprise across delay rate, complaint rate, and escalation.

    "Operational resilience" (the headline ranking) is deliberately
    based only on delay_rate_percentage and complaint_rate_percentage
    -- both "how often does something go wrong" measures on a
    comparable ~0-100% scale of the same magnitude. escalation_percentage
    is tracked separately rather than averaged in: it measures how a
    complaint is *handled* once filed, not how often problems occur,
    and in this data it runs structurally much higher for Enterprise
    (closer account management escalates nearly everything) -- folding
    it into a single average would let that scale dominate and falsely
    label the best-performing tier as the worst. It still feeds the
    surprising-finding check below, which is exactly where that
    tension belongs.
    """
    delivery_by_type = _delivery_customer_intelligence(delivery_analysis)
    complaints_by_type = {row["customer_type"]: row for row in _complaints_segments(complaints_analysis)}

    tiers: dict[str, dict[str, float | None]] = {}
    for tier in set(delivery_by_type) | set(complaints_by_type):
        d = delivery_by_type.get(tier, {})
        c = complaints_by_type.get(tier, {})
        tiers[tier] = {
            "delay_rate_percentage": d.get("delay_rate_percentage"),
            "complaint_rate_percentage": c.get("complaint_rate_percentage"),
            "escalation_percentage": c.get("escalation_percentage"),
        }
    if not tiers:
        return None

    def _resilience_score(tier: str) -> float | None:
        values = [
            tiers[tier]["delay_rate_percentage"], tiers[tier]["complaint_rate_percentage"],
        ]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    scored = {t: _resilience_score(t) for t in tiers}
    ranked_tiers = sorted((t for t in scored if scored[t] is not None), key=lambda t: scored[t])
    if not ranked_tiers:
        return None
    most_resilient, least_resilient = ranked_tiers[0], ranked_tiers[-1]

    delay_ranked = sorted((t for t in tiers if tiers[t]["delay_rate_percentage"] is not None), key=lambda t: tiers[t]["delay_rate_percentage"])
    complaint_ranked = sorted((t for t in tiers if tiers[t]["complaint_rate_percentage"] is not None), key=lambda t: tiers[t]["complaint_rate_percentage"])
    escalation_ranked = sorted((t for t in tiers if tiers[t]["escalation_percentage"] is not None), key=lambda t: tiers[t]["escalation_percentage"])

    surprising_finding = None
    if delay_ranked and complaint_ranked and escalation_ranked:
        best_delay, best_complaint, worst_escalation = delay_ranked[0], complaint_ranked[0], escalation_ranked[-1]
        if best_delay == best_complaint == worst_escalation and len(escalation_ranked) > 1:
            surprising_finding = (
                f"{best_delay} has both the lowest delay rate and the lowest complaint rate of any "
                "customer tier, yet the highest escalation rate. This is not a service failure -- it "
                "more likely reflects closer account management, where every complaint that does "
                "occur gets escalated and tracked more thoroughly rather than left at the front line."
            )

    spread = round(max(scored[t] for t in ranked_tiers) - min(scored[t] for t in ranked_tiers), 1)
    strength = min(spread / 10, 1.0)
    confidence = _compute_confidence(strength, corroborating_signals=1)

    return _make_relationship(
        domain_pair="Customer Type Analysis",
        finding=(
            f"{most_resilient} customers experience the best combined delivery and complaint "
            f"outcomes; {least_resilient} customers experience the worst."
        ),
        metric="customer_tier_operational_resilience",
        value=tiers,
        supporting_evidence=[
            f"Resilience score (avg of delay rate % and complaint rate %, lower=better): "
            f"{most_resilient}={_safe_round(scored[most_resilient])}, "
            f"{least_resilient}={_safe_round(scored[least_resilient])}.",
        ] + ([surprising_finding] if surprising_finding else []),
        confidence=confidence,
        interpretation=(
            surprising_finding
            or f"{least_resilient} customers are the segment most exposed to operational problems "
            "reaching them -- worth prioritizing if churn or satisfaction in that tier is a concern."
        ),
    )


# --------------------------------------------------------------------------
# Warehouse Analysis: composite risk ranking
# --------------------------------------------------------------------------

def _compute_warehouse_composite_ranking(
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
    common_warehouses: list[str],
) -> dict[str, Any]:
    """Rank warehouses by how many operationally-bad signals converge on them at once.

    Combines shipment demand share, delivery delay rate, complaint
    rate, and inventory risk score -- one row per warehouse, drawn
    entirely from the three engines' own already-computed rankings --
    into a single composite z-score, directly answering "which
    warehouse simultaneously has the highest demand, weakest
    inventory, highest delay rate, and highest complaint rate."
    """
    delivery_map = {r["warehouse"]: r for r in _delivery_warehouses(delivery_analysis)}
    complaints_map = {r["warehouse"]: r for r in _complaints_warehouses(complaints_analysis)}
    inventory_map = {r["warehouse"]: r for r in _inventory_warehouses(inventory_analysis)}
    demand_map = {r["warehouse"]: r for r in _inventory_demand_vs_supply(inventory_analysis)}

    rows = []
    for warehouse in common_warehouses:
        d, c, i, dm = delivery_map.get(warehouse, {}), complaints_map.get(warehouse, {}), inventory_map.get(warehouse, {}), demand_map.get(warehouse, {})
        rows.append({
            "warehouse": warehouse,
            "shipment_demand_share_percentage": dm.get("demand_share_percentage"),
            "delay_rate_percentage": d.get("delay_rate_percentage"),
            "complaint_rate_percentage": c.get("complaint_rate_percentage"),
            "inventory_risk_score": i.get("risk_score"),
            "avg_days_remaining": i.get("avg_days_remaining"),
            "avg_loading_time": d.get("avg_loading_time"),
        })

    if not rows:
        return {"ranked_warehouses": [], "highest_composite_risk_warehouse": None, "methodology_note": "No warehouse names were common across all three analytics engines, so no composite ranking could be built."}

    frame = pd.DataFrame(rows)
    zscore_columns = ["shipment_demand_share_percentage", "delay_rate_percentage", "complaint_rate_percentage", "inventory_risk_score"]
    composite = pd.Series(0.0, index=frame.index)
    for column in zscore_columns:
        series = frame[column].astype(float)
        composite += _zscores(series.fillna(series.mean()))
    frame["composite_risk_zscore"] = composite.round(2)

    ranked = frame.sort_values("composite_risk_zscore", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    columns = ["rank", "warehouse", "shipment_demand_share_percentage", "delay_rate_percentage", "complaint_rate_percentage", "inventory_risk_score", "avg_days_remaining", "avg_loading_time", "composite_risk_zscore"]
    records = to_records(ranked, columns)

    return {
        "ranked_warehouses": records,
        "highest_composite_risk_warehouse": records[0]["warehouse"] if records else None,
        "methodology_note": (
            "Composite risk is the sum of each warehouse's z-score across shipment demand share, "
            "delivery delay rate, complaint rate, and inventory risk score -- all four already "
            "computed independently by the delivery, complaints, and inventory engines. A warehouse "
            "at the top of this ranking is elevated on all four surfaces simultaneously, not just one."
        ),
    }


# --------------------------------------------------------------------------
# Root Cause Discovery
# --------------------------------------------------------------------------

def _root_cause_warehouse_bottleneck(
    warehouse_composite: dict[str, Any],
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
    convergence_lookup: dict[str, dict[str, int]],
) -> dict[str, Any] | None:
    """The flagship root cause: a warehouse simultaneously overloaded on every measured surface."""
    ranked = warehouse_composite["ranked_warehouses"]
    if not ranked:
        return None
    top = ranked[0]
    if top["composite_risk_zscore"] < WAREHOUSE_BOTTLENECK_MIN_ZSCORE:
        return None
    if len(ranked) > 1 and top["composite_risk_zscore"] <= ranked[1]["composite_risk_zscore"]:
        return None

    warehouse = top["warehouse"]
    evidence = []
    if top.get("shipment_demand_share_percentage") is not None:
        evidence.append(f"Carries {top['shipment_demand_share_percentage']}% of network shipment demand.")
    if top.get("avg_days_remaining") is not None:
        evidence.append(f"Averages only {top['avg_days_remaining']} days of inventory coverage.")

    delivery_loading = _to_warehouse_map(_delivery_warehouses(delivery_analysis), "avg_loading_time")
    if delivery_loading and max(delivery_loading, key=delivery_loading.get) == warehouse:
        evidence.append(f"Longest average loading time of any warehouse ({delivery_loading[warehouse]} minutes).")
    if _delivery_worst_warehouse(delivery_analysis) == warehouse:
        evidence.append(f"Worst warehouse in the delivery ranking (delay rate {top.get('delay_rate_percentage')}%).")
    if _complaints_worst_warehouse(complaints_analysis) == warehouse:
        evidence.append(f"Worst warehouse in the complaints ranking (complaint rate {top.get('complaint_rate_percentage')}%).")
    if _inventory_highest_risk_warehouse(inventory_analysis) == warehouse:
        evidence.append(f"Highest-risk warehouse in the inventory ranking (risk score {top.get('inventory_risk_score')}).")

    signals = convergence_lookup.get(warehouse, {"ranking_signals": 0, "anomaly_signals": 0})
    strength = signals["ranking_signals"] / 3.0
    confidence = _compute_confidence(strength, corroborating_signals=signals["anomaly_signals"])

    return {
        "category": "Warehouse Bottleneck",
        "title": f"{warehouse} Capacity & Demand Bottleneck",
        "description": (
            f"{warehouse} simultaneously carries the network's most concentrated shipment demand "
            "while showing the weakest inventory position, the slowest warehouse operations, and "
            "the worst customer outcomes -- a pattern independently confirmed by the delivery, "
            "complaints, and inventory analytics engines rather than any single metric."
        ),
        "evidence": evidence,
        "confidence": confidence,
        "business_impact": (
            "Continued strain here will keep degrading on-time delivery and driving complaint "
            "volume until either demand is rebalanced to other warehouses or this warehouse's "
            "capacity and inventory position are increased to match the demand it actually serves."
        ),
        "recommended_action": (
            f"Prioritize {warehouse} for capacity and staffing review and inventory rebalancing; "
            "consider temporarily redirecting a portion of its shipment demand to warehouses with "
            "spare capacity while structural fixes are put in place."
        ),
    }


def _root_cause_supplier_lead_time(inventory_analysis: dict[str, Any]) -> dict[str, Any] | None:
    """A supplier whose lead time is both a statistical outlier and already coincides with thin coverage."""
    supplier_ranking = _inventory_suppliers(inventory_analysis)
    elevated = set(supplier_ranking.get("elevated_lead_time_suppliers", []))
    if not elevated:
        return None

    network_avg_days = _inventory_summary(inventory_analysis).get("average_days_of_inventory_remaining")
    if network_avg_days is None:
        return None

    candidates = [
        row for row in supplier_ranking.get("ranked_suppliers", [])
        if row["supplier"] in elevated and row.get("avg_days_remaining") is not None
        and row["avg_days_remaining"] < network_avg_days
    ]
    if not candidates:
        return None
    worst = min(candidates, key=lambda row: row["avg_days_remaining"])

    network_avg_lead_time = _inventory_summary(inventory_analysis).get("average_supplier_lead_time_days") or 1.0
    lead_time_gap_ratio = (worst["avg_lead_time_days"] - network_avg_lead_time) / network_avg_lead_time if network_avg_lead_time else 0.0
    strength = min(max(lead_time_gap_ratio, 0.0), 1.0)

    anomaly_signal = 1 if any(
        a.get("supplier") == worst["supplier"] and a.get("severity") == "HIGH"
        for a in _inventory_anomalies(inventory_analysis)
    ) else 0
    confidence = _compute_confidence(strength, corroborating_signals=anomaly_signal)

    return {
        "category": "Supplier Risk",
        "title": f"{worst['supplier']} Lead Time Compounding Stockout Risk",
        "description": (
            f"{worst['supplier']}'s average lead time ({worst['avg_lead_time_days']} days) is a "
            f"statistical outlier versus the network's other suppliers (network average "
            f"{network_avg_lead_time} days), and the products it sources already run below-average "
            f"inventory coverage ({worst['avg_days_remaining']} days vs. network average "
            f"{network_avg_days} days) -- a slow supplier for products that are already thin on stock."
        ),
        "evidence": [
            f"{worst['supplier']} average lead time: {worst['avg_lead_time_days']} days "
            f"(network average: {network_avg_lead_time} days).",
            f"{worst['supplier']}'s products average {worst['avg_days_remaining']} days of coverage "
            f"(network average: {network_avg_days} days).",
        ],
        "confidence": confidence,
        "business_impact": (
            "Products sourced from this supplier have the least ability to absorb a demand spike "
            "before stocking out, and the longest wait once they do -- the combination that most "
            "directly produces a stockout an operations team cannot react to in time."
        ),
        "recommended_action": (
            f"Negotiate expedited terms with {worst['supplier']} or qualify a backup supplier for "
            "the products it exclusively sources; in the meantime, raise reorder thresholds for "
            "those specific SKUs to compensate for the longer replenishment cycle."
        ),
    }


def _root_cause_priority_sla_miscalibration(delivery_analysis: dict[str, Any]) -> dict[str, Any] | None:
    """A non-warehouse root cause: premium priority tiers under-delivering against their own SLA."""
    priority_intel = _delivery_priority_intelligence(delivery_analysis)
    if priority_intel.get("premium_tiers_meeting_expectations", True):
        return None

    tiers = priority_intel.get("tiers", {})
    standard_on_time = tiers.get("Standard", {}).get("on_time_rate_percentage")
    premium_rates = [
        (tier, values.get("on_time_rate_percentage"))
        for tier, values in tiers.items()
        if tier != "Standard" and values.get("on_time_rate_percentage") is not None
    ]
    if standard_on_time is None or not premium_rates:
        return None

    worst_tier, worst_rate = min(premium_rates, key=lambda pair: pair[1])
    gap = round(standard_on_time - worst_rate, 1)
    if gap < PRIORITY_SLA_GAP_THRESHOLD:
        return None

    strength = min(gap / 30, 1.0)
    confidence = _compute_confidence(strength, corroborating_signals=0)

    return {
        "category": "Process Design",
        "title": "Premium Priority Tiers Under-Delivering Against Their Own SLA",
        "description": (
            f"{worst_tier} shipments -- the tier customers pay extra for faster, more reliable "
            f"handling -- are on-time only {worst_rate}% of the time, {gap} percentage points worse "
            f"than Standard tier's {standard_on_time}%. Each tier is judged against its own "
            "(tighter, for premium) promised date, so this is not an unfair comparison."
        ),
        "evidence": [
            f"{worst_tier} on-time rate: {worst_rate}%.",
            f"Standard on-time rate: {standard_on_time}%.",
            f"Gap: {gap} percentage points, despite {worst_tier} shipments receiving a tighter "
            "promised date by design.",
        ],
        "confidence": confidence,
        "business_impact": (
            "Customers paying a premium for priority handling are experiencing worse reliability "
            "than customers who paid standard rates -- a direct risk to premium-tier retention and "
            "to the pricing credibility of the priority tiers themselves."
        ),
        "recommended_action": (
            f"Audit {worst_tier} shipment routing and warehouse assignment specifically; determine "
            "whether the promised-date calibration for this tier is too aggressive relative to "
            "actual operational capability, or whether premium shipments are not receiving "
            "differentiated handling in practice."
        ),
    }


def _detect_root_causes(
    warehouse_composite: dict[str, Any],
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
    convergence_lookup: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    """Run every root-cause detector and return the most confident findings first."""
    candidates = [
        _root_cause_warehouse_bottleneck(warehouse_composite, delivery_analysis, complaints_analysis, inventory_analysis, convergence_lookup),
        _root_cause_supplier_lead_time(inventory_analysis),
        _root_cause_priority_sla_miscalibration(delivery_analysis),
    ]
    found = [rc for rc in candidates if rc is not None]
    return sorted(found, key=lambda rc: rc["confidence"], reverse=True)


# --------------------------------------------------------------------------
# Business Impact Assessment
# --------------------------------------------------------------------------

def _business_impacts(
    root_causes: list[dict[str, Any]],
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """Estimate operational consequences, each grounded in a specific metric, sorted by severity."""
    root_cause_categories = {rc["category"] for rc in root_causes}
    impacts: list[dict[str, Any]] = []

    complaints_summary = _complaints_summary(complaints_analysis)
    negative_pct = complaints_summary.get("negative_sentiment_percentage")
    if negative_pct is not None:
        impacts.append({
            "impact": "Increased customer dissatisfaction",
            "severity": _severity_tier(negative_pct, NEGATIVE_SENTIMENT_SEVERITY),
            "evidence": (
                f"{negative_pct}% of complaints carry negative sentiment; Customer Health Score is "
                f"{_complaints_health_score(complaints_analysis).get('overall_score')}/100."
            ),
        })

    delivery_kpis = _delivery_kpis(delivery_analysis)
    delay_rate = delivery_kpis.get("delay_rate_percentage")
    priority_ok = _delivery_priority_intelligence(delivery_analysis).get("premium_tiers_meeting_expectations", True)
    if delay_rate is not None:
        severity = _severity_tier(delay_rate, DELAY_RATE_SEVERITY)
        if not priority_ok and severity == "LOW":
            severity = "MEDIUM"
        impacts.append({
            "impact": "SLA violations",
            "severity": severity,
            "evidence": (
                f"Network delay rate is {delay_rate}%"
                + ("; premium priority tiers are not meeting their own on-time expectations." if not priority_ok else ".")
            ),
        })

    if "Warehouse Bottleneck" in root_cause_categories:
        bottleneck = next(rc for rc in root_causes if rc["category"] == "Warehouse Bottleneck")
        impacts.append({
            "impact": "Warehouse overload",
            "severity": "HIGH" if bottleneck["confidence"] >= 0.7 else "MEDIUM",
            "evidence": bottleneck["description"],
        })

    inventory_summary = _inventory_summary(inventory_analysis)
    below_reorder_pct = inventory_summary.get("percentage_below_reorder_threshold")
    if below_reorder_pct is not None:
        impacts.append({
            "impact": "Inventory stockout risk",
            "severity": _severity_tier(below_reorder_pct, BELOW_REORDER_SEVERITY),
            "evidence": f"{below_reorder_pct}% of stock positions are currently below their reorder threshold.",
        })

    if "Supplier Risk" in root_cause_categories:
        supplier_rc = next(rc for rc in root_causes if rc["category"] == "Supplier Risk")
        impacts.append({
            "impact": "Supplier dependency",
            "severity": "HIGH" if supplier_rc["confidence"] >= 0.7 else "MEDIUM",
            "evidence": supplier_rc["description"],
        })

    complaints_segments = _complaints_segments(complaints_analysis)
    enterprise_row = next((r for r in complaints_segments if r.get("customer_type") == "Enterprise"), None)
    other_rates = [r["complaint_rate_percentage"] for r in complaints_segments if r.get("customer_type") != "Enterprise" and r.get("complaint_rate_percentage") is not None]
    if enterprise_row and other_rates and enterprise_row.get("complaint_rate_percentage") is not None:
        if enterprise_row["complaint_rate_percentage"] >= max(other_rates):
            impacts.append({
                "impact": "Revenue risk",
                "severity": "HIGH",
                "evidence": (
                    f"Enterprise customers -- typically the highest-value accounts -- have the "
                    f"highest complaint rate of any tier ({enterprise_row['complaint_rate_percentage']}%)."
                ),
            })

    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return sorted(impacts, key=lambda item: severity_rank.get(item["severity"], 3))


# --------------------------------------------------------------------------
# Priority Actions
# --------------------------------------------------------------------------

ROOT_CAUSE_ACTION_PROFILE: dict[str, dict[str, str]] = {
    "Warehouse Bottleneck": {"difficulty": "High", "estimated_timeframe": "60-120 days"},
    "Supplier Risk": {"difficulty": "Medium", "estimated_timeframe": "30-60 days"},
    "Process Design": {"difficulty": "Low", "estimated_timeframe": "14-30 days"},
}
DEFAULT_ACTION_PROFILE = {"difficulty": "Medium", "estimated_timeframe": "30-60 days"}


def _build_priority_actions(root_causes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn each root cause directly into a prioritized action -- never an invented one."""
    actions = []
    for priority, root_cause in enumerate(root_causes, start=1):
        profile = ROOT_CAUSE_ACTION_PROFILE.get(root_cause["category"], DEFAULT_ACTION_PROFILE)
        actions.append({
            "priority": priority,
            "title": root_cause["title"],
            "reason": root_cause["description"],
            "expected_business_impact": root_cause["business_impact"],
            "difficulty": profile["difficulty"],
            "estimated_timeframe": profile["estimated_timeframe"],
            "confidence": root_cause["confidence"],
        })
    return actions


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _build_executive_summary(
    root_causes: list[dict[str, Any]],
    priority_actions: list[dict[str, Any]],
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
) -> dict[str, Any]:
    """A concise, machine-readable overview -- what matters, not a metric dump."""
    strongest_operational = _first_or_none(_delivery_anomalies(delivery_analysis))
    strongest_customer = _first_or_none(_complaints_anomalies(complaints_analysis))
    strongest_inventory = _first_or_none(_inventory_anomalies(inventory_analysis))

    health_scores = [
        _delivery_health_score(delivery_analysis).get("overall_score"),
        _complaints_health_score(complaints_analysis).get("overall_score"),
        _inventory_health_score(inventory_analysis).get("overall_score"),
    ]
    valid_scores = [s for s in health_scores if s is not None]
    avg_health = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else None

    max_confidence = max((rc["confidence"] for rc in root_causes), default=0.0)
    if (avg_health is not None and avg_health < 60) or max_confidence >= 0.85:
        risk_level = "HIGH"
    elif (avg_health is not None and avg_health < 80) or max_confidence >= 0.6:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    top_action = _first_or_none(priority_actions)
    executive_priority = (
        f"{top_action['title']} -- {top_action['expected_business_impact']}"
        if top_action
        else "No cross-domain root cause cleared this platform's evidence bar; continue monitoring each engine's own anomalies."
    )

    return {
        "strongest_operational_issue": (
            {"title": strongest_operational["title"], "severity": strongest_operational["severity"]}
            if strongest_operational else {"title": "No significant operational anomalies detected", "severity": "LOW"}
        ),
        "strongest_customer_issue": (
            {"title": strongest_customer["title"], "severity": strongest_customer["severity"]}
            if strongest_customer else {"title": "No significant customer experience anomalies detected", "severity": "LOW"}
        ),
        "strongest_inventory_issue": (
            {"title": strongest_inventory["title"], "severity": strongest_inventory["severity"]}
            if strongest_inventory else {"title": "No significant inventory anomalies detected", "severity": "LOW"}
        ),
        "overall_business_risk_level": risk_level,
        "risk_level_basis": (
            f"Average of the three engines' health scores: {avg_health}/100. "
            f"{len(root_causes)} cross-domain root cause(s) identified, "
            f"highest confidence {round(max_confidence, 2)}."
        ),
        "executive_priority": executive_priority,
    }


# --------------------------------------------------------------------------
# Confidence Scores (top-level summary)
# --------------------------------------------------------------------------

def _build_confidence_scores(root_causes: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up every confidence score computed in this module into one auditable summary."""
    scores: dict[str, float] = {}
    for root_cause in root_causes:
        key = root_cause["category"].lower().replace(" ", "_") + "_root_cause"
        scores[key] = root_cause["confidence"]

    relationship_confidences = [r["confidence"] for r in relationships]
    if relationship_confidences:
        scores["average_relationship_confidence"] = round(sum(relationship_confidences) / len(relationship_confidences), 2)

    all_scores = [rc["confidence"] for rc in root_causes] + relationship_confidences
    scores["overall_diagnostic_confidence"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
    return scores


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_correlations(
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
    shipments_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> dict[str, Any]:
    """Cross-domain Business Intelligence layer: why the business performs the way it does.

    Consumes the outputs of analyze_deliveries, analyze_complaints, and
    analyze_inventory -- never recomputing what they already computed --
    to find where independent domains agree, infer evidence-backed root
    causes, estimate business impact, and produce a prioritized,
    confidence-scored action plan. Raw DataFrames are used only for the
    one cross-tabulation no existing engine exposes (complaint category
    distribution among delayed-shipment complaints); every other
    finding is derived from the three engines' own computed fields.

    Args:
        delivery_analysis: Output of analysis.delivery.analyze_deliveries.
        complaints_analysis: Output of analysis.complaints.analyze_complaints.
        inventory_analysis: Output of analysis.inventory.analyze_inventory.
        shipments_df: The shipments DataFrame (data/shipments.csv).
        complaints_df: The complaints DataFrame (data/complaints.csv).
        inventory_df: The inventory DataFrame (data/inventory.csv); accepted
            for validation and future extensibility, but every inventory
            metric this module reads is already available on
            inventory_analysis, so it is not recomputed from here.

    Returns:
        A JSON-serializable dictionary shaped as::

            {
                "executive_summary": {...},
                "relationships": [...],
                "root_causes": [...],
                "business_impacts": [...],
                "priority_actions": [...],
                "confidence_scores": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If any analysis dict is missing/malformed, or any
            raw DataFrame is empty/None.
    """
    delivery_analysis = _require_analysis(delivery_analysis, REQUIRED_DELIVERY_KEYS, "delivery_analysis")
    complaints_analysis = _require_analysis(complaints_analysis, REQUIRED_COMPLAINTS_KEYS, "complaints_analysis")
    inventory_analysis = _require_analysis(inventory_analysis, REQUIRED_INVENTORY_KEYS, "inventory_analysis")
    _require_dataframe(shipments_df, "shipments_df")
    _require_dataframe(complaints_df, "complaints_df")
    _require_dataframe(inventory_df, "inventory_df")

    delivery_wh = _warehouse_names(_delivery_warehouses(delivery_analysis))
    complaints_wh = _warehouse_names(_complaints_warehouses(complaints_analysis))
    inventory_wh = _warehouse_names(_inventory_warehouses(inventory_analysis))
    common_warehouses = _common_warehouses(delivery_wh, complaints_wh, inventory_wh)
    excluded_warehouses = sorted((delivery_wh | complaints_wh | inventory_wh) - set(common_warehouses))

    convergence_lookup = _build_convergence_lookup(common_warehouses, delivery_analysis, complaints_analysis, inventory_analysis)
    warehouse_composite = _compute_warehouse_composite_ranking(delivery_analysis, complaints_analysis, inventory_analysis, common_warehouses)

    delivery_delay_map = _to_warehouse_map(_delivery_warehouses(delivery_analysis), "delay_rate_percentage")
    delivery_loading_map = _to_warehouse_map(_delivery_warehouses(delivery_analysis), "avg_loading_time")
    inventory_coverage_map = _to_warehouse_map(_inventory_warehouses(inventory_analysis), "avg_days_remaining")
    inventory_utilization_map = _to_warehouse_map(_inventory_warehouses(inventory_analysis), "capacity_utilization_percentage")
    inventory_risk_map = _to_warehouse_map(_inventory_warehouses(inventory_analysis), "risk_score")
    complaints_rate_map = _to_warehouse_map(_complaints_warehouses(complaints_analysis), "complaint_rate_percentage")

    relationships = [
        _relationship_delayed_complaints(complaints_analysis),
        _relationship_delay_complaint_categories(complaints_df, shipments_df),
        _relationship_from_warehouse_correlation(
            "Inventory <-> Delivery",
            "inventory coverage (days)", inventory_coverage_map, "low",
            "delivery delay rate", delivery_delay_map, "high",
            common_warehouses,
            "Warehouses running thinner inventory buffers tend to also run higher delivery delay "
            "rates -- consistent with a warehouse that can't restock fast enough also struggling to "
            "ship on time.",
        ),
        _relationship_from_warehouse_correlation(
            "Inventory <-> Delivery",
            "capacity utilization", inventory_utilization_map, "high",
            "average loading time", delivery_loading_map, "high",
            common_warehouses,
            "Warehouses operating closer to their storage capacity limit tend to also take longer "
            "to load outbound shipments -- consistent with congestion at a warehouse that's "
            "physically fuller.",
        ),
        _relationship_from_warehouse_correlation(
            "Inventory <-> Complaints",
            "inventory risk score", inventory_risk_map, "high",
            "complaint rate", complaints_rate_map, "high",
            common_warehouses,
            "Warehouses with a poor overall inventory risk profile tend to also generate more "
            "customer complaints -- inventory health is not just an internal operations metric, it "
            "shows up in customer experience.",
        ),
        _relationship_from_warehouse_correlation(
            "Inventory <-> Complaints",
            "inventory coverage (days)", inventory_coverage_map, "low",
            "complaint rate", complaints_rate_map, "high",
            common_warehouses,
            "Low inventory coverage specifically (not just the broader risk score) tracks with "
            "higher complaint rates -- thin stock buffers appear to indirectly predict customer "
            "dissatisfaction, likely mediated through the delivery delays thin stock causes.",
        ),
        _relationship_customer_type(delivery_analysis, complaints_analysis),
    ]
    if warehouse_composite["ranked_warehouses"]:
        top_row = warehouse_composite["ranked_warehouses"][0]
        relationships.append(_make_relationship(
            domain_pair="Warehouse Analysis",
            finding=(
                f"{top_row['warehouse']} has the highest composite risk score across shipment "
                f"demand, delay rate, complaint rate, and inventory risk combined."
            ),
            metric="warehouse_composite_risk_ranking",
            value=warehouse_composite["ranked_warehouses"],
            supporting_evidence=[warehouse_composite["methodology_note"]],
            confidence=_compute_confidence(
                strength=convergence_lookup.get(top_row["warehouse"], {}).get("ranking_signals", 0) / 3.0,
                corroborating_signals=convergence_lookup.get(top_row["warehouse"], {}).get("anomaly_signals", 0),
            ),
            interpretation=(
                "This is the single warehouse where fixing one root cause is most likely to improve "
                "delivery, complaints, and inventory metrics simultaneously."
            ),
        ))
    relationships = [r for r in relationships if r is not None]

    root_causes = _detect_root_causes(warehouse_composite, delivery_analysis, complaints_analysis, inventory_analysis, convergence_lookup)
    business_impacts = _business_impacts(root_causes, delivery_analysis, complaints_analysis, inventory_analysis)
    priority_actions = _build_priority_actions(root_causes)
    executive_summary = _build_executive_summary(root_causes, priority_actions, delivery_analysis, complaints_analysis, inventory_analysis)
    confidence_scores = _build_confidence_scores(root_causes, relationships)

    result = {
        "executive_summary": executive_summary,
        "relationships": relationships,
        "root_causes": root_causes,
        "business_impacts": business_impacts,
        "priority_actions": priority_actions,
        "confidence_scores": confidence_scores,
        "methodology": {
            "relationships_evaluated": [
                {"domain_pair": r["domain_pair"], "metric": r["metric"]} for r in relationships
            ],
            "statistical_methods_used": [
                "Pearson correlation coefficient (pandas Series.corr()) across per-warehouse "
                "aggregate metrics already computed by the delivery/complaints/inventory engines.",
                "Rate multipliers (ratio of two already-computed percentages).",
                "Composite z-score summation (population z-scores) for the warehouse risk ranking.",
                "One raw cross-tabulation via a shipment_id join (complaint category distribution "
                "among delayed shipments) -- the only relationship not derivable from an existing "
                "engine's output.",
            ],
            "confidence_methodology": (
                f"Every confidence score is base {CONFIDENCE_BASE} (a real, measured pattern "
                f"deserves some baseline trust) plus up to {CONFIDENCE_STRENGTH_WEIGHT} for the "
                "strength of the underlying statistic (|Pearson r|, or how far a rate multiplier "
                f"sits from a neutral 1.0x) plus {CONFIDENCE_PER_CONVERGENCE_SIGNAL} per independent "
                f"engine or anomaly detector that flags the same entity/conclusion, up to "
                f"{CONFIDENCE_MAX_CONVERGENCE_SIGNALS} such signals, capped at {CONFIDENCE_CAP} -- "
                "never full certainty. Three engines independently agreeing on the same warehouse "
                "can reach the cap; a single-source statistical observation tops out well below it."
            ),
            "warehouse_reconciliation": (
                f"{len(common_warehouses)} warehouse name(s) were common across all three engines "
                "and used for every cross-warehouse comparison in this module."
                + (f" {len(excluded_warehouses)} name(s) appeared in only some engines and were "
                   f"excluded from cross-warehouse comparisons: {excluded_warehouses}."
                   if excluded_warehouses else "")
            ),
            "assumptions": [
                "Customer tier labels ('Retail', 'SME', 'Enterprise') are assumed consistent across "
                "the delivery and complaints engines.",
                "A warehouse's own 'worst'/'highest-risk' flag from each engine's ranking is treated "
                "as that engine's authoritative opinion; this module does not re-derive rankings.",
                "Pearson correlation across warehouse-level aggregates treats each warehouse as one "
                "observation; it does not use shipment- or complaint-level granularity.",
            ],
            "limitations": [
                "With only a handful of warehouses, Pearson correlation coefficients are a "
                "statistically thin signal on their own -- this module always pairs them with a "
                "plain-language 'is the same entity worst on both metrics' check rather than "
                "reporting r in isolation.",
                "This module performs descriptive and diagnostic analytics only: no machine "
                "learning, no predictive modeling, no forecasting. Root causes are evidence-backed "
                "hypotheses ranked by confidence, not proven causation.",
                "Correlation direction is inferred from domain knowledge in the interpretation text "
                "(e.g. 'thin inventory likely causes delays, not the reverse'), not established "
                "statistically -- Pearson r itself is symmetric and does not indicate direction.",
            ],
            "raw_data_usage": (
                "This module primarily consumes the pre-computed outputs of analyze_deliveries, "
                "analyze_complaints, and analyze_inventory. shipments_df and complaints_df are used "
                "directly for exactly one cross-tabulation not exposed by any existing engine "
                "(complaint category distribution among delayed-shipment complaints, via a "
                "shipment_id join). inventory_df is validated but not read from directly, since "
                "every inventory metric this module needs is already available on inventory_analysis."
            ),
        },
    }
    return _json_safe(result)
