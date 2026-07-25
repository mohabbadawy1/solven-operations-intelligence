"""Customer Experience Intelligence engine.

Turns customer_cases.csv into the numbers a Customer Experience
Director or COO needs to answer "where is the buyer journey actually
breaking down, and is it connected to an operational root cause
elsewhere in the business": case volume, first-response and
resolution SLA attainment, reopen and escalation rates, sentiment, and
the category/journey-stage breakdown that lets analysis/correlations.py
connect a spike in handover-related complaints to a specific
construction delay, or a spike in collections complaints to a specific
payment-plan pattern.

Output shape: {summary, kpis, rankings, trends, anomalies,
health_score, methodology}.
"""

from __future__ import annotations

from typing import Any

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

REQUIRED_CASE_COLUMNS = [
    "case_id", "customer_id", "project_id", "unit_id", "created_at", "category", "subcategory",
    "priority", "first_response_at", "resolved_at", "status", "resolution_sla_hours",
    "resolution_sla_met", "reopen_count", "sentiment", "escalation_flag", "responsible_department",
]

NEGATIVE_SENTIMENT_LABEL = "Negative"
MAX_ANOMALIES_RETURNED = 6
ANOMALY_Z_HIGH = 1.6

# --- Customer Experience Health Score ------------------------------------
#
#   sla_attainment          % of resolved cases that met their
#                         resolution SLA
#   sentiment_quality        100 minus the network's negative-sentiment
#                         rate
#   escalation_control        100 minus a penalty for the escalation
#                         rate
#   reopen_control             100 minus a penalty for the reopen rate
#                         among resolved cases -- a proxy for whether
#                         "resolved" actually meant resolved
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "sla_attainment": 0.30,
    "sentiment_quality": 0.30,
    "escalation_control": 0.20,
    "reopen_control": 0.20,
}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("analyze_customer_experience: customer_cases DataFrame is empty or None")
    validate_columns(df, REQUIRED_CASE_COLUMNS, "analyze_customer_experience: customer_cases")

    working = df.copy()
    working["created_at"] = pd.to_datetime(working["created_at"], errors="coerce")
    working["first_response_at"] = pd.to_datetime(working["first_response_at"], errors="coerce")
    working["resolved_at"] = pd.to_datetime(working["resolved_at"], errors="coerce")
    working["month"] = working["created_at"].dt.to_period("M").astype(str)
    working["escalation_flag"] = working["escalation_flag"].astype(bool)
    working["is_resolved"] = working["status"] == "Resolved"
    working["response_minutes"] = (working["first_response_at"] - working["created_at"]).dt.total_seconds() / 60
    return working


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    resolved = df[df["is_resolved"]]
    negative = df[df["sentiment"] == NEGATIVE_SENTIMENT_LABEL]
    escalated = df[df["escalation_flag"]]
    reopened = resolved[resolved["reopen_count"] > 0]
    sla_met = resolved[resolved["resolution_sla_met"] == True]  # noqa: E712 -- explicit bool comparison over pandas nullable bool

    return {
        "total_cases": int(len(df)),
        "resolved_cases": int(len(resolved)),
        "resolution_rate_pct": _percentage(len(resolved), len(df)),
        "average_first_response_minutes": _safe_round(df["response_minutes"].mean()),
        "resolution_sla_attainment_pct": _percentage(len(sla_met), len(resolved)) if len(resolved) else None,
        "negative_sentiment_pct": _percentage(len(negative), len(df)),
        "escalation_rate_pct": _percentage(len(escalated), len(df)),
        "reopen_rate_pct": _percentage(len(reopened), len(resolved)) if len(resolved) else 0.0,
    }


# --------------------------------------------------------------------------
# Category / journey-stage breakdown
# --------------------------------------------------------------------------

def _compute_category_breakdown(df: pd.DataFrame) -> list[dict[str, Any]]:
    total = len(df)
    grouped = df.groupby("category").agg(
        cases=("case_id", "count"),
        negative=("sentiment", lambda s: (s == NEGATIVE_SENTIMENT_LABEL).sum()),
        escalated=("escalation_flag", "sum"),
        avg_resolution_sla_met=("resolution_sla_met", "mean"),
    ).reset_index()
    grouped["share_of_total_pct"] = grouped.apply(lambda r: _percentage(r["cases"], total), axis=1)
    grouped["negative_sentiment_pct"] = grouped.apply(lambda r: _percentage(r["negative"], r["cases"]), axis=1)
    grouped["escalation_rate_pct"] = grouped.apply(lambda r: _percentage(r["escalated"], r["cases"]), axis=1)
    grouped["sla_attainment_pct"] = grouped["avg_resolution_sla_met"].apply(lambda v: _safe_round(v * 100) if pd.notna(v) else None)
    grouped = grouped.sort_values("cases", ascending=False).reset_index(drop=True)
    return to_records(grouped, ["category", "cases", "share_of_total_pct", "negative_sentiment_pct", "escalation_rate_pct", "sla_attainment_pct"])


def _compute_department_breakdown(df: pd.DataFrame) -> list[dict[str, Any]]:
    resolved = df[df["is_resolved"]]
    grouped = df.groupby("responsible_department").agg(cases=("case_id", "count")).reset_index()
    sla_by_dept = resolved.groupby("responsible_department")["resolution_sla_met"].mean()
    grouped["sla_attainment_pct"] = grouped["responsible_department"].map(sla_by_dept).apply(
        lambda v: _safe_round(v * 100) if pd.notna(v) else None
    )
    grouped = grouped.sort_values("cases", ascending=False).reset_index(drop=True)
    return to_records(grouped, ["responsible_department", "cases", "sla_attainment_pct"])


# --------------------------------------------------------------------------
# Rankings: project
# --------------------------------------------------------------------------

def _compute_project_ranking(df: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = df.groupby("project_id").agg(
        cases=("case_id", "count"),
        negative=("sentiment", lambda s: (s == NEGATIVE_SENTIMENT_LABEL).sum()),
        escalated=("escalation_flag", "sum"),
    ).reset_index()
    grouped["negative_sentiment_pct"] = grouped.apply(lambda r: _percentage(r["negative"], r["cases"]), axis=1)
    grouped["escalation_rate_pct"] = grouped.apply(lambda r: _percentage(r["escalated"], r["cases"]), axis=1)
    grouped = grouped.sort_values("negative_sentiment_pct", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", grouped.index + 1)
    return to_records(grouped, ["rank", "project_id", "cases", "negative_sentiment_pct", "escalation_rate_pct"])


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

def _compute_monthly_trends(df: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = df.groupby("month").agg(
        cases=("case_id", "count"),
        negative=("sentiment", lambda s: (s == NEGATIVE_SENTIMENT_LABEL).sum()),
        escalated=("escalation_flag", "sum"),
    ).reset_index().sort_values("month")
    grouped["negative_sentiment_pct"] = grouped.apply(lambda r: _percentage(r["negative"], r["cases"]), axis=1)
    return to_records(grouped, ["month", "cases", "negative_sentiment_pct", "escalated"])


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

def _detect_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag project/category combinations where negative-sentiment share is a statistical outlier."""
    grouped = df.groupby(["project_id", "category"]).agg(
        cases=("case_id", "count"), negative=("sentiment", lambda s: (s == NEGATIVE_SENTIMENT_LABEL).sum()),
    ).reset_index()
    grouped = grouped[grouped["cases"] >= 15]
    grouped["negative_pct"] = grouped.apply(lambda r: _percentage(r["negative"], r["cases"]), axis=1)
    if len(grouped) < 4:
        return []

    z = _zscores(grouped["negative_pct"].astype(float))
    anomalies = []
    for idx, z_value in z.items():
        if z_value < ANOMALY_Z_HIGH:
            continue
        row = grouped.loc[idx]
        anomalies.append({
            "severity": "HIGH" if z_value >= 2.3 else "MEDIUM",
            "category": "Customer Experience",
            "title": f"{row['project_id']} {row['category']} cases run sharply negative",
            "description": (
                f"{row['project_id']}'s '{row['category']}' cases carry {row['negative_pct']:.1f}% "
                f"negative sentiment across {int(row['cases'])} cases, {z_value:.1f} standard "
                "deviations above the portfolio's other project/category combinations."
            ),
            "recommended_action": (
                f"Cross-reference {row['project_id']}'s operational data for the '{row['category']}' "
                "theme (construction milestones for handover-related cases, payment schedules for "
                "collections-related cases) to confirm the operational root cause driving sentiment."
            ),
            "project_id": row["project_id"],
            "case_category": row["category"],
        })
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------

def _compute_health_score(summary: dict[str, Any]) -> dict[str, Any]:
    sla_score = summary["resolution_sla_attainment_pct"] or 0.0
    sentiment_score = max(0.0, 100 - (summary["negative_sentiment_pct"] or 0.0))
    escalation_score = max(0.0, 100 - (summary["escalation_rate_pct"] or 0.0) * 2.0)
    reopen_score = max(0.0, 100 - (summary["reopen_rate_pct"] or 0.0) * 3.0)

    components = {
        "sla_attainment": round(sla_score, 1),
        "sentiment_quality": round(sentiment_score, 1),
        "escalation_control": round(escalation_score, 1),
        "reopen_control": round(reopen_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            "Weighted blend of four 0-100 components: sla_attainment (% of resolved cases meeting "
            "their resolution SLA, weight 0.30), sentiment_quality (100 minus the network's "
            "negative-sentiment percentage, weight 0.30), escalation_control (100 minus 2 points "
            "per percentage point of escalation rate, weight 0.20), and reopen_control (100 minus 3 "
            "points per percentage point of reopen rate among resolved cases, weight 0.20)."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_customer_experience(customer_cases_df: pd.DataFrame) -> dict[str, Any]:
    """Run the full Customer Experience Intelligence suite a CX Director/COO needs.

    Args:
        customer_cases_df: The customer cases DataFrame loaded from
            data/customer_cases.csv.

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},
                "kpis": {"by_category": [...], "by_department": [...]},
                "rankings": {"by_project": [...]},
                "trends": {"monthly": [...]},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If `customer_cases_df` is empty/None or missing
            columns this module depends on.
    """
    df = _prepare(customer_cases_df)

    summary = _compute_summary(df)
    health_score = _compute_health_score(summary)

    result = {
        "summary": summary,
        "kpis": {
            "by_category": _compute_category_breakdown(df),
            "by_department": _compute_department_breakdown(df),
        },
        "rankings": {"by_project": _compute_project_ranking(df)},
        "trends": {"monthly": _compute_monthly_trends(df)},
        "anomalies": _detect_anomalies(df),
        "health_score": health_score,
        "methodology": {
            "case_categories": (
                "Categories (Sales Experience, Collections, Construction Update, Handover, "
                "Snagging / Quality, General Inquiry) are the same journey-stage vocabulary used "
                "across the platform's sales, collections, and construction modules, so this "
                "module's findings can be cross-referenced by category with the operational engine "
                "that owns that stage."
            ),
            "anomaly_detection": (
                "Sentiment anomalies are z-scored across project/category combinations with at "
                "least 15 cases (cross-sectional) -- never by referencing a specific project or "
                "category by name in code."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
