"""Construction Delivery & Handover Readiness Intelligence engine.

Turns construction_milestones.csv (+ handovers.csv and snagging.csv)
into the numbers a Construction Director or Development Director needs
to answer "are we going to deliver on the promised date, and what does
a delay actually expose us to": milestone schedule/cost variance,
building- and contractor-level delay concentration, and -- the
financially material question -- how much contracted value, how many
customers, and how much already-collected cash sits behind units
forecast for a delayed handover.

Handover readiness (snagging volume, resolution time, reopen rate,
keys-released/inspection/notification compliance) is analyzed in the
same module as construction delivery rather than a separate file:
handover risk is the direct downstream consequence of a construction
delay, and both draw on the same building-level grain.

Output shape: {summary, kpis, rankings, anomalies, health_score,
methodology}.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis._config import CONFIG
from analysis._shared import (
    _json_safe,
    _percentage,
    _safe_round,
    _zscores,
    sort_and_limit_anomalies,
    to_records,
    validate_columns,
)

REQUIRED_MILESTONE_COLUMNS = [
    "milestone_id", "project_id", "phase", "building", "milestone_name", "baseline_end_date",
    "forecast_end_date", "actual_end_date", "planned_completion_pct", "actual_completion_pct",
    "variance_days", "status", "contractor", "budgeted_cost", "actual_cost", "cost_variance",
    "issue_category", "issue_severity",
]
REQUIRED_HANDOVER_COLUMNS = [
    "handover_id", "unit_id", "project_id", "original_handover_date", "forecast_handover_date",
    "actual_handover_date", "handover_status", "days_delayed", "final_payment_received",
    "customer_notified", "inspection_completed", "keys_released",
]
REQUIRED_SNAG_COLUMNS = ["snag_id", "handover_id", "unit_id", "category", "severity", "status", "reopen_count"]

DELAYED_HANDOVER_STATUSES = {"Delayed", "Completed (Delayed)"}
MAX_ANOMALIES_RETURNED = 6
ANOMALY_VARIANCE_WATCH_DAYS_DEFAULT = 30
ANOMALY_VARIANCE_CRITICAL_DAYS_DEFAULT = 60

# --- Construction & Handover Health Score ---------------------------------
#
#   schedule_adherence     100 minus a penalty for average milestone
#                         schedule variance across in-progress/complete
#                         milestones
#   cost_control            100 minus a penalty for aggregate cost
#                         variance as a share of budgeted cost
#   handover_delay_exposure  100 minus a penalty for the share of
#                         scheduled handovers currently delayed
#   snagging_quality          100 minus a penalty for snag volume per
#                         handover and reopen rate -- a proxy for
#                         finish quality reaching the customer
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "schedule_adherence": 0.30,
    "cost_control": 0.20,
    "handover_delay_exposure": 0.30,
    "snagging_quality": 0.20,
}


def _prepare_milestones(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("analyze_construction_handover: construction_milestones DataFrame is empty or None")
    validate_columns(df, REQUIRED_MILESTONE_COLUMNS, "analyze_construction_handover: construction_milestones")
    working = df.copy()
    for col in ("baseline_end_date", "forecast_end_date", "actual_end_date"):
        working[col] = pd.to_datetime(working[col], errors="coerce")
    return working


def _prepare_handovers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("analyze_construction_handover: handovers DataFrame is empty or None")
    validate_columns(df, REQUIRED_HANDOVER_COLUMNS, "analyze_construction_handover: handovers")
    working = df.copy()
    for col in ("original_handover_date", "forecast_handover_date", "actual_handover_date"):
        working[col] = pd.to_datetime(working[col], errors="coerce")
    for col in ("final_payment_received", "customer_notified", "inspection_completed", "keys_released"):
        working[col] = working[col].astype(bool)
    working["is_delayed"] = working["handover_status"].isin(DELAYED_HANDOVER_STATUSES)
    return working


def _prepare_snagging(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=REQUIRED_SNAG_COLUMNS)
    validate_columns(df, REQUIRED_SNAG_COLUMNS, "analyze_construction_handover: snagging")
    return df.copy()


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(milestones: pd.DataFrame, handovers: pd.DataFrame, snagging: pd.DataFrame) -> dict[str, Any]:
    in_scope = milestones[milestones["actual_completion_pct"] > 0]
    overdue_milestones = milestones[milestones["status"].isin(["Delayed", "At Risk"])]

    delayed_handovers = handovers[handovers["is_delayed"]]
    open_snags = snagging[snagging["status"] != "Resolved"]

    return {
        "total_milestones": int(len(milestones)),
        "milestones_in_progress_or_complete": int(len(in_scope)),
        "milestones_delayed_or_at_risk": int(len(overdue_milestones)),
        "average_schedule_variance_days": _safe_round(in_scope["variance_days"].mean()),
        "total_budgeted_cost": _safe_round(milestones["budgeted_cost"].sum()),
        # Cost variance is scoped to milestones that have actually
        # started (actual_completion_pct > 0) -- an unstarted milestone
        # has committed budget but $0 actual spend by definition, and
        # including it would make the portfolio look artificially
        # under-budget simply because construction hasn't reached it yet.
        "total_cost_variance": _safe_round(in_scope["cost_variance"].sum()),
        "cost_variance_pct_of_budget": _safe_round(
            100 * in_scope["cost_variance"].sum() / in_scope["budgeted_cost"].sum()
        ) if len(in_scope) and in_scope["budgeted_cost"].sum() else None,
        "total_handovers_scheduled": int(len(handovers)),
        "handovers_delayed": int(len(delayed_handovers)),
        "handover_delay_rate_pct": _percentage(len(delayed_handovers), len(handovers)),
        "average_handover_delay_days": _safe_round(delayed_handovers["days_delayed"].mean()) if len(delayed_handovers) else 0.0,
        "total_snags": int(len(snagging)),
        "open_snags": int(len(open_snags)),
        "snags_per_handover": _safe_round(len(snagging) / len(handovers)) if len(handovers) else None,
    }


# --------------------------------------------------------------------------
# Financial exposure: contracted value, customers, cash collected behind delayed units
# --------------------------------------------------------------------------

def _compute_delay_exposure(handovers: pd.DataFrame, sales_df: pd.DataFrame | None, payment_schedules_df: pd.DataFrame | None) -> dict[str, Any]:
    delayed = handovers[handovers["is_delayed"]]
    result: dict[str, Any] = {
        "units_exposed_to_delay": int(len(delayed)),
        "contracted_value_exposed": None,
        "customers_exposed": None,
        "amount_already_collected_on_delayed_units": None,
        "note": None,
    }
    if sales_df is None or "unit_id" not in sales_df.columns:
        result["note"] = "sales_df not supplied; contracted-value and customer exposure not computed."
        return result

    relevant = sales_df[sales_df["unit_id"].isin(delayed["unit_id"])]
    result["contracted_value_exposed"] = _safe_round(relevant["net_sales_value"].sum())
    result["customers_exposed"] = int(relevant["customer_id"].nunique())

    if payment_schedules_df is not None and "sale_id" in payment_schedules_df.columns:
        collected = payment_schedules_df[payment_schedules_df["sale_id"].isin(relevant["sale_id"])]["amount_paid"].sum()
        result["amount_already_collected_on_delayed_units"] = _safe_round(collected)
    else:
        result["note"] = "payment_schedules_df not supplied; collected-cash-behind-delay figure not computed."
    return result


# --------------------------------------------------------------------------
# Building / contractor / phase delay concentration
# --------------------------------------------------------------------------

def _compute_delay_concentration(milestones: pd.DataFrame) -> dict[str, Any]:
    scoped = milestones[milestones["actual_completion_pct"] > 0]

    def _grouped_variance(col: str, label: str) -> list[dict[str, Any]]:
        grouped = scoped.groupby([col, "project_id"]).agg(
            milestones=("milestone_id", "count"),
            avg_variance_days=("variance_days", "mean"),
            max_variance_days=("variance_days", "max"),
            high_severity_issues=("issue_severity", lambda s: (s == "HIGH").sum()),
        ).reset_index().rename(columns={col: label})
        grouped = grouped.sort_values("avg_variance_days", ascending=False).reset_index(drop=True)
        return to_records(grouped, ["project_id", label, "milestones", "avg_variance_days", "max_variance_days", "high_severity_issues"])

    by_building = _grouped_variance("building", "building")
    by_contractor = scoped.groupby("contractor").agg(
        milestones=("milestone_id", "count"), avg_variance_days=("variance_days", "mean"),
        high_severity_issues=("issue_severity", lambda s: (s == "HIGH").sum()),
    ).reset_index().sort_values("avg_variance_days", ascending=False)
    by_phase = _grouped_variance("phase", "phase")

    issue_counts = scoped["issue_category"].value_counts()
    issue_distribution = [
        {"issue_category": cat, "count": int(count)} for cat, count in issue_counts.items() if pd.notna(cat)
    ]

    return {
        "by_building": by_building,
        "by_contractor": to_records(by_contractor.reset_index(drop=True), ["contractor", "milestones", "avg_variance_days", "high_severity_issues"]),
        "by_phase": by_phase,
        "issue_category_distribution": issue_distribution,
        "most_delayed_building": by_building[0]["building"] if by_building else None,
        "most_delayed_building_project": by_building[0]["project_id"] if by_building else None,
    }


# --------------------------------------------------------------------------
# Handover readiness
# --------------------------------------------------------------------------

def _compute_handover_readiness(handovers: pd.DataFrame, snagging: pd.DataFrame) -> dict[str, Any]:
    by_project = []
    for project_id, group in handovers.groupby("project_id"):
        delayed = group[group["is_delayed"]]
        readiness_score = round(
            100
            - _percentage(len(delayed), len(group)) * 0.6
            - _percentage(int((~group["final_payment_received"]).sum()), len(group)) * 0.2
            - _percentage(int((~group["inspection_completed"]).sum()), len(group)) * 0.2,
            1,
        )
        by_project.append({
            "project_id": project_id,
            "scheduled_handovers": int(len(group)),
            "delayed_handovers": int(len(delayed)),
            "delay_rate_pct": _percentage(len(delayed), len(group)),
            "final_payment_received_pct": _percentage(int(group["final_payment_received"].sum()), len(group)),
            "inspection_completed_pct": _percentage(int(group["inspection_completed"].sum()), len(group)),
            "customer_notified_pct": _percentage(int(group["customer_notified"].sum()), len(group)),
            "handover_readiness_score": max(0.0, min(100.0, readiness_score)),
        })
    by_project.sort(key=lambda r: r["handover_readiness_score"])

    snag_by_severity = snagging["severity"].value_counts()
    resolved = snagging[snagging["status"] == "Resolved"]
    reopened = snagging[snagging["reopen_count"] > 0]

    return {
        "by_project": by_project,
        "least_ready_project": by_project[0]["project_id"] if by_project else None,
        "snag_severity_distribution": [
            {"severity": s, "count": int(c)} for s, c in snag_by_severity.items()
        ],
        "snag_resolution_rate_pct": _percentage(len(resolved), len(snagging)) if len(snagging) else None,
        "snag_reopen_rate_pct": _percentage(len(reopened), len(resolved)) if len(resolved) else None,
    }


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

def _detect_anomalies(delay_concentration: dict[str, Any]) -> list[dict[str, Any]]:
    watch = CONFIG["thresholds"].get("construction_variance_watch_days", ANOMALY_VARIANCE_WATCH_DAYS_DEFAULT)
    critical = CONFIG["thresholds"].get("construction_variance_critical_days", ANOMALY_VARIANCE_CRITICAL_DAYS_DEFAULT)

    anomalies = []
    for row in delay_concentration["by_building"]:
        if row["avg_variance_days"] is None or row["avg_variance_days"] < watch:
            continue
        severity = "HIGH" if row["avg_variance_days"] >= critical else "MEDIUM"
        anomalies.append({
            "severity": severity,
            "category": "Construction Delivery",
            "title": f"{row['project_id']} {row['building']} is running behind schedule",
            "description": (
                f"{row['project_id']} {row['building']} averages {row['avg_variance_days']:.0f} days "
                f"of schedule variance across {int(row['milestones'])} milestones (peak "
                f"{int(row['max_variance_days'])} days), with {int(row['high_severity_issues'])} "
                "high-severity issue(s) logged -- above the "
                f"{watch}-day construction variance watch threshold."
            ),
            "recommended_action": (
                f"Escalate {row['project_id']} {row['building']}'s critical-path milestones "
                "(particularly MEP and finishing) with the responsible contractor, and notify "
                "affected buyers proactively rather than at the original promised date."
            ),
            "project_id": row["project_id"],
            "building": row["building"],
        })
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------

def _compute_health_score(summary: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    variance = summary["average_schedule_variance_days"] or 0.0
    schedule_score = max(0.0, 100 - max(variance, 0) * 1.2)

    cost_variance_pct = summary["cost_variance_pct_of_budget"] or 0.0
    cost_score = max(0.0, 100 - abs(cost_variance_pct) * 4)

    delay_rate = summary["handover_delay_rate_pct"] or 0.0
    handover_score = max(0.0, 100 - delay_rate * 1.5)

    reopen_rate = readiness.get("snag_reopen_rate_pct") or 0.0
    snags_per_handover = summary["snags_per_handover"] or 0.0
    snagging_score = max(0.0, 100 - snags_per_handover * 8 - reopen_rate * 0.5)

    components = {
        "schedule_adherence": round(schedule_score, 1),
        "cost_control": round(cost_score, 1),
        "handover_delay_exposure": round(handover_score, 1),
        "snagging_quality": round(snagging_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            "Weighted blend of four 0-100 components: schedule_adherence (100 minus 1.2 points per "
            "average day of milestone schedule variance, weight 0.30), cost_control (100 minus 4 "
            "points per percentage point of aggregate cost variance vs. budget, weight 0.20), "
            "handover_delay_exposure (100 minus 1.5 points per percentage point of scheduled "
            "handovers currently delayed, weight 0.30), and snagging_quality (100 minus a penalty "
            "for snags-per-handover and the snag reopen rate, weight 0.20)."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_construction_handover(
    construction_milestones_df: pd.DataFrame,
    handovers_df: pd.DataFrame,
    snagging_df: pd.DataFrame | None = None,
    sales_df: pd.DataFrame | None = None,
    payment_schedules_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the full Construction Delivery & Handover Readiness Intelligence suite.

    Args:
        construction_milestones_df: From data/construction_milestones.csv.
        handovers_df: From data/handovers.csv.
        snagging_df: From data/snagging.csv. Optional; snagging KPIs
            report as empty if omitted.
        sales_df: From data/sales.csv, used only to translate delayed
            units into contracted-value/customer exposure. Optional.
        payment_schedules_df: From data/payment_schedules.csv, used
            only to compute cash already collected on delayed units.
            Optional.

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},
                "kpis": {"delay_exposure": {...}, "delay_concentration": {...}, "handover_readiness": {...}},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If `construction_milestones_df` or `handovers_df`
            is empty/None or missing columns this module depends on.
    """
    milestones = _prepare_milestones(construction_milestones_df)
    handovers = _prepare_handovers(handovers_df)
    snagging = _prepare_snagging(snagging_df)

    summary = _compute_summary(milestones, handovers, snagging)
    delay_concentration = _compute_delay_concentration(milestones)
    readiness = _compute_handover_readiness(handovers, snagging)
    health_score = _compute_health_score(summary, readiness)

    result = {
        "summary": summary,
        "kpis": {
            "delay_exposure": _compute_delay_exposure(handovers, sales_df, payment_schedules_df),
            "delay_concentration": delay_concentration,
            "handover_readiness": readiness,
        },
        "anomalies": _detect_anomalies(delay_concentration),
        "health_score": health_score,
        "methodology": {
            "delay_exposure": (
                "contracted_value_exposed and customers_exposed are computed by joining delayed "
                "handover unit_ids back to sales.csv; amount_already_collected_on_delayed_units "
                "joins forward to payment_schedules.csv's amount_paid for those same sale_ids -- "
                "the figure that distinguishes 'the unit will be late' from the financially and "
                "reputationally sharper 'the customer has already paid and the unit will be late.'"
            ),
            "handover_readiness_score": (
                "100 minus 0.6 points per percentage point of scheduled handovers delayed, minus "
                "0.2 points per percentage point missing final payment, minus 0.2 points per "
                "percentage point missing a completed inspection -- a compact per-project "
                "operational readiness indicator, distinct from the platform-wide "
                "handover_delay_exposure health-score component."
            ),
            "anomaly_detection": (
                "Building-level schedule anomalies are flagged against the fixed variance "
                "thresholds in config/real_estate_demo.yml (thresholds.construction_variance_watch_days "
                "/ _critical_days) -- never by referencing a specific building or project by name in code."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
