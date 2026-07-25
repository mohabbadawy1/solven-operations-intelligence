"""Lead Funnel & Sales Capacity Intelligence engine.

Turns leads.csv into the numbers a Sales Director or Marketing
Director needs to answer "where are we losing demand, and is it a
marketing-quality problem or a sales-capacity problem": response SLA
attainment, stage-to-stage conversion, lead aging, loss reasons, and
per-agent workload concentration -- the last of which is what lets
this module (in concert with analysis/correlations.py) distinguish a
lead-quality issue from an overloaded sales team unable to process
good leads fast enough.

Every conversion rate here is a *stage-reached* rate (did this lead
ever get a first_contact_at / appointment_date / site_visit_date /
sale_id), not a read of the single current_stage snapshot column --
current_stage reflects where a lead sits today, which for a mature
CRM export is dominated by leads that have since resolved to lost or
gone dormant; stage-reached timestamps are the durable record of what
actually happened during the lead's active life and are therefore the
basis for every conversion-rate calculation in this module.

Output shape: {summary, kpis, rankings, trends, anomalies,
health_score, methodology}.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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

REQUIRED_LEAD_COLUMNS = [
    "lead_id", "created_at", "project_interest", "channel", "lead_score", "sales_agent_id",
    "broker_id", "first_response_at", "first_contact_at", "current_stage", "loss_reason",
    "qualification_status", "appointment_date", "site_visit_date", "sale_id",
]

DORMANT_STAGES = {"dormant", "lost"}
RESOLVED_ACTIVE_WINDOW_DAYS = 90  # a lead older than this with no resolution is treated as stale, not "in-flight"
MAX_ANOMALIES_RETURNED = 8
ANOMALY_Z_MEDIUM = 1.5
ANOMALY_Z_HIGH = 2.2

# --- Lead Funnel Health Score ------------------------------------------
#
#   response_sla           % of leads first-responded-to within the
#                         configured SLA window
#   qualification_yield    % of contacted leads that qualify
#   conversion_depth        lead-to-reservation conversion rate,
#                         rescaled so the portfolio's own typical rate
#                         reads as a reasonable baseline
#   workload_balance        100 minus a penalty for how unevenly
#                         active (non-resolved) leads are distributed
#                         across agents -- captures capacity risk a
#                         pure conversion-rate metric would miss
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "response_sla": 0.30,
    "qualification_yield": 0.20,
    "conversion_depth": 0.30,
    "workload_balance": 0.20,
}
# A lead-to-reservation conversion rate at or above this is treated as
# a fully healthy 100 on the conversion_depth component; real-estate
# funnels typically convert in the low single digits, so 3.5% is
# intentionally a strong, not average, benchmark.
CONVERSION_DEPTH_FULL_HEALTH_PCT = 3.5


def _prepare(leads_df: pd.DataFrame) -> pd.DataFrame:
    if leads_df is None or len(leads_df) == 0:
        raise ValueError("analyze_lead_funnel: leads DataFrame is empty or None")
    validate_columns(leads_df, REQUIRED_LEAD_COLUMNS, "analyze_lead_funnel: leads")

    df = leads_df.copy()
    for col in ("created_at", "first_response_at", "first_contact_at", "appointment_date", "site_visit_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["month"] = df["created_at"].dt.to_period("M").astype(str)

    df["response_minutes"] = (df["first_response_at"] - df["created_at"]).dt.total_seconds() / 60
    df["was_contacted"] = df["first_contact_at"].notna()
    df["was_qualified"] = df["qualification_status"] == "Qualified"
    df["had_appointment"] = df["appointment_date"].notna()
    df["had_site_visit"] = df["site_visit_date"].notna()
    df["converted_to_sale"] = df["sale_id"].notna()
    return df


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    sla_minutes = CONFIG["thresholds"]["response_sla_minutes"]
    responded = df["response_minutes"].notna()
    within_sla = responded & (df["response_minutes"] <= sla_minutes)

    total = len(df)
    contacted = int(df["was_contacted"].sum())
    qualified = int(df["was_qualified"].sum())
    appointments = int(df["had_appointment"].sum())
    site_visits = int(df["had_site_visit"].sum())
    conversions = int(df["converted_to_sale"].sum())
    dormant = int(df["current_stage"].isin(DORMANT_STAGES).sum())

    return {
        "total_leads": total,
        "contacted_leads": contacted,
        "contact_rate_pct": _percentage(contacted, total),
        "response_sla_minutes_threshold": sla_minutes,
        "response_sla_attainment_pct": _percentage(int(within_sla.sum()), int(responded.sum())) if responded.sum() else None,
        "average_first_response_minutes": _safe_round(df.loc[responded, "response_minutes"].mean()),
        "qualified_leads": qualified,
        "qualification_rate_pct": _percentage(qualified, contacted) if contacted else 0.0,
        "appointments_booked": appointments,
        "appointment_rate_of_qualified_pct": _percentage(appointments, qualified) if qualified else 0.0,
        "site_visits_completed": site_visits,
        "site_visit_rate_of_appointments_pct": _percentage(site_visits, appointments) if appointments else 0.0,
        "leads_converted_to_sale": conversions,
        "lead_to_reservation_conversion_pct": _percentage(conversions, total),
        "site_visit_to_reservation_conversion_pct": _percentage(conversions, site_visits) if site_visits else 0.0,
        "dormant_or_lost_leads": dormant,
        "dormant_or_lost_rate_pct": _percentage(dormant, total),
    }


# --------------------------------------------------------------------------
# Funnel leakage waterfall
# --------------------------------------------------------------------------

def _compute_funnel_waterfall(df: pd.DataFrame) -> list[dict[str, Any]]:
    stages = [
        ("New Leads", len(df)),
        ("Contacted", int(df["was_contacted"].sum())),
        ("Qualified", int(df["was_qualified"].sum())),
        ("Appointment Booked", int(df["had_appointment"].sum())),
        ("Site Visit Completed", int(df["had_site_visit"].sum())),
        ("Converted to Sale", int(df["converted_to_sale"].sum())),
    ]
    waterfall = []
    prior_count = None
    for label, count in stages:
        step_conversion_pct = _percentage(count, prior_count) if prior_count else 100.0
        waterfall.append({
            "stage": label, "count": count,
            "pct_of_total": _percentage(count, len(df)),
            "step_conversion_pct": step_conversion_pct,
        })
        prior_count = count
    return waterfall


def _compute_loss_reasons(df: pd.DataFrame) -> list[dict[str, Any]]:
    lost = df[df["current_stage"] == "lost"]
    total = len(lost)
    counts = lost["loss_reason"].value_counts()
    return [
        {"loss_reason": reason, "count": int(count), "percentage": _percentage(count, total)}
        for reason, count in counts.items()
    ]


# --------------------------------------------------------------------------
# Lead aging (in-flight leads only)
# --------------------------------------------------------------------------

def _compute_lead_aging(df: pd.DataFrame) -> dict[str, Any]:
    """Age distribution of leads still in an active (non-resolved) stage."""
    active = df[~df["current_stage"].isin(DORMANT_STAGES | {"contract", "reservation"})]
    as_of = df["created_at"].max()
    age_days = (as_of - active["created_at"]).dt.days

    def _bucket(days):
        if days <= 7:
            return "0-7 days"
        if days <= 21:
            return "8-21 days"
        if days <= 45:
            return "22-45 days"
        return "45+ days"

    buckets = age_days.apply(_bucket).value_counts()
    order = ["0-7 days", "8-21 days", "22-45 days", "45+ days"]
    distribution = [
        {"age_bucket": b, "count": int(buckets.get(b, 0)), "percentage": _percentage(buckets.get(b, 0), len(active))}
        for b in order
    ]
    return {
        "in_flight_leads": int(len(active)),
        "average_age_days": _safe_round(age_days.mean()) if len(active) else None,
        "age_distribution": distribution,
        "stale_45plus_days_count": int(buckets.get("45+ days", 0)),
    }


# --------------------------------------------------------------------------
# Rankings: project / channel / agent / campaign-adjacent
# --------------------------------------------------------------------------

def _conversion_ranking(df: pd.DataFrame, group_col: str, label: str) -> list[dict[str, Any]]:
    grouped = df.groupby(group_col).agg(
        leads=("lead_id", "count"),
        contacted=("was_contacted", "sum"),
        qualified=("was_qualified", "sum"),
        site_visits=("had_site_visit", "sum"),
        conversions=("converted_to_sale", "sum"),
        avg_response_minutes=("response_minutes", "mean"),
    ).reset_index().rename(columns={group_col: label})
    grouped["lead_to_contract_conversion_pct"] = grouped.apply(
        lambda r: _percentage(r["conversions"], r["leads"]), axis=1
    )
    grouped["qualification_rate_pct"] = grouped.apply(
        lambda r: _percentage(r["qualified"], r["contacted"]) if r["contacted"] else 0.0, axis=1
    )
    grouped = grouped.sort_values("lead_to_contract_conversion_pct", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", grouped.index + 1)
    columns = ["rank", label, "leads", "qualification_rate_pct", "lead_to_contract_conversion_pct", "avg_response_minutes"]
    return to_records(grouped, columns)


def _compute_agent_workload(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Active (open, non-resolved) lead count per agent -- the capacity signal."""
    active = df[~df["current_stage"].isin(DORMANT_STAGES | {"contract", "reservation"})]
    grouped = active.groupby("sales_agent_id").agg(
        active_leads=("lead_id", "count"),
        avg_response_minutes=("response_minutes", "mean"),
        premium_active_leads=("lead_score", lambda s: (s >= 80).sum()),
    ).reset_index()
    grouped = grouped.sort_values("active_leads", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", grouped.index + 1)
    return to_records(grouped, ["rank", "sales_agent_id", "active_leads", "premium_active_leads", "avg_response_minutes"])


def _compute_rankings(df: pd.DataFrame) -> dict[str, Any]:
    by_project = _conversion_ranking(df, "project_interest", "project_id")
    return {
        "by_project": by_project,
        "by_channel": _conversion_ranking(df, "channel", "channel"),
        "agent_workload": _compute_agent_workload(df),
        "strongest_converting_project": by_project[0]["project_id"] if by_project else None,
        "weakest_converting_project": by_project[-1]["project_id"] if by_project else None,
    }


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

def _compute_monthly_trends(df: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = df.groupby("month").agg(
        leads=("lead_id", "count"),
        contacted=("was_contacted", "sum"),
        qualified=("was_qualified", "sum"),
        conversions=("converted_to_sale", "sum"),
        avg_response_minutes=("response_minutes", "mean"),
    ).reset_index().sort_values("month")
    grouped["lead_to_contract_conversion_pct"] = grouped.apply(
        lambda r: _percentage(r["conversions"], r["leads"]), axis=1
    )
    return to_records(grouped, ["month", "leads", "contacted", "qualified", "conversions",
                                 "avg_response_minutes", "lead_to_contract_conversion_pct"])


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

def _detect_response_time_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag project-months where average first-response time is a statistical outlier vs. that project's own baseline."""
    grouped = df.groupby(["project_interest", "month"]).agg(
        leads=("lead_id", "count"), avg_response_minutes=("response_minutes", "mean"),
    ).reset_index()
    grouped = grouped[grouped["leads"] >= 10].dropna(subset=["avg_response_minutes"])

    anomalies = []
    for project_id, group in grouped.groupby("project_interest"):
        if len(group) < 3:
            continue
        group = group.sort_values("month").reset_index(drop=True)
        z = _zscores(group["avg_response_minutes"])
        for idx, z_value in z.items():
            if z_value < ANOMALY_Z_MEDIUM:
                continue
            row = group.loc[idx]
            severity = "HIGH" if z_value >= ANOMALY_Z_HIGH else "MEDIUM"
            anomalies.append({
                "severity": severity,
                "category": "Sales Capacity",
                "title": f"{project_id} first-response time spike in {row['month']}",
                "description": (
                    f"{project_id}'s average first-response time reached "
                    f"{row['avg_response_minutes']:.0f} minutes in {row['month']}, "
                    f"{z_value:.1f} standard deviations above its own baseline, across "
                    f"{int(row['leads'])} leads."
                ),
                "recommended_action": (
                    f"Review {project_id}'s active lead volume per agent for {row['month']} to "
                    "determine whether the team was over capacity relative to lead inflow."
                ),
                "project_id": project_id,
                "month": row["month"],
            })
    return anomalies


def _detect_workload_anomalies(rankings_workload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag agents whose active lead count is a statistical outlier vs. their peers."""
    if len(rankings_workload) < 4:
        return []
    frame = pd.DataFrame(rankings_workload)
    z = _zscores(frame["active_leads"].astype(float))
    anomalies = []
    for idx, z_value in z.items():
        if z_value < ANOMALY_Z_HIGH:
            continue
        row = frame.loc[idx]
        anomalies.append({
            "severity": "HIGH",
            "category": "Sales Capacity",
            "title": f"{row['sales_agent_id']} carries a disproportionate active-lead load",
            "description": (
                f"{row['sales_agent_id']} is currently working {int(row['active_leads'])} active "
                f"leads ({int(row['premium_active_leads'])} of them premium-scored), "
                f"{z_value:.1f} standard deviations above the peer average."
            ),
            "recommended_action": (
                f"Rebalance a portion of {row['sales_agent_id']}'s active pipeline to agents with "
                "spare capacity, prioritizing premium-scored leads first."
            ),
            "sales_agent_id": row["sales_agent_id"],
        })
    return anomalies


def _detect_anomalies(df: pd.DataFrame, rankings: dict[str, Any]) -> list[dict[str, Any]]:
    anomalies = _detect_response_time_anomalies(df) + _detect_workload_anomalies(rankings["agent_workload"])
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------

def _compute_health_score(summary: dict[str, Any], rankings: dict[str, Any]) -> dict[str, Any]:
    response_score = summary["response_sla_attainment_pct"] or 0.0
    qualification_score = summary["qualification_rate_pct"] or 0.0

    conversion_pct = summary["lead_to_reservation_conversion_pct"] or 0.0
    conversion_score = min(100.0, (conversion_pct / CONVERSION_DEPTH_FULL_HEALTH_PCT) * 100)

    workload = [row["active_leads"] for row in rankings["agent_workload"]]
    workload_balance_score = 100.0
    if len(workload) > 1 and sum(workload):
        std = float(np.std(workload))
        mean = float(np.mean(workload))
        cv = std / mean if mean else 0.0
        workload_balance_score = max(0.0, 100 - cv * 80)

    components = {
        "response_sla": round(response_score, 1),
        "qualification_yield": round(qualification_score, 1),
        "conversion_depth": round(conversion_score, 1),
        "workload_balance": round(workload_balance_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            f"Weighted blend of four 0-100 components: response_sla (% of leads first-responded-to "
            f"within {CONFIG['thresholds']['response_sla_minutes']} minutes, weight 0.30), "
            "qualification_yield (% of contacted leads that qualify, weight 0.20), conversion_depth "
            f"(lead-to-reservation conversion rate rescaled against a {CONVERSION_DEPTH_FULL_HEALTH_PCT}% "
            "full-health benchmark, weight 0.30), and workload_balance (100 minus a penalty for the "
            "coefficient of variation of active-lead count across agents, weight 0.20)."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_lead_funnel(leads_df: pd.DataFrame) -> dict[str, Any]:
    """Run the full Lead Funnel & Sales Capacity Intelligence suite a Sales/Marketing Director needs.

    Args:
        leads_df: The leads DataFrame loaded from data/leads.csv.

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},
                "kpis": {"funnel_waterfall": [...], "loss_reasons": [...], "lead_aging": {...}},
                "rankings": {...},
                "trends": {"monthly": [...]},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If `leads_df` is empty/None or missing columns
            this module depends on.
    """
    df = _prepare(leads_df)

    summary = _compute_summary(df)
    rankings = _compute_rankings(df)
    health_score = _compute_health_score(summary, rankings)

    result = {
        "summary": summary,
        "kpis": {
            "funnel_waterfall": _compute_funnel_waterfall(df),
            "loss_reasons": _compute_loss_reasons(df),
            "lead_aging": _compute_lead_aging(df),
        },
        "rankings": rankings,
        "trends": {"monthly": _compute_monthly_trends(df)},
        "anomalies": _detect_anomalies(df, rankings),
        "health_score": health_score,
        "methodology": {
            "conversion_definition": (
                "Every conversion rate in this module is a stage-reached rate, based on whether a "
                "durable timestamp/field was ever populated (first_contact_at, appointment_date, "
                "site_visit_date, sale_id) -- never a read of the single current_stage snapshot "
                "column, which for a mature CRM export is dominated by leads that have since "
                "resolved to 'lost' or gone 'dormant' regardless of how far they actually progressed."
            ),
            "workload_signal": (
                "agent_workload counts leads in a non-resolved stage (excludes dormant, lost, "
                "reservation, and contract) as of the most recent lead's created_at date -- this is "
                "the capacity signal analysis/correlations.py uses to distinguish a lead-quality "
                "problem from a sales-capacity problem."
            ),
            "anomaly_detection": (
                "Response-time anomalies are z-scored against each project's own monthly baseline; "
                "workload anomalies are z-scored across agents' current active-lead counts (a "
                "cross-sectional, not time-based, comparison). Neither references a specific "
                "project or agent by name in code."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
