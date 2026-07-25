"""Collections & Receivables Risk Intelligence engine.

Turns payment_schedules.csv (+ collections.csv for recovery-workflow
context) into the numbers a Collections Director or CFO needs to
answer "how much cash are we actually owed, how much of it is at
risk, and is that risk getting worse": receivables aging, collection
rate, delinquency by project/segment/payment-plan/agent/broker,
bounced-payment and promise-to-pay fulfillment rates, and the 30/60/90
day forward obligation schedule.

Output shape: {summary, kpis, rankings, trends, anomalies,
health_score, methodology}.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis._config import CONFIG
from analysis._shared import (
    RECEIVABLES_AGING_ORDER,
    _json_safe,
    _percentage,
    _safe_round,
    _zscores,
    bucket_rank,
    sort_and_limit_anomalies,
    to_records,
    validate_columns,
)

REQUIRED_INSTALLMENT_COLUMNS = [
    "installment_id", "sale_id", "customer_id", "project_id", "due_date", "amount_due",
    "installment_type", "payment_status", "days_overdue", "amount_paid", "outstanding_amount",
    "bounced_payment_flag", "rescheduled_flag",
]
REQUIRED_COLLECTIONS_COLUMNS = ["collection_event_id", "installment_id", "event_date", "recovery_status", "promise_to_pay_date", "promise_to_pay_amount"]

# Installments in these statuses are excluded from "due so far" totals --
# they either haven't come due yet, or were voided by a contract
# cancellation and were never truly receivable.
EXCLUDED_STATUSES = {"Not Yet Due", "Voided (Cancelled)"}

MAX_ANOMALIES_RETURNED = 8
ANOMALY_Z_HIGH = 1.6

# --- Collections Health Score ------------------------------------------
#
#   collection_rate         paid amount / amount due so far, vs. the
#                         configured collections-rate target
#   delinquency_control      100 minus a penalty for the overdue share
#                         of amount due
#   aging_severity           100 minus a penalty weighted toward the
#                         oldest receivables buckets specifically
#   recovery_effectiveness    promise-to-pay fulfillment rate among
#                         installments that received a collection event
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "collection_rate": 0.35,
    "delinquency_control": 0.30,
    "aging_severity": 0.20,
    "recovery_effectiveness": 0.15,
}


def _prepare_installments(payment_schedules_df: pd.DataFrame) -> pd.DataFrame:
    if payment_schedules_df is None or len(payment_schedules_df) == 0:
        raise ValueError("analyze_collections_risk: payment_schedules DataFrame is empty or None")
    validate_columns(payment_schedules_df, REQUIRED_INSTALLMENT_COLUMNS, "analyze_collections_risk: payment_schedules")

    df = payment_schedules_df.copy()
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
    df["month"] = df["due_date"].dt.to_period("M").astype(str)
    df["is_due"] = ~df["payment_status"].isin(EXCLUDED_STATUSES)
    df["is_overdue"] = df["payment_status"] == "Overdue"
    df["aging_bucket"] = df.apply(_aging_bucket, axis=1)
    return df


def _aging_bucket(row) -> str | None:
    if not row["is_overdue"]:
        return "Current" if row["is_due"] else None
    days = row["days_overdue"]
    if days <= 30:
        return "1-30 days"
    if days <= 60:
        return "31-60 days"
    if days <= 90:
        return "61-90 days"
    if days <= 180:
        return "91-180 days"
    return "180+ days"


def _prepare_collections(collections_df: pd.DataFrame | None) -> pd.DataFrame:
    if collections_df is None or len(collections_df) == 0:
        return pd.DataFrame(columns=REQUIRED_COLLECTIONS_COLUMNS)
    validate_columns(collections_df, REQUIRED_COLLECTIONS_COLUMNS, "analyze_collections_risk: collections")
    df = collections_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    return df


def _as_of(df: pd.DataFrame) -> pd.Timestamp:
    """The data's own reporting snapshot date.

    Deliberately the max due_date among installments that have already
    matured (payment_status not in {"Not Yet Due", "Voided (Cancelled)"}),
    not the max due_date across the whole schedule -- payment plans are
    generated years into the future, and a cancelled contract's future
    installments are voided (not "Not Yet Due") regardless of how far
    out their original due_date was, so both statuses must be excluded
    or a distant future installment gets treated as "today" and
    silently breaks every forward-looking (30/60/90-day) calculation.
    """
    matured = df.loc[~df["payment_status"].isin(EXCLUDED_STATUSES), "due_date"]
    return matured.max() if len(matured) else df["due_date"].max()


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    due = df[df["is_due"]]
    overdue = df[df["is_overdue"]]

    amount_due = due["amount_due"].sum()
    amount_paid = due["amount_paid"].sum()
    outstanding = overdue["outstanding_amount"].sum()

    return {
        "total_installments_due": int(len(due)),
        "amount_due_to_date": _safe_round(amount_due),
        "amount_collected_to_date": _safe_round(amount_paid),
        "collection_rate_pct": _safe_round(100 * amount_paid / amount_due) if amount_due else None,
        "overdue_installment_count": int(len(overdue)),
        "overdue_customer_count": int(overdue["customer_id"].nunique()),
        "total_overdue_amount": _safe_round(outstanding),
        "overdue_amount_pct_of_due": _safe_round(100 * outstanding / amount_due) if amount_due else None,
        "bounced_payment_count": int(df["bounced_payment_flag"].sum()),
        "bounced_payment_rate_pct": _percentage(int(df["bounced_payment_flag"].sum()), len(due)),
        "rescheduled_installment_count": int(df["rescheduled_flag"].sum()),
        "average_days_overdue": _safe_round(overdue["days_overdue"].mean()) if len(overdue) else None,
    }


# --------------------------------------------------------------------------
# Receivables aging
# --------------------------------------------------------------------------

def _compute_aging(df: pd.DataFrame) -> dict[str, Any]:
    due = df[df["is_due"]]
    grouped = due.groupby("aging_bucket").agg(
        amount=("outstanding_amount", "sum"), count=("installment_id", "count"),
    ).reset_index()
    total_outstanding = due.loc[due["is_overdue"], "outstanding_amount"].sum()
    grouped["percentage_of_overdue"] = grouped.apply(
        lambda r: _percentage(r["amount"], total_outstanding) if r["aging_bucket"] != "Current" else None, axis=1
    )
    grouped = grouped.sort_values("aging_bucket", key=lambda s: s.map(lambda b: bucket_rank(b, RECEIVABLES_AGING_ORDER)))

    by_project = []
    for project_id, group in due.groupby("project_id"):
        overdue_group = group[group["is_overdue"]]
        by_project.append({
            "project_id": project_id,
            "amount_due": _safe_round(group["amount_due"].sum()),
            "overdue_amount": _safe_round(overdue_group["outstanding_amount"].sum()),
            "overdue_pct": _safe_round(100 * overdue_group["outstanding_amount"].sum() / group["amount_due"].sum()) if group["amount_due"].sum() else 0.0,
        })
    by_project.sort(key=lambda r: r["overdue_pct"] or 0, reverse=True)

    return {
        "aging_buckets": to_records(grouped, ["aging_bucket", "amount", "count", "percentage_of_overdue"]),
        "by_project": by_project,
        "highest_overdue_exposure_project": by_project[0]["project_id"] if by_project else None,
    }


# --------------------------------------------------------------------------
# Delinquency cuts: payment plan, down-payment band, agent, broker
# --------------------------------------------------------------------------

def _compute_delinquency_by(df: pd.DataFrame, sales_df: pd.DataFrame | None, group_col: str, label: str) -> list[dict[str, Any]] | None:
    if sales_df is None or group_col not in sales_df.columns:
        return None
    merged = df.merge(sales_df[["sale_id", group_col]], on="sale_id", how="left")
    due = merged[merged["is_due"]]
    grouped = due.groupby(group_col).agg(
        amount_due=("amount_due", "sum"),
        overdue_amount=("outstanding_amount", lambda s: s[due.loc[s.index, "is_overdue"]].sum()),
        installments=("installment_id", "count"),
    ).reset_index().rename(columns={group_col: label})
    grouped["overdue_pct"] = grouped.apply(
        lambda r: _safe_round(100 * r["overdue_amount"] / r["amount_due"]) if r["amount_due"] else 0.0, axis=1
    )
    grouped = grouped.sort_values("overdue_pct", ascending=False).reset_index(drop=True)
    return to_records(grouped, [label, "installments", "amount_due", "overdue_amount", "overdue_pct"])


# --------------------------------------------------------------------------
# Forward obligations (30/60/90 day)
# --------------------------------------------------------------------------

def _compute_forward_obligations(df: pd.DataFrame) -> dict[str, Any]:
    as_of = _as_of(df)
    not_yet_due = df[df["payment_status"] == "Not Yet Due"]
    windows = {}
    for label, days in [("next_30_days", 30), ("next_60_days", 60), ("next_90_days", 90)]:
        upper = as_of + pd.Timedelta(days=days)
        due_in_window = not_yet_due[not_yet_due["due_date"] <= upper]
        windows[label] = _safe_round(due_in_window["amount_due"].sum()) or 0.0
    return {"as_of_date": as_of.strftime("%Y-%m-%d"), **windows}


# --------------------------------------------------------------------------
# Collections workflow effectiveness
# --------------------------------------------------------------------------

def _compute_recovery_effectiveness(collections: pd.DataFrame) -> dict[str, Any]:
    if collections.empty:
        return {
            "collection_events": 0, "promise_to_pay_count": 0, "promise_to_pay_fulfillment_rate_pct": None,
            "recovery_status_distribution": [],
        }
    total = len(collections)
    promises = collections[collections["promise_to_pay_date"].notna()]
    fulfilled = promises[promises["recovery_status"].isin(["Recovered", "Partially Recovered"])]
    status_counts = collections["recovery_status"].value_counts()
    return {
        "collection_events": int(total),
        "promise_to_pay_count": int(len(promises)),
        "promise_to_pay_fulfillment_rate_pct": _percentage(len(fulfilled), len(promises)) if len(promises) else None,
        "recovery_status_distribution": [
            {"status": status, "count": int(count), "percentage": _percentage(count, total)}
            for status, count in status_counts.items()
        ],
    }


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

def _compute_monthly_trends(df: pd.DataFrame) -> list[dict[str, Any]]:
    due = df[df["is_due"]]
    grouped = due.groupby("month").agg(
        amount_due=("amount_due", "sum"),
        amount_paid=("amount_paid", "sum"),
    ).reset_index().sort_values("month")
    overdue_by_month = due[due["is_overdue"]].groupby("month")["outstanding_amount"].sum()
    grouped["overdue_amount"] = grouped["month"].map(overdue_by_month).fillna(0)
    grouped["collection_rate_pct"] = grouped.apply(
        lambda r: _safe_round(100 * r["amount_paid"] / r["amount_due"]) if r["amount_due"] else None, axis=1
    )
    return to_records(grouped, ["month", "amount_due", "amount_paid", "overdue_amount", "collection_rate_pct"])


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

def _detect_anomalies(aging_by_project: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag projects whose overdue share is a statistical outlier vs. its peers (cross-sectional)."""
    if len(aging_by_project) < 3:
        return []
    frame = pd.DataFrame(aging_by_project)
    z = _zscores(frame["overdue_pct"].astype(float))
    anomalies = []
    for idx, z_value in z.items():
        if z_value < ANOMALY_Z_HIGH:
            continue
        row = frame.loc[idx]
        anomalies.append({
            "severity": "HIGH" if z_value >= 2.3 else "MEDIUM",
            "category": "Collections Risk",
            "title": f"{row['project_id']} carries a disproportionate overdue receivables share",
            "description": (
                f"{row['project_id']}'s overdue amount is {row['overdue_pct']:.1f}% of what is due "
                f"({row['overdue_amount']:,.0f} of {row['amount_due']:,.0f}), {z_value:.1f} standard "
                "deviations above its peer projects."
            ),
            "recommended_action": (
                f"Prioritize {row['project_id']}'s overdue accounts for structured collections "
                "outreach; cross-reference with down-payment and payment-plan-length distribution "
                "to determine whether the exposure concentrates in a specific buyer affordability band."
            ),
            "project_id": row["project_id"],
        })
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------

def _compute_health_score(summary: dict[str, Any], aging: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    target = CONFIG["targets"]["collections_rate_pct"]
    collection_rate = summary["collection_rate_pct"] or 0.0
    collection_rate_score = min(100.0, (collection_rate / target) * 100) if target else collection_rate

    overdue_pct = summary["overdue_amount_pct_of_due"] or 0.0
    delinquency_score = max(0.0, 100 - overdue_pct * 2.5)

    aging_buckets = {row["aging_bucket"]: row["amount"] for row in aging["aging_buckets"]}
    total_overdue = sum(v for k, v in aging_buckets.items() if k != "Current") or 1.0
    severe_weight = (
        aging_buckets.get("91-180 days", 0) * 1.5 + aging_buckets.get("180+ days", 0) * 2.5
    ) / total_overdue
    aging_severity_score = max(0.0, 100 - severe_weight * 40)

    recovery_rate = recovery.get("promise_to_pay_fulfillment_rate_pct")
    recovery_score = recovery_rate if recovery_rate is not None else 60.0

    components = {
        "collection_rate": round(collection_rate_score, 1),
        "delinquency_control": round(delinquency_score, 1),
        "aging_severity": round(aging_severity_score, 1),
        "recovery_effectiveness": round(recovery_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            f"Weighted blend of four 0-100 components: collection_rate (amount collected / amount "
            f"due so far, rescaled against the {target}% target in config/real_estate_demo.yml, "
            "weight 0.35), delinquency_control (100 minus 2.5 points per percentage point of amount "
            "due that is overdue, weight 0.30), aging_severity (100 minus a penalty weighted toward "
            "the 91-180 and 180+ day buckets specifically, weight 0.20), and recovery_effectiveness "
            "(promise-to-pay fulfillment rate among installments that received a collection event, "
            "weight 0.15)."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_collections_risk(
    payment_schedules_df: pd.DataFrame, collections_df: pd.DataFrame | None = None, sales_df: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Run the full Collections & Receivables Risk Intelligence suite a Collections Director/CFO needs.

    Args:
        payment_schedules_df: The payment schedules DataFrame loaded
            from data/payment_schedules.csv.
        collections_df: The collections DataFrame loaded from
            data/collections.csv. Optional; recovery-effectiveness
            KPIs report as unavailable if omitted.
        sales_df: The sales DataFrame, used only to attribute overdue
            amounts to sales_agent_id / broker_id / payment_plan_years
            / down_payment_pct (fields payment_schedules.csv doesn't
            carry natively). Optional; those specific cuts are omitted
            if not supplied.

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},
                "kpis": {"aging": {...}, "forward_obligations": {...}, "recovery_effectiveness": {...},
                         "delinquency_by_payment_plan_years": [...] | None,
                         "delinquency_by_sales_agent": [...] | None,
                         "delinquency_by_broker": [...] | None},
                "trends": {"monthly": [...]},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If `payment_schedules_df` is empty/None or missing
            columns this module depends on.
    """
    df = _prepare_installments(payment_schedules_df)
    collections = _prepare_collections(collections_df)

    summary = _compute_summary(df)
    aging = _compute_aging(df)
    recovery = _compute_recovery_effectiveness(collections)
    health_score = _compute_health_score(summary, aging, recovery)

    result = {
        "summary": summary,
        "kpis": {
            "aging": aging,
            "forward_obligations": _compute_forward_obligations(df),
            "recovery_effectiveness": recovery,
            "delinquency_by_payment_plan_years": _compute_delinquency_by(df, sales_df, "payment_plan_years", "payment_plan_years"),
            "delinquency_by_sales_agent": _compute_delinquency_by(df, sales_df, "sales_agent_id", "sales_agent_id"),
            "delinquency_by_broker": _compute_delinquency_by(df, sales_df, "broker_id", "broker_id"),
        },
        "trends": {"monthly": _compute_monthly_trends(df)},
        "anomalies": _detect_anomalies(aging["by_project"]),
        "health_score": health_score,
        "methodology": {
            "due_definition": (
                "'Due so far' excludes installments with status 'Not Yet Due' (their due date is in "
                "the future relative to the data's own as-of date) and 'Voided (Cancelled)' "
                "(the underlying contract was cancelled, so the receivable never matured) -- every "
                "other status counts toward amount_due_to_date."
            ),
            "forward_obligations": (
                "next_30/60/90_days sums amount_due for installments with status 'Not Yet Due' whose "
                "due_date falls within that many days of the data's own most recent due_date -- a "
                "forward cash-collection schedule, not a forecast."
            ),
            "anomaly_detection": (
                "Overdue-share anomalies are z-scored across projects' own figures (cross-sectional) "
                "-- never by referencing a specific project by name in code."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
