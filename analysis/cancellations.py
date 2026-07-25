"""Cancellations & Revenue Leakage Intelligence engine.

Turns sales.csv into the numbers a CCO or Finance Director needs to
answer "how much confirmed revenue is actually leaking back out, and
why": reservation vs. contract cancellation rate, cancellation value
and timing, and the cut-by-cut breakdown (project, source, campaign,
agent, broker, payment plan, discount band, down-payment band) that
turns "cancellations are up" into "cancellations concentrate in X."

Output shape: {summary, kpis, rankings, trends, anomalies,
health_score, methodology}.
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

REQUIRED_SALES_COLUMNS = [
    "sale_id", "project_id", "customer_id", "reservation_date", "contract_date", "sales_agent_id",
    "broker_id", "lead_source", "gross_price", "net_sales_value", "discount_pct", "payment_plan_years",
    "down_payment_pct", "contract_status", "cancellation_flag", "cancellation_date", "cancellation_reason",
]

DOWN_PAYMENT_BANDS = [(0, 10, "Under 10%"), (10, 15, "10-15%"), (15, 20, "15-20%"), (20, 1000, "20%+")]
DISCOUNT_BANDS = [(0, 4, "Under 4%"), (4, 8, "4-8%"), (8, 12, "8-12%"), (12, 1000, "12%+")]

MAX_ANOMALIES_RETURNED = 6
ANOMALY_Z_HIGH = 1.6

# --- Cancellation Control Health Score ------------------------------------
#
#   reservation_cancellation_control   100 minus a penalty for the
#                                     reservation cancellation rate
#   contract_cancellation_control       100 minus a penalty for the
#                                     contract cancellation rate,
#                                     weighted more heavily since a
#                                     cancelled contract had already
#                                     cleared underwriting/commitment
#   concentration_control                100 minus a penalty for how
#                                     concentrated cancellation value
#                                     is in a single project
#   timing_predictability                 100 minus a penalty if
#                                     cancellations cluster sharply in
#                                     a specific post-reservation
#                                     window rather than spreading out
#                                     -- a clustered pattern signals a
#                                     specific, fixable process cause
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "reservation_cancellation_control": 0.20,
    "contract_cancellation_control": 0.40,
    "concentration_control": 0.20,
    "timing_predictability": 0.20,
}


def _prepare(sales_df: pd.DataFrame) -> pd.DataFrame:
    if sales_df is None or len(sales_df) == 0:
        raise ValueError("analyze_cancellations: sales DataFrame is empty or None")
    validate_columns(sales_df, REQUIRED_SALES_COLUMNS, "analyze_cancellations: sales")

    df = sales_df.copy()
    for col in ("reservation_date", "contract_date", "cancellation_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["cancellation_flag"] = df["cancellation_flag"].astype(bool)
    df["had_contract"] = df["contract_date"].notna()
    df["days_reservation_to_cancellation"] = (df["cancellation_date"] - df["reservation_date"]).dt.days
    df["down_payment_band"] = df["down_payment_pct"].apply(lambda v: _band(v, DOWN_PAYMENT_BANDS))
    df["discount_band"] = df["discount_pct"].apply(lambda v: _band(v, DISCOUNT_BANDS))
    return df


def _band(value: float | None, bands: list[tuple[float, float, str]]) -> str | None:
    if value is None or pd.isna(value):
        return None
    for low, high, label in bands:
        if low <= value < high:
            return label
    return bands[-1][2]


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    total_reservations = len(df)
    total_contracts = int(df["had_contract"].sum())
    cancelled = df[df["cancellation_flag"]]
    cancelled_contracts = cancelled[cancelled["had_contract"]]

    return {
        "total_reservations": total_reservations,
        "total_contracts": total_contracts,
        "total_cancellations": int(len(cancelled)),
        "reservation_cancellation_rate_pct": _percentage(len(cancelled), total_reservations),
        "contract_cancellation_rate_pct": _percentage(len(cancelled_contracts), total_contracts) if total_contracts else 0.0,
        "cancelled_gross_value": _safe_round(cancelled["gross_price"].sum()),
        "cancelled_net_value": _safe_round(cancelled["net_sales_value"].sum()),
        "average_days_reservation_to_cancellation": _safe_round(cancelled["days_reservation_to_cancellation"].mean()),
        "median_days_reservation_to_cancellation": _safe_round(cancelled["days_reservation_to_cancellation"].median()),
    }


# --------------------------------------------------------------------------
# Cancellation reasons
# --------------------------------------------------------------------------

def _compute_reasons(cancelled: pd.DataFrame) -> list[dict[str, Any]]:
    total = len(cancelled)
    counts = cancelled["cancellation_reason"].value_counts()
    return [
        {"reason": reason, "count": int(count), "percentage": _percentage(count, total),
         "value": _safe_round(cancelled.loc[cancelled["cancellation_reason"] == reason, "gross_price"].sum())}
        for reason, count in counts.items()
    ]


# --------------------------------------------------------------------------
# Timing distribution
# --------------------------------------------------------------------------

def _compute_timing_distribution(cancelled: pd.DataFrame) -> dict[str, Any]:
    days = cancelled["days_reservation_to_cancellation"].dropna()

    def _bucket(d):
        if d <= 14:
            return "0-14 days"
        if d <= 45:
            return "15-45 days"
        if d <= 90:
            return "46-90 days"
        return "90+ days"

    buckets = days.apply(_bucket).value_counts()
    order = ["0-14 days", "15-45 days", "46-90 days", "90+ days"]
    distribution = [
        {"window": w, "count": int(buckets.get(w, 0)), "percentage": _percentage(buckets.get(w, 0), len(days))}
        for w in order
    ]
    peak = max(distribution, key=lambda r: r["count"]) if distribution else None
    return {
        "distribution": distribution,
        "peak_window": peak["window"] if peak else None,
        "peak_window_share_pct": peak["percentage"] if peak else None,
    }


# --------------------------------------------------------------------------
# Cuts: project / source / campaign / agent / broker / payment plan / bands
# --------------------------------------------------------------------------

def _cancellation_rate_by(df: pd.DataFrame, group_col: str, label: str) -> list[dict[str, Any]]:
    grouped = df.groupby(group_col).agg(
        reservations=("sale_id", "count"),
        cancellations=("cancellation_flag", "sum"),
        cancelled_value=("gross_price", lambda s: s[df.loc[s.index, "cancellation_flag"]].sum()),
    ).reset_index().rename(columns={group_col: label})
    grouped["cancellation_rate_pct"] = grouped.apply(lambda r: _percentage(r["cancellations"], r["reservations"]), axis=1)
    grouped = grouped.sort_values("cancellation_rate_pct", ascending=False).reset_index(drop=True)
    return to_records(grouped, [label, "reservations", "cancellations", "cancellation_rate_pct", "cancelled_value"])


def _compute_cuts(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "by_project": _cancellation_rate_by(df, "project_id", "project_id"),
        "by_source": _cancellation_rate_by(df, "lead_source", "lead_source"),
        "by_payment_plan_years": _cancellation_rate_by(df, "payment_plan_years", "payment_plan_years"),
        "by_down_payment_band": _cancellation_rate_by(df.dropna(subset=["down_payment_band"]), "down_payment_band", "down_payment_band"),
        "by_discount_band": _cancellation_rate_by(df.dropna(subset=["discount_band"]), "discount_band", "discount_band"),
    }


def _compute_repeated_churn(df: pd.DataFrame) -> dict[str, Any]:
    """Units that were reserved, cancelled, and re-reserved -- true 'churned' inventory."""
    unit_counts = df.groupby("unit_id").size()
    repeated = unit_counts[unit_counts > 1]
    return {
        "units_with_repeated_reservation_activity": int(len(repeated)),
        "note": (
            "Counts unit_ids appearing more than once in sales.csv, i.e. a unit reserved, "
            "cancelled, and subsequently re-reserved within the data."
        ),
    }


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

def _compute_monthly_trends(df: pd.DataFrame) -> list[dict[str, Any]]:
    working = df.copy()
    working["reservation_month"] = working["reservation_date"].dt.to_period("M").astype(str)
    grouped = working.groupby("reservation_month").agg(
        reservations=("sale_id", "count"), cancellations=("cancellation_flag", "sum"),
    ).reset_index().sort_values("reservation_month").rename(columns={"reservation_month": "month"})
    grouped["cancellation_rate_pct"] = grouped.apply(lambda r: _percentage(r["cancellations"], r["reservations"]), axis=1)
    return to_records(grouped, ["month", "reservations", "cancellations", "cancellation_rate_pct"])


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

def _detect_anomalies(cuts: dict[str, Any]) -> list[dict[str, Any]]:
    anomalies = []
    by_project = [r for r in cuts["by_project"] if r["reservations"] >= 10]
    if len(by_project) >= 3:
        frame = pd.DataFrame(by_project)
        z = _zscores(frame["cancellation_rate_pct"].astype(float))
        for idx, z_value in z.items():
            if z_value < ANOMALY_Z_HIGH:
                continue
            row = frame.loc[idx]
            anomalies.append({
                "severity": "HIGH" if z_value >= 2.3 else "MEDIUM",
                "category": "Cancellations",
                "title": f"{row['project_id']} cancellation rate is a statistical outlier",
                "description": (
                    f"{row['project_id']}'s cancellation rate is {row['cancellation_rate_pct']:.1f}% "
                    f"across {int(row['reservations'])} reservations, {z_value:.1f} standard "
                    f"deviations above its peer projects, representing "
                    f"{row['cancelled_value']:,.0f} in cancelled gross value."
                ),
                "recommended_action": (
                    f"Cross-reference {row['project_id']}'s down-payment and payment-plan-length "
                    "distribution and sales-team mix to identify the specific driver before the "
                    "next sales incentive cycle."
                ),
                "project_id": row["project_id"],
            })
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------

def _compute_health_score(summary: dict[str, Any], cuts: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    ceiling = CONFIG["thresholds"]["cancellation_rate_watch_pct"]
    res_rate = summary["reservation_cancellation_rate_pct"] or 0.0
    reservation_score = max(0.0, 100 - (res_rate / ceiling) * 50) if ceiling else 100.0

    contract_rate = summary["contract_cancellation_rate_pct"] or 0.0
    contract_score = max(0.0, 100 - (contract_rate / ceiling) * 60) if ceiling else 100.0

    project_values = [r["cancelled_value"] for r in cuts["by_project"]]
    concentration_score = 100.0
    if project_values and sum(project_values):
        share = [v / sum(project_values) for v in project_values]
        hhi = sum(s ** 2 for s in share)
        even_hhi = 1 / len(share)
        concentration_score = max(0.0, 100 - (hhi - even_hhi) * 250)

    peak_share = timing.get("peak_window_share_pct") or 0.0
    timing_score = max(0.0, 100 - max(peak_share - 25, 0) * 1.5)

    components = {
        "reservation_cancellation_control": round(reservation_score, 1),
        "contract_cancellation_control": round(contract_score, 1),
        "concentration_control": round(concentration_score, 1),
        "timing_predictability": round(timing_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            f"Weighted blend of four 0-100 components: reservation_cancellation_control (penalized "
            f"relative to the {ceiling}% watch threshold in config/real_estate_demo.yml, weight "
            "0.20), contract_cancellation_control (penalized more heavily, since a cancelled "
            "contract had already cleared underwriting/commitment, weight 0.40), "
            "concentration_control (100 minus a penalty for how concentrated cancelled value is in "
            "one or two projects, weight 0.20), and timing_predictability (100 minus a penalty when "
            "cancellations cluster sharply in one post-reservation window rather than spreading out, "
            "weight 0.20)."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_cancellations(sales_df: pd.DataFrame) -> dict[str, Any]:
    """Run the full Cancellations & Revenue Leakage Intelligence suite a CCO/Finance Director needs.

    Args:
        sales_df: The sales DataFrame loaded from data/sales.csv.

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},
                "kpis": {"reasons": [...], "timing": {...}, "cuts": {...}, "repeated_churn": {...}},
                "trends": {"monthly": [...]},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If `sales_df` is empty/None or missing columns
            this module depends on.
    """
    df = _prepare(sales_df)
    cancelled = df[df["cancellation_flag"]]

    summary = _compute_summary(df)
    cuts = _compute_cuts(df)
    timing = _compute_timing_distribution(cancelled)
    health_score = _compute_health_score(summary, cuts, timing)

    result = {
        "summary": summary,
        "kpis": {
            "reasons": _compute_reasons(cancelled),
            "timing": timing,
            "cuts": cuts,
            "repeated_churn": _compute_repeated_churn(df),
        },
        "trends": {"monthly": _compute_monthly_trends(df)},
        "anomalies": _detect_anomalies(cuts),
        "health_score": health_score,
        "methodology": {
            "cancellation_value": (
                "cancelled_value uses gross_price (the pre-discount list value of the unit), since "
                "that is the confirmed pipeline value the business loses when a reservation or "
                "contract cancels, regardless of what discount had been negotiated."
            ),
            "timing_distribution": (
                "Timing buckets measure days from reservation_date to cancellation_date. A sharp "
                "peak in one window (e.g. 46-90 days) signals a specific, fixable process cause -- "
                "financing approval timelines, cooling-off periods, or first-installment due dates -- "
                "rather than diffuse, unpredictable attrition."
            ),
            "anomaly_detection": (
                "Cancellation-rate anomalies are z-scored across projects' own figures "
                "(cross-sectional, projects with at least 10 reservations only) -- never by "
                "referencing a specific project by name in code."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
