"""Commercial Performance Intelligence engine.

Turns sales.csv (+ units.csv for inventory-value context) into the
numbers a CCO, Sales Director, or Finance Director opens the report
to see first: are we on track against target, what is net contracted
sales actually worth after discounts, which projects/phases/unit
types/channels are carrying the business, and how has that changed
period over period. Also owns pricing/discount analytics (gross-to-net
realization, discount leakage) -- kept in this module rather than a
separate one, since discounting is a sales-performance lever, not an
independent domain, and every discount metric here is already grouped
by the same project/phase/team cuts sales performance uses.

Output shape mirrors the platform's established convention:
{summary, kpis, rankings, trends, period_comparison, anomalies,
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
    period_over_period,
    sort_and_limit_anomalies,
    to_records,
    validate_columns,
)

REQUIRED_SALES_COLUMNS = [
    "sale_id", "unit_id", "project_id", "customer_id", "reservation_date", "contract_date",
    "sales_agent_id", "team_id", "broker_id", "gross_price", "discount_value", "discount_pct",
    "net_sales_value", "payment_plan_years", "down_payment_value", "sales_channel", "buyer_type",
    "contract_status", "cancellation_flag", "cancellation_date",
]
REQUIRED_UNIT_COLUMNS = ["unit_id", "project_id", "unit_type", "built_up_area_sqm", "list_price", "price_per_sqm"]

# A sale counts toward "net contracted" sales value once it has a
# contract_date and was never cancelled -- reservations that never
# became a contract, and contracts later cancelled, are excluded from
# revenue-facing KPIs but still counted in funnel/cancellation metrics
# elsewhere in the platform.
CONTRACTED_NOT_CANCELLED = lambda df: df["contract_date"].notna() & ~df["cancellation_flag"]

TOP_N = 5
MAX_ANOMALIES_RETURNED = 8

ANOMALY_Z_MEDIUM = 1.5
ANOMALY_Z_HIGH = 2.2

# --- Sales Performance Health Score ---------------------------------------
#
#   target_attainment     YTD net contracted sales vs. the pro-rated
#                         annual target, capped at 100
#   realization           gross-to-net realization rate (100 - avg
#                         discount%), the direct margin-erosion signal
#   velocity_consistency   100 minus a penalty for how unevenly sales
#                         velocity is spread across projects -- a
#                         portfolio with one carrying project and four
#                         stalled ones is riskier than an even spread
#   cancellation_control    100 minus a penalty for the contract
#                         cancellation rate
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "target_attainment": 0.35,
    "realization": 0.25,
    "velocity_consistency": 0.15,
    "cancellation_control": 0.25,
}


def _prepare(sales_df: pd.DataFrame, units_df: pd.DataFrame) -> pd.DataFrame:
    if sales_df is None or len(sales_df) == 0:
        raise ValueError("analyze_sales_performance: sales DataFrame is empty or None")
    if units_df is None or len(units_df) == 0:
        raise ValueError("analyze_sales_performance: units DataFrame is empty or None")
    validate_columns(sales_df, REQUIRED_SALES_COLUMNS, "analyze_sales_performance: sales")
    validate_columns(units_df, REQUIRED_UNIT_COLUMNS, "analyze_sales_performance: units")

    df = sales_df.copy()
    for col in ("reservation_date", "contract_date", "cancellation_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["cancellation_flag"] = df["cancellation_flag"].astype(bool)
    df["month"] = df["reservation_date"].dt.to_period("M").astype(str)
    df["is_net_contracted"] = CONTRACTED_NOT_CANCELLED(df)

    df = df.merge(units_df[["unit_id", "unit_type", "built_up_area_sqm"]], on="unit_id", how="left", suffixes=("", "_unit"))
    df["price_per_sqm_realized"] = df.apply(
        lambda r: _safe_round(r["net_sales_value"] / r["built_up_area_sqm"]) if r["built_up_area_sqm"] else None, axis=1
    )
    return df


def _as_of(df: pd.DataFrame) -> pd.Timestamp:
    return df["reservation_date"].max()


def _period_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for current/prior month, quarter, YTD, and rolling-90 windows."""
    as_of = _as_of(df)
    current_month_start = as_of.to_period("M").start_time
    prior_month_start = (current_month_start - pd.DateOffset(months=1))
    prior_month_end = current_month_start - pd.Timedelta(days=1)

    current_quarter_start = as_of.to_period("Q").start_time
    prior_quarter_start = (current_quarter_start - pd.DateOffset(months=3))
    prior_quarter_end = current_quarter_start - pd.Timedelta(days=1)

    ytd_start = pd.Timestamp(year=as_of.year, month=1, day=1)
    prior_ytd_start = pd.Timestamp(year=as_of.year - 1, month=1, day=1)
    prior_ytd_end = pd.Timestamp(year=as_of.year - 1, month=as_of.month, day=1) + pd.offsets.MonthEnd(0)

    rolling_start = as_of - pd.Timedelta(days=90)
    prior_rolling_start = as_of - pd.Timedelta(days=180)
    prior_rolling_end = as_of - pd.Timedelta(days=91)

    rd = df["reservation_date"]
    return {
        "current_month": (rd >= current_month_start) & (rd <= as_of),
        "prior_month": (rd >= prior_month_start) & (rd <= prior_month_end),
        "current_quarter": (rd >= current_quarter_start) & (rd <= as_of),
        "prior_quarter": (rd >= prior_quarter_start) & (rd <= prior_quarter_end),
        "ytd": (rd >= ytd_start) & (rd <= as_of),
        "prior_ytd": (rd >= prior_ytd_start) & (rd <= prior_ytd_end),
        "rolling_90": (rd >= rolling_start) & (rd <= as_of),
        "prior_rolling_90": (rd >= prior_rolling_start) & (rd <= prior_rolling_end),
    }


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    net = df[df["is_net_contracted"]]
    reservations = len(df)
    contracts = int(df["contract_date"].notna().sum())
    cancellations = int(df["cancellation_flag"].sum())

    return {
        "total_reservations": reservations,
        "total_contracts": contracts,
        "total_cancellations": cancellations,
        "units_sold_net": int(len(net)),
        "gross_sales_value": _safe_round(df["gross_price"].sum()),
        "net_contracted_sales_value": _safe_round(net["net_sales_value"].sum()),
        "total_discount_value": _safe_round(net["discount_value"].sum()),
        "average_discount_pct": _safe_round(net["discount_pct"].mean()),
        "average_unit_value": _safe_round(net["net_sales_value"].mean()),
        "average_price_per_sqm": _safe_round(net["price_per_sqm_realized"].mean()),
        "gross_to_net_realization_pct": _safe_round(
            100 * net["net_sales_value"].sum() / net["gross_price"].sum()
        ) if net["gross_price"].sum() else None,
        "reservation_to_contract_conversion_pct": _percentage(contracts, reservations),
        "contract_cancellation_rate_pct": _percentage(cancellations, contracts) if contracts else 0.0,
    }


# --------------------------------------------------------------------------
# Target attainment
# --------------------------------------------------------------------------

def _compute_target_attainment(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    targets = CONFIG["targets"]
    as_of = _as_of(df)
    months_elapsed = as_of.month
    prorated_sales_target = round(targets["annual_net_contracted_sales"] * months_elapsed / 12, 2)
    prorated_units_target = round(targets["annual_units_sold"] * months_elapsed / 12, 1)

    ytd_net = df[masks["ytd"] & df["is_net_contracted"]]
    ytd_sales_value = _safe_round(ytd_net["net_sales_value"].sum()) or 0.0
    ytd_units = int(len(ytd_net))

    return {
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "ytd_net_contracted_sales_value": ytd_sales_value,
        "prorated_annual_sales_target": prorated_sales_target,
        "sales_target_attainment_pct": _percentage(ytd_sales_value, prorated_sales_target),
        "ytd_units_sold": ytd_units,
        "prorated_annual_units_target": prorated_units_target,
        "units_target_attainment_pct": _percentage(ytd_units, prorated_units_target),
        "full_year_annual_sales_target": targets["annual_net_contracted_sales"],
        "full_year_annual_units_target": targets["annual_units_sold"],
        "methodology": (
            "Targets are pro-rated by calendar months elapsed in the reporting year "
            f"({months_elapsed}/12) against the full annual targets in "
            "config/real_estate_demo.yml -- a straight-line pacing assumption, not a "
            "seasonality-adjusted budget phasing."
        ),
    }


# --------------------------------------------------------------------------
# Rankings: project / phase / unit type / channel
# --------------------------------------------------------------------------

def _ranking_by(df: pd.DataFrame, group_col: str, label: str) -> list[dict[str, Any]]:
    net = df[df["is_net_contracted"]]
    grouped = net.groupby(group_col).agg(
        units_sold=("sale_id", "count"),
        net_sales_value=("net_sales_value", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_price_per_sqm=("price_per_sqm_realized", "mean"),
    ).reset_index().rename(columns={group_col: label})
    grouped = grouped.sort_values("net_sales_value", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", grouped.index + 1)
    return to_records(grouped, ["rank", label, "units_sold", "net_sales_value", "avg_discount_pct", "avg_price_per_sqm"])


def _compute_rankings(df: pd.DataFrame) -> dict[str, Any]:
    by_project = _ranking_by(df, "project_id", "project_id")
    return {
        "by_project": by_project,
        "by_unit_type": _ranking_by(df, "unit_type", "unit_type"),
        "by_channel": _ranking_by(df, "sales_channel", "sales_channel"),
        "by_buyer_type": _ranking_by(df, "buyer_type", "buyer_type"),
        "strongest_project": by_project[0]["project_id"] if by_project else None,
        "weakest_project": by_project[-1]["project_id"] if by_project else None,
    }


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

def _compute_monthly_trends(df: pd.DataFrame) -> list[dict[str, Any]]:
    net = df[df["is_net_contracted"]]
    grouped = net.groupby("month").agg(
        units_sold=("sale_id", "count"),
        net_sales_value=("net_sales_value", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
    ).reset_index().sort_values("month")

    reservations_by_month = df.groupby("month")["sale_id"].count()
    grouped["reservations"] = grouped["month"].map(reservations_by_month).fillna(0).astype(int)

    trend = to_records(grouped, ["month", "units_sold", "net_sales_value", "avg_discount_pct", "reservations"])
    # rolling 3-month net sales value, appended per point for a smoothed trend line
    values = [row["net_sales_value"] for row in trend]
    for i, row in enumerate(trend):
        window = values[max(0, i - 2): i + 1]
        row["rolling_3month_net_sales_value"] = _safe_round(sum(window) / len(window)) if window else None
    return trend


# --------------------------------------------------------------------------
# Period comparison
# --------------------------------------------------------------------------

def _compute_period_comparison(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, Any]:
    def _net_value(mask):
        return _safe_round(df.loc[mask & df["is_net_contracted"], "net_sales_value"].sum()) or 0.0

    def _units(mask):
        return int((mask & df["is_net_contracted"]).sum())

    return {
        "month_over_month": {
            "net_sales_value": period_over_period(_net_value(masks["current_month"]), _net_value(masks["prior_month"])),
            "units_sold": period_over_period(_units(masks["current_month"]), _units(masks["prior_month"])),
        },
        "quarter_over_quarter": {
            "net_sales_value": period_over_period(_net_value(masks["current_quarter"]), _net_value(masks["prior_quarter"])),
            "units_sold": period_over_period(_units(masks["current_quarter"]), _units(masks["prior_quarter"])),
        },
        "ytd_vs_prior_ytd": {
            "net_sales_value": period_over_period(_net_value(masks["ytd"]), _net_value(masks["prior_ytd"])),
            "units_sold": period_over_period(_units(masks["ytd"]), _units(masks["prior_ytd"])),
        },
        "rolling_90_days": {
            "net_sales_value": period_over_period(_net_value(masks["rolling_90"]), _net_value(masks["prior_rolling_90"])),
            "units_sold": period_over_period(_units(masks["rolling_90"]), _units(masks["prior_rolling_90"])),
        },
    }


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

def _detect_discount_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag project-months where average discount is a statistical outlier vs. that project's own baseline."""
    net = df[df["is_net_contracted"]]
    grouped = net.groupby(["project_id", "month"]).agg(
        units=("sale_id", "count"), avg_discount=("discount_pct", "mean"),
    ).reset_index()
    grouped = grouped[grouped["units"] >= 3]

    anomalies = []
    for project_id, group in grouped.groupby("project_id"):
        if len(group) < 3:
            continue
        group = group.sort_values("month").reset_index(drop=True)
        z = _zscores(group["avg_discount"])
        for idx, z_value in z.items():
            if z_value < ANOMALY_Z_MEDIUM:
                continue
            row = group.loc[idx]
            severity = "HIGH" if z_value >= ANOMALY_Z_HIGH else "MEDIUM"
            anomalies.append({
                "severity": severity,
                "category": "Pricing",
                "title": f"{project_id} discounting spike in {row['month']}",
                "description": (
                    f"{project_id}'s average discount reached {row['avg_discount']:.1f}% in "
                    f"{row['month']}, {z_value:.1f} standard deviations above its own baseline, "
                    f"across {int(row['units'])} contracted units."
                ),
                "recommended_action": (
                    f"Review {project_id}'s pricing approvals for {row['month']} to confirm the "
                    "discount level was authorized policy rather than ad hoc deal-making."
                ),
                "project_id": project_id,
                "month": row["month"],
            })
    return anomalies


def _detect_velocity_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag project-months with a statistically low net-contracted unit count vs. that project's own baseline."""
    net = df[df["is_net_contracted"]]
    grouped = net.groupby(["project_id", "month"]).agg(units=("sale_id", "count")).reset_index()

    anomalies = []
    for project_id, group in grouped.groupby("project_id"):
        if len(group) < 3:
            continue
        group = group.sort_values("month").reset_index(drop=True)
        z = _zscores(group["units"])
        for idx, z_value in z.items():
            if z_value > -ANOMALY_Z_MEDIUM:
                continue
            row = group.loc[idx]
            severity = "HIGH" if z_value <= -ANOMALY_Z_HIGH else "MEDIUM"
            anomalies.append({
                "severity": severity,
                "category": "Sales Velocity",
                "title": f"{project_id} sales velocity trough in {row['month']}",
                "description": (
                    f"{project_id} closed only {int(row['units'])} net contracts in {row['month']}, "
                    f"{abs(z_value):.1f} standard deviations below its own monthly baseline."
                ),
                "recommended_action": (
                    f"Cross-reference {project_id}'s lead funnel and marketing spend for {row['month']} "
                    "to determine whether this is a demand, pricing, or capacity issue."
                ),
                "project_id": project_id,
                "month": row["month"],
            })
    return anomalies


def _detect_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    anomalies = _detect_discount_anomalies(df) + _detect_velocity_anomalies(df)
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------

def _compute_health_score(summary: dict[str, Any], target_attainment: dict[str, Any], rankings: dict[str, Any]) -> dict[str, Any]:
    target_score = min(100.0, target_attainment["sales_target_attainment_pct"])

    realization_pct = summary["gross_to_net_realization_pct"] or 100.0
    realization_score = max(0.0, min(100.0, realization_pct))

    # Herfindahl-style concentration: 1/n is perfectly even across n
    # projects; the further above that a portfolio's HHI sits, the more
    # net sales value is concentrated in one or two carrying projects,
    # which is a resilience risk even if total sales look healthy.
    project_values = [row["net_sales_value"] for row in rankings["by_project"]]
    velocity_consistency_score = 100.0
    if project_values and sum(project_values):
        share = [v / sum(project_values) for v in project_values]
        hhi = sum(s ** 2 for s in share)
        even_hhi = 1 / len(share)
        velocity_consistency_score = max(0.0, 100 - (hhi - even_hhi) * 300)

    cancellation_rate = summary["contract_cancellation_rate_pct"] or 0.0
    cancellation_score = max(0.0, 100 - cancellation_rate * 4.0)

    components = {
        "target_attainment": round(target_score, 1),
        "realization": round(realization_score, 1),
        "velocity_consistency": round(velocity_consistency_score, 1),
        "cancellation_control": round(cancellation_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            "Weighted blend of four 0-100 components: target_attainment (YTD net contracted "
            "sales vs. pro-rated annual target, capped at 100, weight 0.35), realization "
            "(gross-to-net realization rate after discounts, weight 0.25), velocity_consistency "
            "(100 minus a penalty for how concentrated net sales value is in one or two projects "
            "vs. spread evenly across the portfolio, weight 0.15), and cancellation_control (100 "
            "minus 4 points per percentage point of contract cancellation rate, weight 0.25)."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_sales_performance(sales_df: pd.DataFrame, units_df: pd.DataFrame) -> dict[str, Any]:
    """Run the full Commercial Performance Intelligence suite a CCO/Sales Director needs.

    Args:
        sales_df: The sales DataFrame loaded from data/sales.csv.
        units_df: The units DataFrame loaded from data/units.csv, used
            to attribute unit_type and area to each sale.

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},
                "kpis": {"target_attainment": {...}},
                "rankings": {...},
                "trends": {"monthly": [...]},
                "period_comparison": {...},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If either DataFrame is empty/None or missing
            columns this module depends on.
    """
    df = _prepare(sales_df, units_df)
    masks = _period_masks(df)

    summary = _compute_summary(df)
    target_attainment = _compute_target_attainment(df, masks)
    rankings = _compute_rankings(df)
    health_score = _compute_health_score(summary, target_attainment, rankings)

    result = {
        "summary": summary,
        "kpis": {"target_attainment": target_attainment},
        "rankings": rankings,
        "trends": {"monthly": _compute_monthly_trends(df)},
        "period_comparison": _compute_period_comparison(df, masks),
        "anomalies": _detect_anomalies(df),
        "health_score": health_score,
        "methodology": {
            "net_contracted_definition": (
                "A sale counts as 'net contracted' once it has a contract_date and was never "
                "cancelled. Pure reservations that never became a contract, and contracts later "
                "cancelled, are excluded from revenue-facing KPIs (net_contracted_sales_value, "
                "units_sold_net, rankings, trends) but remain visible in reservation/cancellation "
                "counts and are analyzed in full by analysis/cancellations.py."
            ),
            "period_windows": (
                "Current/prior month, quarter, and YTD windows are computed relative to the most "
                "recent reservation_date in the data (not the system clock), so this module "
                "produces the same comparison regardless of when it happens to run."
            ),
            "anomaly_detection": (
                "Anomalies are detected by z-scoring each project's own monthly discount level and "
                "monthly net-contracted unit count against that project's own historical baseline -- "
                "never by referencing a specific project or month in code."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
