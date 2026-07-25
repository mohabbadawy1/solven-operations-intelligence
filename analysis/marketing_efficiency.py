"""Marketing Efficiency Intelligence engine.

Turns campaigns.csv into the numbers a Marketing Director or CCO needs
to answer "which channels are actually worth the spend": cost per
lead/qualified lead/appointment/site visit/reservation/contract,
attributed revenue, marketing efficiency ratio (attributed revenue /
spend), and the specific pattern this platform is built to catch --
channels that produce plentiful, cheap leads that convert far below
the portfolio average (a lead-quality problem marketing owns) versus
channels whose leads are expensive but convert well (a budget-
allocation opportunity, not a quality problem).

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

REQUIRED_CAMPAIGN_COLUMNS = [
    "campaign_id", "project_id", "platform", "channel", "start_date", "end_date", "spend",
    "leads", "qualified_leads", "appointments", "site_visits", "reservations", "contracts",
    "contracted_sales_value", "attributed_revenue",
]

MIN_LEADS_FOR_RANKING = 20
MAX_ANOMALIES_RETURNED = 6
ANOMALY_Z_HIGH = 1.6

# --- Marketing Efficiency Health Score ------------------------------------
#
#   roas                  average marketing efficiency ratio
#                         (attributed revenue / spend), rescaled
#   lead_quality           average qualified-lead rate across paid
#                         channels
#   funnel_yield            average lead-to-contract conversion rate
#                         across paid channels, rescaled
#   channel_diversification  100 minus a penalty for how concentrated
#                         spend is in a single channel
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "roas": 0.35,
    "lead_quality": 0.20,
    "funnel_yield": 0.30,
    "channel_diversification": 0.15,
}
ROAS_FULL_HEALTH_MULTIPLE = 12.0  # attributed revenue / spend at or above this reads as fully healthy
FUNNEL_YIELD_FULL_HEALTH_PCT = 3.0


def _prepare(campaigns_df: pd.DataFrame) -> pd.DataFrame:
    if campaigns_df is None or len(campaigns_df) == 0:
        raise ValueError("analyze_marketing_efficiency: campaigns DataFrame is empty or None")
    validate_columns(campaigns_df, REQUIRED_CAMPAIGN_COLUMNS, "analyze_marketing_efficiency: campaigns")

    df = campaigns_df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["month"] = df["start_date"].dt.to_period("M").astype(str)
    df["is_paid"] = df["spend"] > 0

    for numerator, denominator, out in [
        ("spend", "leads", "cost_per_lead"),
        ("spend", "qualified_leads", "cost_per_qualified_lead"),
        ("spend", "appointments", "cost_per_appointment"),
        ("spend", "site_visits", "cost_per_site_visit"),
        ("spend", "reservations", "cost_per_reservation"),
        ("spend", "contracts", "cost_per_contract"),
    ]:
        df[out] = df.apply(lambda r, n=numerator, d=denominator: (r[n] / r[d]) if r[d] else None, axis=1)

    df["qualified_lead_rate_pct"] = df.apply(lambda r: _percentage(r["qualified_leads"], r["leads"]), axis=1)
    df["lead_to_contract_pct"] = df.apply(lambda r: _percentage(r["contracts"], r["leads"]), axis=1)
    df["marketing_efficiency_ratio"] = df.apply(
        lambda r: round(r["attributed_revenue"] / r["spend"], 2) if r["spend"] else None, axis=1
    )
    return df


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    paid = df[df["is_paid"]]
    total_spend = paid["spend"].sum()
    total_leads = df["leads"].sum()
    total_qualified = df["qualified_leads"].sum()
    total_contracts = df["contracts"].sum()
    total_attributed_revenue = df["attributed_revenue"].sum()

    return {
        "total_campaigns": int(len(df)),
        "paid_campaigns": int(len(paid)),
        "total_spend": _safe_round(total_spend),
        "total_leads": int(total_leads),
        "total_qualified_leads": int(total_qualified),
        "total_reservations": int(df["reservations"].sum()),
        "total_contracts": int(total_contracts),
        "total_contracted_sales_value": _safe_round(df["contracted_sales_value"].sum()),
        "total_attributed_revenue": _safe_round(total_attributed_revenue),
        "overall_cost_per_lead": _safe_round(total_spend / total_leads) if total_leads else None,
        "overall_cost_per_contract": _safe_round(total_spend / total_contracts) if total_contracts else None,
        "overall_marketing_efficiency_ratio": _safe_round(total_attributed_revenue / total_spend) if total_spend else None,
        "portfolio_lead_to_contract_pct": _percentage(total_contracts, total_leads),
    }


# --------------------------------------------------------------------------
# Rankings: channel / platform / project
# --------------------------------------------------------------------------

def _channel_ranking(df: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = df.groupby("channel").agg(
        campaigns=("campaign_id", "count"), spend=("spend", "sum"), leads=("leads", "sum"),
        qualified_leads=("qualified_leads", "sum"), reservations=("reservations", "sum"),
        contracts=("contracts", "sum"), contracted_sales_value=("contracted_sales_value", "sum"),
        attributed_revenue=("attributed_revenue", "sum"),
    ).reset_index()
    grouped["cost_per_lead"] = grouped.apply(lambda r: _safe_round(r["spend"] / r["leads"]) if r["leads"] and r["spend"] else None, axis=1)
    grouped["cost_per_contract"] = grouped.apply(lambda r: _safe_round(r["spend"] / r["contracts"]) if r["contracts"] and r["spend"] else None, axis=1)
    grouped["qualified_lead_rate_pct"] = grouped.apply(lambda r: _percentage(r["qualified_leads"], r["leads"]), axis=1)
    grouped["lead_to_contract_pct"] = grouped.apply(lambda r: _percentage(r["contracts"], r["leads"]), axis=1)
    grouped["marketing_efficiency_ratio"] = grouped.apply(
        lambda r: round(r["attributed_revenue"] / r["spend"], 2) if r["spend"] else None, axis=1
    )
    grouped = grouped.sort_values("spend", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", grouped.index + 1)
    columns = ["rank", "channel", "campaigns", "spend", "leads", "contracts", "qualified_lead_rate_pct",
               "cost_per_lead", "lead_to_contract_pct", "cost_per_contract", "marketing_efficiency_ratio"]
    return to_records(grouped, columns)


def _project_ranking(df: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = df.groupby("project_id").agg(
        spend=("spend", "sum"), leads=("leads", "sum"), contracts=("contracts", "sum"),
        attributed_revenue=("attributed_revenue", "sum"),
    ).reset_index()
    grouped["marketing_efficiency_ratio"] = grouped.apply(
        lambda r: round(r["attributed_revenue"] / r["spend"], 2) if r["spend"] else None, axis=1
    )
    grouped["cost_per_contract"] = grouped.apply(lambda r: _safe_round(r["spend"] / r["contracts"]) if r["contracts"] else None, axis=1)
    grouped = grouped.sort_values("marketing_efficiency_ratio", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", grouped.index + 1)
    return to_records(grouped, ["rank", "project_id", "spend", "leads", "contracts", "cost_per_contract", "marketing_efficiency_ratio"])


def _compute_rankings(df: pd.DataFrame) -> dict[str, Any]:
    by_channel = _channel_ranking(df)
    by_project = _project_ranking(df)
    eligible = [row for row in by_channel if row["leads"] and row["leads"] >= MIN_LEADS_FOR_RANKING]
    most_efficient = max(eligible, key=lambda r: r["marketing_efficiency_ratio"] or 0, default=None)
    least_efficient = min(
        (r for r in eligible if r["spend"]), key=lambda r: r["marketing_efficiency_ratio"] or 0, default=None
    )
    return {
        "by_channel": by_channel,
        "by_project": by_project,
        "most_efficient_channel": most_efficient["channel"] if most_efficient else None,
        "least_efficient_paid_channel": least_efficient["channel"] if least_efficient else None,
    }


# --------------------------------------------------------------------------
# Cheap-but-low-quality vs. expensive-but-high-value channel detection
# --------------------------------------------------------------------------

def _compute_channel_quality_matrix(by_channel: list[dict[str, Any]], portfolio_conversion_pct: float) -> dict[str, Any]:
    """Classify each paid channel against the portfolio's own average lead-to-contract rate."""
    cheap_low_quality, expensive_high_value = [], []
    channel_cpl = [r["cost_per_lead"] for r in by_channel if r["cost_per_lead"] is not None]
    median_cpl = sorted(channel_cpl)[len(channel_cpl) // 2] if channel_cpl else 0.0

    for row in by_channel:
        if row["cost_per_lead"] is None or row["leads"] < MIN_LEADS_FOR_RANKING:
            continue
        conversion = row["lead_to_contract_pct"] or 0.0
        if row["cost_per_lead"] <= median_cpl and conversion < portfolio_conversion_pct * 0.6:
            cheap_low_quality.append({
                "channel": row["channel"], "cost_per_lead": row["cost_per_lead"],
                "lead_to_contract_pct": conversion, "portfolio_average_pct": portfolio_conversion_pct,
            })
        elif row["cost_per_lead"] > median_cpl and conversion >= portfolio_conversion_pct:
            expensive_high_value.append({
                "channel": row["channel"], "cost_per_lead": row["cost_per_lead"],
                "lead_to_contract_pct": conversion, "portfolio_average_pct": portfolio_conversion_pct,
            })

    return {
        "median_cost_per_lead": _safe_round(median_cpl),
        "cheap_but_low_quality_channels": cheap_low_quality,
        "expensive_but_high_value_channels": expensive_high_value,
    }


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

def _compute_monthly_trends(df: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = df.groupby("month").agg(
        spend=("spend", "sum"), leads=("leads", "sum"), contracts=("contracts", "sum"),
        attributed_revenue=("attributed_revenue", "sum"),
    ).reset_index().sort_values("month")
    grouped["marketing_efficiency_ratio"] = grouped.apply(
        lambda r: round(r["attributed_revenue"] / r["spend"], 2) if r["spend"] else None, axis=1
    )
    grouped["cost_per_lead"] = grouped.apply(lambda r: _safe_round(r["spend"] / r["leads"]) if r["leads"] else None, axis=1)
    return to_records(grouped, ["month", "spend", "leads", "contracts", "cost_per_lead", "marketing_efficiency_ratio"])


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

def _detect_anomalies(by_channel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag channels whose cost-per-contract is a statistical outlier vs. its peer channels."""
    eligible = [r for r in by_channel if r["cost_per_contract"] is not None and r["contracts"] and r["contracts"] >= 3]
    if len(eligible) < 3:
        return []
    frame = pd.DataFrame(eligible)
    z = _zscores(frame["cost_per_contract"].astype(float))
    anomalies = []
    for idx, z_value in z.items():
        if z_value < ANOMALY_Z_HIGH:
            continue
        row = frame.loc[idx]
        anomalies.append({
            "severity": "HIGH" if z_value >= 2.3 else "MEDIUM",
            "category": "Marketing Efficiency",
            "title": f"{row['channel']} has a disproportionately high cost per contract",
            "description": (
                f"{row['channel']}'s cost per contract is {row['cost_per_contract']:,.0f}, "
                f"{z_value:.1f} standard deviations above its peer channels, across "
                f"{int(row['contracts'])} contracts."
            ),
            "recommended_action": (
                f"Reallocate a portion of {row['channel']}'s budget toward higher-efficiency "
                "channels, or audit its targeting/creative for the specific cause of the gap "
                "before the next planning cycle."
            ),
            "channel": row["channel"],
        })
    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------

def _compute_health_score(summary: dict[str, Any], rankings: dict[str, Any]) -> dict[str, Any]:
    ratio = summary["overall_marketing_efficiency_ratio"] or 0.0
    roas_score = min(100.0, (ratio / ROAS_FULL_HEALTH_MULTIPLE) * 100)

    paid_channels = [r for r in rankings["by_channel"] if r["spend"]]
    quality_values = [r["qualified_lead_rate_pct"] for r in paid_channels if r["qualified_lead_rate_pct"] is not None]
    lead_quality_score = sum(quality_values) / len(quality_values) if quality_values else 50.0

    yield_pct = summary["portfolio_lead_to_contract_pct"] or 0.0
    funnel_yield_score = min(100.0, (yield_pct / FUNNEL_YIELD_FULL_HEALTH_PCT) * 100)

    spend_values = [r["spend"] for r in paid_channels]
    diversification_score = 100.0
    if spend_values and sum(spend_values):
        shares = [v / sum(spend_values) for v in spend_values]
        hhi = sum(s ** 2 for s in shares)
        even_hhi = 1 / len(shares)
        diversification_score = max(0.0, 100 - (hhi - even_hhi) * 250)

    components = {
        "roas": round(roas_score, 1),
        "lead_quality": round(lead_quality_score, 1),
        "funnel_yield": round(funnel_yield_score, 1),
        "channel_diversification": round(diversification_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            f"Weighted blend of four 0-100 components: roas (portfolio attributed-revenue/spend "
            f"ratio rescaled against a {ROAS_FULL_HEALTH_MULTIPLE}x full-health benchmark, weight "
            "0.35), lead_quality (average qualified-lead rate across paid channels, weight 0.20), "
            f"funnel_yield (portfolio lead-to-contract rate rescaled against a "
            f"{FUNNEL_YIELD_FULL_HEALTH_PCT}% full-health benchmark, weight 0.30), and "
            "channel_diversification (100 minus a penalty for how concentrated spend is in one "
            "or two channels, weight 0.15)."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_marketing_efficiency(campaigns_df: pd.DataFrame) -> dict[str, Any]:
    """Run the full Marketing Efficiency Intelligence suite a Marketing Director/CCO needs.

    Args:
        campaigns_df: The campaigns DataFrame loaded from data/campaigns.csv.

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},
                "kpis": {"channel_quality_matrix": {...}},
                "rankings": {...},
                "trends": {"monthly": [...]},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If `campaigns_df` is empty/None or missing columns
            this module depends on.
    """
    df = _prepare(campaigns_df)

    summary = _compute_summary(df)
    rankings = _compute_rankings(df)
    channel_quality_matrix = _compute_channel_quality_matrix(rankings["by_channel"], summary["portfolio_lead_to_contract_pct"])
    health_score = _compute_health_score(summary, rankings)

    result = {
        "summary": summary,
        "kpis": {"channel_quality_matrix": channel_quality_matrix},
        "rankings": rankings,
        "trends": {"monthly": _compute_monthly_trends(df)},
        "anomalies": _detect_anomalies(rankings["by_channel"]),
        "health_score": health_score,
        "methodology": {
            "attribution": (
                "Every campaign figure (leads, appointments, site visits, reservations, contracts, "
                "attributed_revenue) is read directly from campaigns.csv's own last-touch "
                "attribution fields, as exported from the marketing platform/CRM integration -- "
                "this module performs no multi-touch attribution modeling."
            ),
            "channel_quality_matrix": (
                "A channel is flagged 'cheap but low quality' when its cost per lead is at or below "
                "the portfolio's median and its lead-to-contract rate is under 60% of the portfolio "
                "average; 'expensive but high value' when its cost per lead is above the median but "
                "its conversion rate meets or beats the portfolio average. Channels below the "
                f"{MIN_LEADS_FOR_RANKING}-lead minimum are excluded from this classification as "
                "statistically too thin to judge."
            ),
            "anomaly_detection": (
                "Cost-per-contract anomalies are z-scored across channels' own figures "
                "(cross-sectional, not time-based) -- never by referencing a specific channel by "
                "name in code."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
