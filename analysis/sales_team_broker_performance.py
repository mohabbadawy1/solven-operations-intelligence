"""Sales Team & Broker Performance Intelligence engine.

Turns sales.csv (+ leads.csv for pipeline/workload context, +
brokers.csv for roster metadata) into the numbers a Sales Director or
CCO needs to answer "who is actually selling well, once you look past
raw volume": conversion quality, discount discipline, cancellation
rate, and commission-adjusted contribution -- for both direct sales
agents/teams and third-party brokers. The two are analyzed together
in one module because they answer the same underlying question
("which channel/person converts quality contracts, not just
reservations") with near-identical KPI shapes; kept as one file rather
than two nearly-duplicate ones.

Deliberately never ranks purely on absolute sales volume: every
ranking either normalizes by lead/reservation volume (a rate, not a
count) or is paired with a quality metric (cancellation rate, average
discount) alongside it, so a high-volume-low-quality performer is
visible as exactly that, not disguised as a top performer.

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

REQUIRED_SALES_COLUMNS = [
    "sale_id", "project_id", "sales_agent_id", "team_id", "broker_id", "gross_price",
    "discount_pct", "net_sales_value", "commission_value", "contract_date", "contract_status",
    "cancellation_flag", "sales_channel",
]
REQUIRED_BROKER_COLUMNS = [
    "broker_id", "broker_name", "broker_type", "commission_rate", "data_quality_score",
    "compliance_status", "lead_volume", "reservation_count", "contract_count",
    "cancelled_contracts", "gross_sales_value", "commission_paid", "average_discount_pct",
]

MIN_RESERVATIONS_FOR_RANKING = 5
MAX_ANOMALIES_RETURNED = 8
ANOMALY_Z_HIGH = 1.6

# --- Commercial Channel Health Score (agents + brokers combined) ---------
#
#   direct_conversion_quality   average reservation-to-contract rate
#                              across direct sales agents/teams
#   broker_conversion_quality    average reservation-to-contract rate
#                              across brokers
#   discount_discipline           100 minus a penalty for the spread
#                              between the best- and worst-disciplined
#                              channel's average discount
#   broker_concentration           100 minus a penalty for how much of
#                              broker-sourced volume sits with the
#                              single largest broker -- concentration
#                              risk, independent of that broker's
#                              individual quality
HEALTH_SCORE_WEIGHTS: dict[str, float] = {
    "direct_conversion_quality": 0.30,
    "broker_conversion_quality": 0.20,
    "discount_discipline": 0.25,
    "broker_concentration": 0.25,
}


def _prepare_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    if sales_df is None or len(sales_df) == 0:
        raise ValueError("analyze_sales_team_broker_performance: sales DataFrame is empty or None")
    validate_columns(sales_df, REQUIRED_SALES_COLUMNS, "analyze_sales_team_broker_performance: sales")
    df = sales_df.copy()
    df["contract_date"] = pd.to_datetime(df["contract_date"], errors="coerce")
    df["cancellation_flag"] = df["cancellation_flag"].astype(bool)
    df["is_net_contracted"] = df["contract_date"].notna() & ~df["cancellation_flag"]
    return df


def _prepare_brokers(brokers_df: pd.DataFrame) -> pd.DataFrame:
    if brokers_df is None or len(brokers_df) == 0:
        raise ValueError("analyze_sales_team_broker_performance: brokers DataFrame is empty or None")
    validate_columns(brokers_df, REQUIRED_BROKER_COLUMNS, "analyze_sales_team_broker_performance: brokers")
    return brokers_df.copy()


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------

def _compute_summary(sales: pd.DataFrame, brokers: pd.DataFrame) -> dict[str, Any]:
    direct = sales[sales["sales_channel"] == "Direct"]
    broker_sourced = sales[sales["sales_channel"] == "Broker"]

    direct_contracts = int(direct["contract_date"].notna().sum())
    broker_contracts = int(broker_sourced["contract_date"].notna().sum())

    return {
        "total_agents": int(sales["sales_agent_id"].nunique()),
        "total_active_brokers": int((brokers["reservation_count"] > 0).sum()),
        "direct_reservations": int(len(direct)),
        "direct_contracts": direct_contracts,
        "direct_reservation_to_contract_pct": _percentage(direct_contracts, len(direct)) if len(direct) else 0.0,
        "broker_reservations": int(len(broker_sourced)),
        "broker_contracts": broker_contracts,
        "broker_reservation_to_contract_pct": _percentage(broker_contracts, len(broker_sourced)) if len(broker_sourced) else 0.0,
        "broker_share_of_reservations_pct": _percentage(len(broker_sourced), len(sales)),
        "total_commission_paid": _safe_round(sales.loc[sales["is_net_contracted"], "commission_value"].sum()),
        "commission_as_pct_of_net_sales": _safe_round(
            100 * sales.loc[sales["is_net_contracted"], "commission_value"].sum()
            / sales.loc[sales["is_net_contracted"], "net_sales_value"].sum()
        ) if sales.loc[sales["is_net_contracted"], "net_sales_value"].sum() else None,
    }


# --------------------------------------------------------------------------
# Sales agent / team rankings (volume-adjusted, never raw volume alone)
# --------------------------------------------------------------------------

def _compute_agent_ranking(sales: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = sales.groupby("sales_agent_id").agg(
        reservations=("sale_id", "count"),
        contracts=("contract_date", lambda s: s.notna().sum()),
        cancellations=("cancellation_flag", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        team_id=("team_id", "first"),
        project_id=("project_id", "first"),
    ).reset_index()
    net_by_agent = sales[sales["is_net_contracted"]].groupby("sales_agent_id")["net_sales_value"].sum()
    grouped["net_sales_value"] = grouped["sales_agent_id"].map(net_by_agent).fillna(0)
    grouped["reservation_to_contract_pct"] = grouped.apply(lambda r: _percentage(r["contracts"], r["reservations"]), axis=1)
    grouped["cancellation_rate_pct"] = grouped.apply(lambda r: _percentage(r["cancellations"], r["reservations"]), axis=1)

    eligible = grouped[grouped["reservations"] >= MIN_RESERVATIONS_FOR_RANKING].copy()
    eligible = eligible.sort_values(
        ["reservation_to_contract_pct", "cancellation_rate_pct"], ascending=[False, True]
    ).reset_index(drop=True)
    eligible.insert(0, "rank", eligible.index + 1)

    columns = ["rank", "sales_agent_id", "team_id", "project_id", "reservations", "contracts",
               "reservation_to_contract_pct", "cancellation_rate_pct", "avg_discount_pct", "net_sales_value"]
    return to_records(eligible, columns)


def _compute_team_ranking(sales: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = sales.groupby("team_id").agg(
        reservations=("sale_id", "count"),
        contracts=("contract_date", lambda s: s.notna().sum()),
        cancellations=("cancellation_flag", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        project_id=("project_id", "first"),
    ).reset_index()
    net_by_team = sales[sales["is_net_contracted"]].groupby("team_id")["net_sales_value"].sum()
    grouped["net_sales_value"] = grouped["team_id"].map(net_by_team).fillna(0)
    grouped["reservation_to_contract_pct"] = grouped.apply(lambda r: _percentage(r["contracts"], r["reservations"]), axis=1)
    grouped["cancellation_rate_pct"] = grouped.apply(lambda r: _percentage(r["cancellations"], r["reservations"]), axis=1)
    grouped = grouped.sort_values(
        ["reservation_to_contract_pct", "cancellation_rate_pct"], ascending=[False, True]
    ).reset_index(drop=True)
    grouped.insert(0, "rank", grouped.index + 1)

    columns = ["rank", "team_id", "project_id", "reservations", "contracts",
               "reservation_to_contract_pct", "cancellation_rate_pct", "avg_discount_pct", "net_sales_value"]
    return to_records(grouped, columns)


# --------------------------------------------------------------------------
# Broker ranking + commission-adjusted contribution
# --------------------------------------------------------------------------

def _compute_broker_ranking(brokers: pd.DataFrame) -> list[dict[str, Any]]:
    df = brokers.copy()
    df["cancellation_rate_pct"] = df.apply(
        lambda r: _percentage(r["cancelled_contracts"], r["reservation_count"]) if r["reservation_count"] else 0.0, axis=1
    )
    df["reservation_to_contract_pct"] = df.apply(
        lambda r: _percentage(r["contract_count"], r["reservation_count"]) if r["reservation_count"] else 0.0, axis=1
    )
    # Net contribution: gross sales less the discount already embedded
    # in gross_sales_value's underlying contracts (approximated via
    # average_discount_pct) and less commission paid -- the figure that
    # answers "what does this broker actually net the business," not
    # just headline reservation volume.
    df["estimated_net_contribution"] = df.apply(
        lambda r: _safe_round(r["gross_sales_value"] * (1 - r["average_discount_pct"] / 100) - r["commission_paid"]),
        axis=1,
    )

    eligible = df[df["reservation_count"] >= MIN_RESERVATIONS_FOR_RANKING].copy()
    eligible = eligible.sort_values("estimated_net_contribution", ascending=False).reset_index(drop=True)
    eligible.insert(0, "rank", eligible.index + 1)

    columns = ["rank", "broker_id", "broker_name", "broker_type", "reservation_count", "contract_count",
               "reservation_to_contract_pct", "cancellation_rate_pct", "average_discount_pct",
               "gross_sales_value", "commission_paid", "estimated_net_contribution",
               "data_quality_score", "compliance_status"]
    return to_records(eligible, columns)


def _compute_broker_concentration(brokers: pd.DataFrame) -> dict[str, Any]:
    active = brokers[brokers["reservation_count"] > 0].copy()
    total_reservations = active["reservation_count"].sum()
    active["share_of_broker_reservations_pct"] = active.apply(
        lambda r: _percentage(r["reservation_count"], total_reservations), axis=1
    )
    ranked = active.sort_values("share_of_broker_reservations_pct", ascending=False).reset_index(drop=True)
    top = ranked.iloc[0] if len(ranked) else None
    return {
        "largest_broker": top["broker_id"] if top is not None else None,
        "largest_broker_name": top["broker_name"] if top is not None else None,
        "largest_broker_share_pct": _safe_round(top["share_of_broker_reservations_pct"]) if top is not None else None,
        "top_3_broker_share_pct": _safe_round(ranked.head(3)["share_of_broker_reservations_pct"].sum()) if len(ranked) else None,
    }


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------

MATERIAL_BROKER_MIN_RESERVATIONS = 12


def _detect_broker_anomalies(broker_ranking: list[dict[str, Any]], concentration: dict[str, Any]) -> list[dict[str, Any]]:
    """Flag brokers that are simultaneously elevated on cancellation rate, discount, and volume.

    A single-metric z-score (e.g. cancellation rate alone) lets a
    small broker with a handful of reservations and noisy statistics
    dominate the ranking, while the broker that actually matters --
    high volume, high discount, AND elevated cancellations at once --
    can look unremarkable on any one axis. This composite approach
    (same technique analysis/inventory_velocity.py's peer group would
    use) sums each broker's oriented z-score across all three
    dimensions and flags outliers on the combined score, then requires
    a higher minimum reservation count so the flagged broker is a
    genuinely material, not merely noisy-small-sample, finding.
    """
    anomalies = []
    frame = pd.DataFrame([r for r in broker_ranking if r["reservation_count"] >= MATERIAL_BROKER_MIN_RESERVATIONS])
    if len(frame) >= 3:
        composite = (
            _zscores(frame["cancellation_rate_pct"].astype(float))
            + _zscores(frame["average_discount_pct"].astype(float))
            + _zscores(frame["reservation_count"].astype(float))
        )
        for idx, z_value in composite.items():
            if z_value < ANOMALY_Z_HIGH:
                continue
            row = frame.loc[idx]
            anomalies.append({
                "severity": "HIGH" if z_value >= 3.0 else "MEDIUM",
                "category": "Broker Risk",
                "title": f"{row['broker_name']} combines high volume with elevated cancellations and discounting",
                "description": (
                    f"{row['broker_name']} ({row['broker_id']}) sources {int(row['reservation_count'])} "
                    f"reservations -- among the platform's highest broker volumes -- but converts them "
                    f"with a {row['cancellation_rate_pct']:.1f}% cancellation rate and a "
                    f"{row['average_discount_pct']:.1f}% average discount, a combined risk profile "
                    f"{z_value:.1f} standard deviations above its peer brokers."
                ),
                "recommended_action": (
                    f"Review {row['broker_name']}'s deal-qualification standards and discount "
                    "authorization limits before renewing or expanding its allocation; its headline "
                    "reservation volume overstates its net contribution once cancellations and "
                    "discounting are accounted for."
                ),
                "broker_id": row["broker_id"],
            })

    if concentration.get("largest_broker_share_pct") and concentration["largest_broker_share_pct"] >= CONFIG["thresholds"]["broker_concentration_watch_pct"]:
        anomalies.append({
            "severity": "MEDIUM",
            "category": "Broker Risk",
            "title": f"Broker channel is concentrated in {concentration['largest_broker_name']}",
            "description": (
                f"{concentration['largest_broker_name']} alone sources "
                f"{concentration['largest_broker_share_pct']:.1f}% of all broker-channel reservations, "
                f"above the {CONFIG['thresholds']['broker_concentration_watch_pct']:.0f}% concentration "
                "watch threshold."
            ),
            "recommended_action": (
                "Diversify broker sourcing and strengthen the direct channel so a single broker "
                "relationship change cannot materially disrupt reservation volume."
            ),
            "broker_id": concentration.get("largest_broker"),
        })

    return sort_and_limit_anomalies(anomalies, MAX_ANOMALIES_RETURNED)


# --------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------

def _compute_health_score(
    summary: dict[str, Any], agent_ranking: list[dict[str, Any]], broker_ranking: list[dict[str, Any]], concentration: dict[str, Any]
) -> dict[str, Any]:
    direct_conversion_score = summary["direct_reservation_to_contract_pct"] or 0.0
    broker_conversion_score = summary["broker_reservation_to_contract_pct"] or 0.0

    all_discounts = [r["avg_discount_pct"] for r in agent_ranking if r["avg_discount_pct"] is not None] + \
                     [r["average_discount_pct"] for r in broker_ranking if r["average_discount_pct"] is not None]
    discount_discipline_score = 100.0
    if len(all_discounts) > 1:
        spread = max(all_discounts) - min(all_discounts)
        discount_discipline_score = max(0.0, 100 - spread * 4)

    concentration_pct = concentration.get("largest_broker_share_pct") or 0.0
    broker_concentration_score = max(0.0, 100 - concentration_pct * 1.5)

    components = {
        "direct_conversion_quality": round(direct_conversion_score, 1),
        "broker_conversion_quality": round(broker_conversion_score, 1),
        "discount_discipline": round(discount_discipline_score, 1),
        "broker_concentration": round(broker_concentration_score, 1),
    }
    overall = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)

    return {
        "overall_score": round(overall, 1),
        "components": components,
        "weights": HEALTH_SCORE_WEIGHTS,
        "methodology": (
            "Weighted blend of four 0-100 components: direct_conversion_quality (direct-channel "
            "reservation-to-contract rate, weight 0.30), broker_conversion_quality (broker-channel "
            "reservation-to-contract rate, weight 0.20), discount_discipline (100 minus a penalty "
            "for the spread between the most- and least-disciplined agent/broker's average discount, "
            "weight 0.25), and broker_concentration (100 minus a penalty for the largest broker's "
            "share of broker-sourced reservations, weight 0.25)."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_sales_team_broker_performance(sales_df: pd.DataFrame, brokers_df: pd.DataFrame) -> dict[str, Any]:
    """Run the full Sales Team & Broker Performance Intelligence suite a Sales Director/CCO needs.

    Args:
        sales_df: The sales DataFrame loaded from data/sales.csv.
        brokers_df: The brokers DataFrame loaded from data/brokers.csv
            (already carries lead_volume/reservation_count/contract_count/
            cancelled_contracts/gross_sales_value/commission_paid/
            average_discount_pct as pre-aggregated fields).

    Returns:
        A hierarchical, JSON-serializable dictionary shaped as::

            {
                "summary": {...},
                "kpis": {"broker_concentration": {...}},
                "rankings": {"agents": [...], "teams": [...], "brokers": [...]},
                "anomalies": [...],
                "health_score": {...},
                "methodology": {...},
            }

    Raises:
        ValueError: If either DataFrame is empty/None or missing
            columns this module depends on.
    """
    sales = _prepare_sales(sales_df)
    brokers = _prepare_brokers(brokers_df)

    summary = _compute_summary(sales, brokers)
    agent_ranking = _compute_agent_ranking(sales)
    team_ranking = _compute_team_ranking(sales)
    broker_ranking = _compute_broker_ranking(brokers)
    concentration = _compute_broker_concentration(brokers)
    health_score = _compute_health_score(summary, agent_ranking, broker_ranking, concentration)

    result = {
        "summary": summary,
        "kpis": {"broker_concentration": concentration},
        "rankings": {"agents": agent_ranking, "teams": team_ranking, "brokers": broker_ranking},
        "anomalies": _detect_broker_anomalies(broker_ranking, concentration),
        "health_score": health_score,
        "methodology": {
            "ranking_fairness": (
                f"Agents, teams, and brokers are ranked by reservation-to-contract rate paired with "
                "cancellation rate (a quality-adjusted view), never by raw reservation volume alone -- "
                f"entities with fewer than {MIN_RESERVATIONS_FOR_RANKING} reservations are excluded "
                "from ranking as statistically too thin to judge fairly."
            ),
            "net_contribution": (
                "estimated_net_contribution approximates each broker's gross sales value after its "
                "own average discount and commission paid, giving a truer 'what this broker actually "
                "nets the business' figure than headline gross sales value alone."
            ),
            "anomaly_detection": (
                "Broker risk anomalies use a composite z-score across cancellation rate, average "
                "discount, and reservation volume (cross-sectional, brokers with at least "
                f"{MATERIAL_BROKER_MIN_RESERVATIONS} reservations only) -- never by referencing a "
                "specific broker by name in code. Concentration risk is flagged against the fixed "
                "threshold in config/real_estate_demo.yml (thresholds.broker_concentration_watch_pct)."
            ),
            "health_score": health_score["methodology"],
        },
    }
    return _json_safe(result)
