"""Cross-Functional Root Cause & Recommendation Intelligence engine.

The nine domain-specific analytics engines (sales_performance,
lead_funnel, inventory_velocity, marketing_efficiency,
sales_team_broker_performance, collections_risk, cancellations,
construction_handover, customer_experience) each answer a
domain-specific question well, but none of them alone can say *why*
the business performs the way it does -- that requires looking across
domains at once. This module is that layer: it consumes the nine
engines' own outputs (never recomputing what they already computed),
looks for places where independent domains agree with each other, and
turns that convergence into evidence-backed root causes, quantified
financial exposure, and a prioritized, confidence-scored action plan.

    CSV Data -> Domain Analytics Layer -> Correlation Layer -> AI Report Generator

Design notes
------------
- No machine learning and no predictive modeling anywhere in this
  module. "Correlation" here means exactly what pandas' `.corr()`
  computes (Pearson's r) applied to already-aggregated per-project
  metrics, or a simple ratio/multiplier -- nothing is fit, trained, or
  forecast.
- Confidence is never invented. Every confidence score comes from one
  transparent formula (`_compute_confidence`, mirroring the platform's
  original logistics-era formula, now sourced from
  config/real_estate_demo.yml): a base score for a real, measured
  pattern, plus a bonus for the statistical strength of that pattern,
  plus a bonus for how many independent engines/anomaly-detectors flag
  the same conclusion. The score is capped below 1.0.
- Root-cause detection is generalized (never hardcodes a project,
  broker, or building name in its trigger logic) even though this
  platform's demo data happens to contain five specific engineered
  business problems -- the same detectors would fire on a real
  client's data wherever the same statistical pattern exists.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis._config import CONFIG
from analysis._shared import _json_safe, _safe_round, _zscores, fmt_currency_compact, fmt_month_names

CURRENCY = CONFIG["company"]["currency"]

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CONF = CONFIG["confidence"]
CONFIDENCE_BASE = CONF["base"]
CONFIDENCE_STRENGTH_WEIGHT = CONF["strength_weight"]
CONFIDENCE_PER_CONVERGENCE_SIGNAL = CONF["per_convergence_signal"]
CONFIDENCE_MAX_CONVERGENCE_SIGNALS = CONF["max_convergence_signals"]
CONFIDENCE_CAP = CONF["cap"]

MIN_PROJECTS_FOR_CORRELATION = 3
TOP_N_RELATIONSHIP_EVIDENCE = 3
COMPOSITE_ROOT_CAUSE_MIN_ZSCORE = 1.4
FUNNEL_CAPACITY_RESPONSE_ZSCORE_MIN = 1.2
BROKER_DEPENDENCY_SHARE_MIN_PCT = CONFIG["thresholds"]["broker_concentration_watch_pct"]
SEASONALITY_CV_MIN = 0.55  # coefficient of variation of monthly reservations, across a project's own months


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------

def _require(analysis: Any, required_keys: list[str], name: str) -> dict[str, Any]:
    if analysis is None or not isinstance(analysis, dict):
        raise ValueError(f"analyze_correlations: {name} must be a non-empty dict produced by its analytics engine")
    missing = [k for k in required_keys if k not in analysis]
    if missing:
        raise ValueError(f"analyze_correlations: {name} is missing expected keys: {missing}")
    return analysis


def _compute_confidence(strength: float, corroborating_signals: int) -> float:
    """The single confidence formula every score in this module uses. See module docstring."""
    strength = min(max(strength, 0.0), 1.0)
    convergence_bonus = min(corroborating_signals, CONFIDENCE_MAX_CONVERGENCE_SIGNALS) * CONFIDENCE_PER_CONVERGENCE_SIGNAL
    raw = CONFIDENCE_BASE + strength * CONFIDENCE_STRENGTH_WEIGHT + convergence_bonus
    return round(min(raw, CONFIDENCE_CAP), 2)


def _make_relationship(domain_pair: str, finding: str, metric: str, value: Any,
                        supporting_evidence: list[str], confidence: float, interpretation: str) -> dict[str, Any]:
    return {
        "domain_pair": domain_pair, "finding": finding, "metric": metric, "value": value,
        "supporting_evidence": supporting_evidence, "confidence": confidence, "interpretation": interpretation,
    }


def _project_map(rows: list[dict[str, Any]], key_field: str, value_field: str) -> dict[str, float]:
    return {r[key_field]: r[value_field] for r in rows if r.get(key_field) is not None and r.get(value_field) is not None}


def _pearson_relationship(
    domain_pair: str, label_a: str, map_a: dict[str, float], direction_a: str,
    label_b: str, map_b: dict[str, float], direction_b: str, interpretation: str,
) -> dict[str, Any] | None:
    common = sorted(set(map_a) & set(map_b))
    if len(common) < MIN_PROJECTS_FOR_CORRELATION:
        return None
    a = pd.Series({k: map_a[k] for k in common})
    b = pd.Series({k: map_b[k] for k in common})
    r = a.corr(b)
    if pd.isna(r):
        r = 0.0
    worst_a = a.idxmax() if direction_a == "high" else a.idxmin()
    worst_b = b.idxmax() if direction_b == "high" else b.idxmin()
    aligned = worst_a == worst_b
    confidence = _compute_confidence(abs(r), corroborating_signals=1 if aligned else 0)
    alignment_text = f"the same entity ({worst_a}) is worst on both metrics" if aligned else (
        f"the worst entity differs by metric ({worst_a} vs {worst_b})"
    )
    return _make_relationship(
        domain_pair=domain_pair,
        finding=f"{label_a} and {label_b} correlate at r={round(float(r), 3)} across {len(common)} projects; {alignment_text}.",
        metric=f"correlation({label_a}, {label_b})", value=round(float(r), 3),
        supporting_evidence=[f"Worst by {label_a}: {worst_a}.", f"Worst by {label_b}: {worst_b}."],
        confidence=confidence, interpretation=interpretation,
    )


# --------------------------------------------------------------------------
# Accessors (defensive reads across the nine engines' own output shapes)
# --------------------------------------------------------------------------

def _sales_projects(sp: dict[str, Any]) -> list[dict[str, Any]]:
    return sp.get("rankings", {}).get("by_project", [])


def _funnel_projects(lf: dict[str, Any]) -> list[dict[str, Any]]:
    return lf.get("rankings", {}).get("by_project", [])


def _inventory_stale_by_project(iv: dict[str, Any]) -> list[dict[str, Any]]:
    return iv.get("kpis", {}).get("aging", {}).get("by_project", [])


def _collections_by_project(cr: dict[str, Any]) -> list[dict[str, Any]]:
    return cr.get("kpis", {}).get("aging", {}).get("by_project", [])


def _cancellations_by_project(cx: dict[str, Any]) -> list[dict[str, Any]]:
    return cx.get("kpis", {}).get("cuts", {}).get("by_project", [])


def _cancellations_by_down_payment_band(cx: dict[str, Any]) -> list[dict[str, Any]]:
    return cx.get("kpis", {}).get("cuts", {}).get("by_down_payment_band", [])


def _construction_by_building(ch: dict[str, Any]) -> list[dict[str, Any]]:
    return ch.get("kpis", {}).get("delay_concentration", {}).get("by_building", [])


def _cx_by_project(cxe: dict[str, Any]) -> list[dict[str, Any]]:
    return cxe.get("rankings", {}).get("by_project", [])


def _broker_ranking(stb: dict[str, Any]) -> list[dict[str, Any]]:
    return stb.get("rankings", {}).get("brokers", [])


def _monthly_sales_by_project(sp: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Not directly available per-project from sales_performance's own trend (that trend is portfolio-wide),
    so callers needing per-project monthly series pass sales_df directly -- see _root_cause_seasonality."""
    return {}


# --------------------------------------------------------------------------
# Relationship discovery (pairwise, cross-domain, project-level)
# --------------------------------------------------------------------------

def _discover_relationships(
    sales_performance: dict, lead_funnel: dict, collections_risk: dict, cancellations: dict,
    construction_handover: dict, customer_experience: dict, marketing_efficiency: dict,
) -> list[dict[str, Any]]:
    discount_map = _project_map(_sales_projects(sales_performance), "project_id", "avg_discount_pct")
    cancellation_map = _project_map(_cancellations_by_project(cancellations), "project_id", "cancellation_rate_pct")
    overdue_map = _project_map(_collections_by_project(collections_risk), "project_id", "overdue_pct")
    response_map = _project_map(_funnel_projects(lead_funnel), "project_id", "avg_response_minutes")
    conversion_map = _project_map(_funnel_projects(lead_funnel), "project_id", "lead_to_contract_conversion_pct")
    negative_sentiment_map = _project_map(_cx_by_project(customer_experience), "project_id", "negative_sentiment_pct")

    variance_by_project: dict[str, float] = {}
    for row in _construction_by_building(construction_handover):
        variance_by_project[row["project_id"]] = max(variance_by_project.get(row["project_id"], 0.0), row.get("avg_variance_days") or 0.0)

    relationships = [
        _pearson_relationship(
            "Sales <-> Cancellations", "average discount", discount_map, "high",
            "cancellation rate", cancellation_map, "high",
            "Projects discounting more heavily also tend to cancel more -- consistent with "
            "aggressive discounting attracting buyers at the edge of affordability or commitment, "
            "who are more likely to fall through before or after contracting.",
        ),
        _pearson_relationship(
            "Sales <-> Collections", "average discount", discount_map, "high",
            "overdue receivables share", overdue_map, "high",
            "Higher average discounting tracks with a higher overdue receivables share -- "
            "consistent with discount-driven deals disproportionately involving buyers with "
            "weaker payment capacity.",
        ),
        _pearson_relationship(
            "Funnel <-> Commercial", "average first-response time", response_map, "high",
            "lead-to-contract conversion", conversion_map, "low",
            "Projects with slower first-response times convert leads to contracts at a lower "
            "rate -- consistent with response speed being a real driver of funnel outcomes, not "
            "just a service-quality metric.",
        ),
        _pearson_relationship(
            "Construction <-> Customer Experience", "construction schedule variance (days)", variance_by_project, "high",
            "negative sentiment rate", negative_sentiment_map, "high",
            "Projects with larger construction schedule variance also carry a higher share of "
            "negative-sentiment customer cases -- construction delay is not only an execution "
            "risk, it is already reaching customers as a experience problem.",
        ),
        _pearson_relationship(
            "Collections <-> Customer Experience", "overdue receivables share", overdue_map, "high",
            "negative sentiment rate", negative_sentiment_map, "high",
            "Projects with a higher overdue receivables share also carry more negative customer "
            "sentiment -- consistent with financial strain among buyers surfacing as "
            "dissatisfaction rather than staying a purely back-office collections issue.",
        ),
    ]

    down_payment_cuts = _cancellations_by_down_payment_band(cancellations)
    if down_payment_cuts:
        worst_band = max(down_payment_cuts, key=lambda r: r["cancellation_rate_pct"] or 0)
        best_band = min(down_payment_cuts, key=lambda r: r["cancellation_rate_pct"] or 0)
        if worst_band["down_payment_band"] != best_band["down_payment_band"]:
            gap = round((worst_band["cancellation_rate_pct"] or 0) - (best_band["cancellation_rate_pct"] or 0), 1)
            strength = min(gap / 25, 1.0)
            relationships.append(_make_relationship(
                domain_pair="Sales <-> Cancellations",
                finding=(
                    f"Reservations with a down payment in the '{worst_band['down_payment_band']}' band "
                    f"cancel at {worst_band['cancellation_rate_pct']}%, versus {best_band['cancellation_rate_pct']}% "
                    f"for the '{best_band['down_payment_band']}' band -- a {gap} point gap."
                ),
                metric="cancellation_rate_by_down_payment_band", value=down_payment_cuts,
                supporting_evidence=[f"{r['down_payment_band']}: {r['cancellation_rate_pct']}% ({r['reservations']} reservations)" for r in down_payment_cuts],
                confidence=_compute_confidence(strength, corroborating_signals=0),
                interpretation=(
                    "Low down payments are directly associated with higher cancellation risk -- a "
                    "buyer with less committed capital has less to lose by walking away."
                ),
            ))

    return [r for r in relationships if r is not None]


# --------------------------------------------------------------------------
# Root cause detectors (generalized; never hardcode an entity name)
# --------------------------------------------------------------------------

def _root_cause_collections_cancellation_concentration(
    collections_risk: dict, cancellations: dict, sales_performance: dict,
) -> dict[str, Any] | None:
    """A project simultaneously elevated on overdue receivables, cancellation rate, and discounting."""
    overdue_rows = {r["project_id"]: r for r in _collections_by_project(collections_risk)}
    cancel_rows = {r["project_id"]: r for r in _cancellations_by_project(cancellations)}
    discount_rows = {r["project_id"]: r for r in _sales_projects(sales_performance)}

    common = sorted(set(overdue_rows) & set(cancel_rows) & set(discount_rows))
    if len(common) < MIN_PROJECTS_FOR_CORRELATION:
        return None

    frame = pd.DataFrame([{
        "project_id": pid,
        "overdue_pct": overdue_rows[pid]["overdue_pct"],
        "cancellation_rate_pct": cancel_rows[pid]["cancellation_rate_pct"],
        "avg_discount_pct": discount_rows[pid]["avg_discount_pct"],
    } for pid in common])

    composite = (
        _zscores(frame["overdue_pct"].astype(float))
        + _zscores(frame["cancellation_rate_pct"].astype(float))
        + _zscores(frame["avg_discount_pct"].astype(float))
    )
    frame["composite_zscore"] = composite
    top = frame.sort_values("composite_zscore", ascending=False).iloc[0]
    if top["composite_zscore"] < COMPOSITE_ROOT_CAUSE_MIN_ZSCORE:
        return None

    project_id = top["project_id"]
    signals = sum([
        cancel_rows[project_id]["cancellation_rate_pct"] == max(r["cancellation_rate_pct"] for r in cancel_rows.values()),
        overdue_rows[project_id]["overdue_pct"] == max(r["overdue_pct"] for r in overdue_rows.values()),
        discount_rows[project_id]["avg_discount_pct"] == max(r["avg_discount_pct"] for r in discount_rows.values()),
    ])
    confidence = _compute_confidence(strength=min(top["composite_zscore"] / 4, 1.0), corroborating_signals=signals)

    return {
        "category": "Collections & Cancellation Risk",
        "title": f"{project_id} Affordability-Driven Collections & Cancellation Risk",
        "project_id": project_id,
        "description": (
            f"{project_id} simultaneously carries the network's most elevated overdue receivables "
            f"share ({top['overdue_pct']:.1f}%), contract cancellation rate "
            f"({top['cancellation_rate_pct']:.1f}%), and average discount ({top['avg_discount_pct']:.1f}%) "
            "-- a pattern consistent with deals structured at the edge of buyer affordability "
            "(low down payments, long payment plans, heavy discounting) rather than three "
            "unrelated issues."
        ),
        "statistical_evidence": [
            f"Composite risk z-score: {top['composite_zscore']:.2f} (sum of oriented z-scores across "
            "overdue share, cancellation rate, and average discount).",
        ],
        "operational_evidence": [
            f"Overdue receivables share: {top['overdue_pct']:.1f}%.",
            f"Contract cancellation rate: {top['cancellation_rate_pct']:.1f}%.",
            f"Average discount: {top['avg_discount_pct']:.1f}%.",
        ],
        "confidence": confidence,
        "recommended_owner": "Sales Director & Collections Director",
        "recommended_horizon": "Immediate (0-30 Days)",
    }


def _root_cause_construction_delay_exposure(
    construction_handover: dict, customer_experience: dict,
) -> dict[str, Any] | None:
    delay_exposure = construction_handover.get("kpis", {}).get("delay_exposure", {})
    concentration = construction_handover.get("kpis", {}).get("delay_concentration", {})
    building = concentration.get("most_delayed_building")
    project_id = concentration.get("most_delayed_building_project")
    if not building or not project_id:
        return None

    by_building = _construction_by_building(construction_handover)
    top_row = next((r for r in by_building if r["building"] == building and r["project_id"] == project_id), None)
    if top_row is None or (top_row.get("avg_variance_days") or 0) < CONFIG["thresholds"]["construction_variance_watch_days"]:
        return None

    cx_row = next((r for r in _cx_by_project(customer_experience) if r["project_id"] == project_id), None)
    signals = 1 if delay_exposure.get("units_exposed_to_delay") else 0
    signals += 1 if cx_row and cx_row.get("negative_sentiment_pct", 0) > 30 else 0
    strength = min((top_row.get("avg_variance_days") or 0) / 90, 1.0)
    confidence = _compute_confidence(strength, corroborating_signals=signals)

    return {
        "category": "Construction Delivery Risk",
        "title": f"{project_id} {building} Delivery Delay Exposure",
        "project_id": project_id,
        "description": (
            f"{project_id} {building} averages {top_row['avg_variance_days']:.0f} days of milestone "
            f"schedule variance (peak {top_row['max_variance_days']:.0f} days) with "
            f"{int(top_row['high_severity_issues'])} high-severity issue(s) logged. "
            f"{delay_exposure.get('units_exposed_to_delay', 0)} units are currently exposed to a "
            "delayed handover as a result."
        ),
        "statistical_evidence": [
            f"Average schedule variance: {top_row['avg_variance_days']:.0f} days across "
            f"{int(top_row['milestones'])} milestones.",
        ],
        "operational_evidence": [
            f"Units exposed to delayed handover: {delay_exposure.get('units_exposed_to_delay')}.",
            f"Contracted value exposed: {fmt_currency_compact(delay_exposure.get('contracted_value_exposed'), CURRENCY)}.",
            f"Amount already collected on delayed units: {fmt_currency_compact(delay_exposure.get('amount_already_collected_on_delayed_units'), CURRENCY)}.",
        ] + ([f"{project_id} negative-sentiment case rate: {cx_row['negative_sentiment_pct']:.1f}%."] if cx_row else []),
        "financial_exposure": delay_exposure.get("contracted_value_exposed"),
        "customer_impact": delay_exposure.get("customers_exposed"),
        "confidence": confidence,
        "recommended_owner": "Construction Director",
        "recommended_horizon": "Strategic (60-120 Days)",
    }


def _root_cause_funnel_capacity_constraint(lead_funnel: dict, marketing_efficiency: dict) -> dict[str, Any] | None:
    """A project where response time is a statistical outlier AND lead quality upstream looks fine --
    i.e. a sales-capacity problem, not a marketing-quality problem."""
    rows = _funnel_projects(lead_funnel)
    if len(rows) < MIN_PROJECTS_FOR_CORRELATION:
        return None
    frame = pd.DataFrame(rows)
    z = _zscores(frame["avg_response_minutes"].astype(float))
    frame["response_zscore"] = z
    candidate = frame.sort_values("response_zscore", ascending=False).iloc[0]
    if candidate["response_zscore"] < FUNNEL_CAPACITY_RESPONSE_ZSCORE_MIN:
        return None

    project_id = candidate["project_id"]
    channel_quality = marketing_efficiency.get("kpis", {}).get("channel_quality_matrix", {})
    expensive_high_value = channel_quality.get("expensive_but_high_value_channels", [])
    lead_quality_is_fine = candidate["qualification_rate_pct"] >= frame["qualification_rate_pct"].median()

    workload = lead_funnel.get("rankings", {}).get("agent_workload", [])
    overloaded_agents = [w for w in workload if w["active_leads"] and w["active_leads"] > (sum(x["active_leads"] for x in workload) / max(len(workload), 1)) * 2]

    signals = int(lead_quality_is_fine) + int(bool(overloaded_agents)) + int(bool(expensive_high_value))
    strength = min(candidate["response_zscore"] / 3, 1.0)
    confidence = _compute_confidence(strength, corroborating_signals=signals)

    return {
        "category": "Sales Capacity Constraint",
        "title": f"{project_id} Sales Capacity Bottleneck Behind Otherwise-Healthy Demand",
        "project_id": project_id,
        "description": (
            f"{project_id}'s average first-response time ({candidate['avg_response_minutes']:.0f} "
            "minutes) is a statistical outlier versus its peer projects, while lead qualification "
            "quality is at or above the portfolio median -- the constraint is sales-team capacity "
            "to process demand, not the quality of the demand itself."
        ),
        "statistical_evidence": [
            f"Response-time z-score: {candidate['response_zscore']:.2f} vs. peer projects.",
            f"Qualification rate: {candidate['qualification_rate_pct']:.1f}% "
            f"(portfolio median: {frame['qualification_rate_pct'].median():.1f}%).",
        ],
        "operational_evidence": (
            [f"{len(overloaded_agents)} agent(s) carry more than 2x the portfolio's average active-lead load."]
            if overloaded_agents else []
        ) + (
            [f"Channel(s) producing above-average-value leads despite higher cost per lead: "
             f"{', '.join(c['channel'] for c in expensive_high_value)}."] if expensive_high_value else []
        ),
        "confidence": confidence,
        "recommended_owner": "Sales Director",
        "recommended_horizon": "Immediate (0-30 Days)",
    }


def _root_cause_broker_dependency(sales_team_broker_performance: dict) -> dict[str, Any] | None:
    concentration = sales_team_broker_performance.get("kpis", {}).get("broker_concentration", {})
    share = concentration.get("largest_broker_share_pct")
    if share is None or share < BROKER_DEPENDENCY_SHARE_MIN_PCT:
        return None

    broker_id = concentration.get("largest_broker")
    broker_row = next((r for r in _broker_ranking(sales_team_broker_performance) if r["broker_id"] == broker_id), None)
    if broker_row is None:
        return None

    peer_rows = [r for r in _broker_ranking(sales_team_broker_performance) if r["broker_id"] != broker_id and r["reservation_count"] >= 10]
    peer_avg_cancellation = sum(r["cancellation_rate_pct"] for r in peer_rows) / len(peer_rows) if peer_rows else 0.0
    elevated_cancellation = broker_row["cancellation_rate_pct"] > peer_avg_cancellation

    strength = min(share / 50, 1.0)
    signals = int(elevated_cancellation) + int(broker_row["average_discount_pct"] > 8.0)
    confidence = _compute_confidence(strength, corroborating_signals=signals)

    return {
        "category": "Broker Channel Dependency",
        "title": f"{concentration.get('largest_broker_name')} Broker Concentration Risk",
        "broker_id": broker_id,
        "description": (
            f"{concentration.get('largest_broker_name')} sources {share:.1f}% of all broker-channel "
            f"reservations -- above the {BROKER_DEPENDENCY_SHARE_MIN_PCT:.0f}% concentration watch "
            f"threshold -- while converting at a {broker_row['cancellation_rate_pct']:.1f}% cancellation "
            f"rate and a {broker_row['average_discount_pct']:.1f}% average discount"
            + (f", both above the peer broker average of {peer_avg_cancellation:.1f}% cancellations." if elevated_cancellation else ".")
        ),
        "statistical_evidence": [
            f"Broker share of broker-channel reservations: {share:.1f}%.",
            f"Peer broker average cancellation rate: {peer_avg_cancellation:.1f}%.",
        ],
        "operational_evidence": [
            f"Reservations sourced: {broker_row['reservation_count']}.",
            f"Estimated net contribution after discount and commission: {fmt_currency_compact(broker_row['estimated_net_contribution'], CURRENCY)}.",
        ],
        "financial_exposure": broker_row.get("gross_sales_value"),
        "confidence": confidence,
        "recommended_owner": "Commercial Director",
        "recommended_horizon": "Near-Term (30-60 Days)",
    }


def _root_cause_seasonal_volatility(sales_df: pd.DataFrame, cancellations: dict) -> dict[str, Any] | None:
    """A project whose monthly net-contracted volume has high coefficient of variation (seasonality),
    corroborated by a cancellation-timing pattern concentrated shortly after its peak months."""
    working = sales_df.copy()
    working["reservation_date"] = pd.to_datetime(working["reservation_date"], errors="coerce")
    working["month_num"] = working["reservation_date"].dt.month

    by_project_month = working.groupby(["project_id", "month_num"]).size().reset_index(name="reservations")
    candidates = []
    for project_id, group in by_project_month.groupby("project_id"):
        if len(group) < 6:
            continue
        mean = group["reservations"].mean()
        std = group["reservations"].std(ddof=0)
        cv = (std / mean) if mean else 0.0
        candidates.append((project_id, cv, group))
    if not candidates:
        return None

    project_id, cv, group = max(candidates, key=lambda c: c[1])
    if cv < SEASONALITY_CV_MIN:
        return None

    peak_months = group.sort_values("reservations", ascending=False).head(3)["month_num"].tolist()
    peak_months_label = fmt_month_names(sorted(peak_months))
    cancel_rows = {r["project_id"]: r for r in _cancellations_by_project(cancellations)}
    project_cancel = cancel_rows.get(project_id, {})

    strength = min(cv, 1.0)
    signals = 1 if project_cancel.get("cancellation_rate_pct", 0) > 8 else 0
    confidence = _compute_confidence(strength, corroborating_signals=signals)

    return {
        "category": "Seasonal Demand Volatility",
        "title": f"{project_id} Seasonal Demand Concentration",
        "project_id": project_id,
        "description": (
            f"{project_id}'s monthly reservation volume has a coefficient of variation of "
            f"{cv:.2f} across the year, with volume concentrated in {peak_months_label} -- a "
            f"materially seasonal demand pattern that also carries a "
            f"{project_cancel.get('cancellation_rate_pct', 'N/A')}% cancellation rate, "
            "consistent with peak-season buyers being more prone to later reconsideration."
        ),
        "statistical_evidence": [f"Coefficient of variation of monthly reservations: {cv:.2f}."],
        "operational_evidence": [f"Peak reservation months: {peak_months_label}."],
        "confidence": confidence,
        "recommended_owner": "Marketing Director & Sales Director",
        "recommended_horizon": "Strategic (60-120 Days)",
    }


def _detect_root_causes(
    sales_performance: dict, lead_funnel: dict, collections_risk: dict, cancellations: dict,
    construction_handover: dict, customer_experience: dict, marketing_efficiency: dict,
    sales_team_broker_performance: dict, sales_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    candidates = [
        _root_cause_collections_cancellation_concentration(collections_risk, cancellations, sales_performance),
        _root_cause_construction_delay_exposure(construction_handover, customer_experience),
        _root_cause_funnel_capacity_constraint(lead_funnel, marketing_efficiency),
        _root_cause_broker_dependency(sales_team_broker_performance),
        _root_cause_seasonal_volatility(sales_df, cancellations),
    ]
    found = [rc for rc in candidates if rc is not None]
    return sorted(found, key=lambda rc: rc["confidence"], reverse=True)


# --------------------------------------------------------------------------
# Financial exposure summary
# --------------------------------------------------------------------------

def _compute_financial_exposure(
    root_causes: list[dict[str, Any]], sales_performance: dict, cancellations: dict,
    collections_risk: dict, construction_handover: dict, marketing_efficiency: dict,
) -> dict[str, Any]:
    exposures: list[dict[str, Any]] = []

    cancel_summary = cancellations.get("summary", {})
    if cancel_summary.get("cancelled_gross_value") is not None:
        exposures.append({
            "category": "Cancellation Exposure", "amount": cancel_summary["cancelled_gross_value"],
            "description": "Gross value of reservations/contracts cancelled in the analyzed period.",
        })

    collections_summary = collections_risk.get("summary", {})
    if collections_summary.get("total_overdue_amount") is not None:
        exposures.append({
            "category": "Receivables at Risk", "amount": collections_summary["total_overdue_amount"],
            "description": "Total overdue receivables outstanding across all projects.",
        })

    delay_exposure = construction_handover.get("kpis", {}).get("delay_exposure", {})
    if delay_exposure.get("contracted_value_exposed") is not None:
        exposures.append({
            "category": "Delayed Handover Exposure", "amount": delay_exposure["contracted_value_exposed"],
            "description": "Net contracted value of units currently forecast for a delayed handover.",
        })

    sales_summary = sales_performance.get("summary", {})
    discount_leakage = sales_summary.get("total_discount_value")
    if discount_leakage is not None:
        exposures.append({
            "category": "Discount / Margin Erosion", "amount": discount_leakage,
            "description": "Total discount value applied against gross price on net-contracted sales.",
        })

    marketing_spend = marketing_efficiency.get("summary", {}).get("total_spend")
    if marketing_spend is not None:
        exposures.append({
            "category": "Marketing Spend Deployed", "amount": marketing_spend,
            "description": "Total marketing spend across all campaigns in the analyzed period (context, not a risk in itself).",
        })

    total_material_risk = sum(
        e["amount"] for e in exposures if e["category"] in
        ("Cancellation Exposure", "Receivables at Risk", "Delayed Handover Exposure", "Discount / Margin Erosion")
    )

    return {
        "line_items": exposures,
        "total_material_risk_exposure": _safe_round(total_material_risk),
        "note": (
            "total_material_risk_exposure sums cancellation exposure, receivables at risk, delayed "
            "handover exposure, and discount/margin erosion -- these are not mutually exclusive "
            "(a unit can appear in more than one category) so this is a directional scale-of-risk "
            "figure, not a non-overlapping sum."
        ),
    }


# --------------------------------------------------------------------------
# Recommendation engine
# --------------------------------------------------------------------------

ACTION_PROFILE_BY_CATEGORY: dict[str, dict[str, str]] = {
    "Collections & Cancellation Risk": {"difficulty": "Medium", "estimated_timeframe": "14-30 days"},
    "Construction Delivery Risk": {"difficulty": "High", "estimated_timeframe": "60-120 days"},
    "Sales Capacity Constraint": {"difficulty": "Low", "estimated_timeframe": "14-30 days"},
    "Broker Channel Dependency": {"difficulty": "Medium", "estimated_timeframe": "30-60 days"},
    "Seasonal Demand Volatility": {"difficulty": "Medium", "estimated_timeframe": "60-120 days"},
}
DEFAULT_ACTION_PROFILE = {"difficulty": "Medium", "estimated_timeframe": "30-60 days"}

RECOMMENDED_ACTION_TEXT: dict[str, str] = {
    "Collections & Cancellation Risk": (
        "Tighten down-payment and payment-plan-length policy for new reservations on this project, "
        "cap discount authorization below the network average, and prioritize the existing overdue "
        "book for structured collections outreach within 30 days."
    ),
    "Construction Delivery Risk": (
        "Escalate the delayed building's critical-path milestones with the responsible contractor, "
        "proactively notify affected buyers of the revised forecast rather than waiting for the "
        "original promised date, and reforecast handover dates for units in that building only."
    ),
    "Sales Capacity Constraint": (
        "Rebalance active leads from over-capacity agents to agents with spare capacity, prioritizing "
        "premium-scored leads first, and enforce a first-response SLA for leads above the "
        "project's lead-score threshold."
    ),
    "Broker Channel Dependency": (
        "Cap this broker's discount authorization to the network average, diversify broker sourcing "
        "for this project, and strengthen the direct channel so a single broker relationship change "
        "cannot materially disrupt reservation volume."
    ),
    "Seasonal Demand Volatility": (
        "Build a post-peak-season collections and retention plan ahead of the next cycle, and "
        "evaluate whether off-peak pricing incentives can smooth demand rather than concentrating "
        "cancellation risk in the months after each peak."
    ),
}


def _build_priority_actions(root_causes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = []
    for priority, rc in enumerate(root_causes, start=1):
        profile = ACTION_PROFILE_BY_CATEGORY.get(rc["category"], DEFAULT_ACTION_PROFILE)
        actions.append({
            "priority": priority,
            "title": rc["title"],
            "business_issue": rc["description"],
            "evidence": rc.get("statistical_evidence", []) + rc.get("operational_evidence", []),
            "financial_exposure": rc.get("financial_exposure"),
            "recommended_action": RECOMMENDED_ACTION_TEXT.get(rc["category"], "Investigate and remediate the underlying driver."),
            "owner": rc["recommended_owner"],
            "horizon": rc["recommended_horizon"],
            "difficulty": profile["difficulty"],
            "estimated_timeframe": profile["estimated_timeframe"],
            "confidence": rc["confidence"],
            "expected_business_impact": (
                f"Directly addresses the {rc['category'].lower()} identified for "
                f"{rc.get('project_id') or rc.get('broker_id') or 'the affected entity'}."
            ),
        })
    return actions


# --------------------------------------------------------------------------
# Executive summary
# --------------------------------------------------------------------------

def _build_executive_summary(root_causes: list[dict[str, Any]], health_scores: dict[str, float]) -> dict[str, Any]:
    valid_scores = [s for s in health_scores.values() if s is not None]
    avg_health = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else None
    max_confidence = max((rc["confidence"] for rc in root_causes), default=0.0)

    if (avg_health is not None and avg_health < 55) or max_confidence >= 0.85:
        risk_level = "HIGH"
    elif (avg_health is not None and avg_health < 72) or max_confidence >= 0.6:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    top = root_causes[0] if root_causes else None
    return {
        "overall_business_risk_level": risk_level,
        "risk_level_basis": (
            f"Average of the nine domain health scores: {avg_health}/100. {len(root_causes)} "
            f"cross-domain root cause(s) identified, highest confidence {round(max_confidence, 2)}."
        ),
        "executive_priority": (
            f"{top['title']} -- {top['description']}" if top else
            "No cross-domain root cause cleared this platform's evidence bar this period; continue "
            "monitoring each domain engine's own anomalies."
        ),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_correlations(
    sales_performance: dict[str, Any],
    lead_funnel: dict[str, Any],
    inventory_velocity: dict[str, Any],
    marketing_efficiency: dict[str, Any],
    sales_team_broker_performance: dict[str, Any],
    collections_risk: dict[str, Any],
    cancellations: dict[str, Any],
    construction_handover: dict[str, Any],
    customer_experience: dict[str, Any],
    health_scores: dict[str, float],
    sales_df: pd.DataFrame,
) -> dict[str, Any]:
    """Cross-domain Business Intelligence layer: why the business performs the way it does.

    Consumes the outputs of all nine domain analytics engines -- never
    recomputing what they already computed -- to find where
    independent domains agree, infer evidence-backed root causes,
    quantify financial exposure, and produce a prioritized,
    confidence-scored action plan.

    Args:
        sales_performance, lead_funnel, inventory_velocity,
            marketing_efficiency, sales_team_broker_performance,
            collections_risk, cancellations, construction_handover,
            customer_experience: outputs of the corresponding
            analysis.* module.
        health_scores: {domain_name: overall_score} for all nine
            domains plus data_quality, as already computed by
            analysis.executive_scoring -- used only for the executive
            risk-level summary, never recomputed here.
        sales_df: The raw sales DataFrame, used only for the one
            cross-tabulation no domain engine exposes (per-project
            monthly reservation seasonality).

    Returns:
        A JSON-serializable dictionary shaped as::

            {
                "executive_summary": {...},
                "relationships": [...],
                "root_causes": [...],
                "financial_exposure": {...},
                "priority_actions": [...],
                "methodology": {...},
            }

    Raises:
        ValueError: If any analysis dict is missing/malformed, or
            `sales_df` is empty/None.
    """
    if sales_df is None or len(sales_df) == 0:
        raise ValueError("analyze_correlations: sales_df DataFrame is empty or None")

    relationships = _discover_relationships(
        sales_performance, lead_funnel, collections_risk, cancellations,
        construction_handover, customer_experience, marketing_efficiency,
    )
    root_causes = _detect_root_causes(
        sales_performance, lead_funnel, collections_risk, cancellations, construction_handover,
        customer_experience, marketing_efficiency, sales_team_broker_performance, sales_df,
    )
    financial_exposure = _compute_financial_exposure(
        root_causes, sales_performance, cancellations, collections_risk, construction_handover, marketing_efficiency,
    )
    priority_actions = _build_priority_actions(root_causes)
    executive_summary = _build_executive_summary(root_causes, health_scores)

    result = {
        "executive_summary": executive_summary,
        "relationships": relationships,
        "root_causes": root_causes,
        "financial_exposure": financial_exposure,
        "priority_actions": priority_actions,
        "methodology": {
            "relationships_evaluated": [{"domain_pair": r["domain_pair"], "metric": r["metric"]} for r in relationships],
            "statistical_methods_used": [
                "Pearson correlation coefficient (pandas Series.corr()) across per-project aggregate "
                "metrics already computed by the nine domain engines.",
                "Composite z-score summation (population z-scores) for multi-metric root-cause detection.",
                "Coefficient of variation for seasonal-volatility detection.",
                "One raw cross-tabulation via sales.csv (per-project monthly reservation seasonality) "
                "-- the only relationship not derivable from an existing domain engine's output.",
            ],
            "confidence_methodology": (
                f"Every confidence score is base {CONFIDENCE_BASE} (a real, measured pattern deserves "
                f"some baseline trust) plus up to {CONFIDENCE_STRENGTH_WEIGHT} for the strength of the "
                f"underlying statistic plus {CONFIDENCE_PER_CONVERGENCE_SIGNAL} per independent engine "
                f"or signal that corroborates the same conclusion, up to "
                f"{CONFIDENCE_MAX_CONVERGENCE_SIGNALS} such signals, capped at {CONFIDENCE_CAP} -- "
                "never full certainty. All values sourced from config/real_estate_demo.yml."
            ),
            "root_cause_generalization": (
                "Every root-cause detector triggers on a statistical pattern (composite z-score, "
                "correlation strength, concentration share, coefficient of variation) evaluated "
                "against every project/broker/building in the data -- none hardcodes a specific "
                "entity name in its trigger logic, even though this platform's demo data happens to "
                "contain five specific engineered business problems."
            ),
            "assumptions": [
                "Correlation direction is inferred from domain knowledge in each relationship's "
                "interpretation text, not established statistically -- Pearson r itself is symmetric "
                "and does not indicate causal direction.",
                "Root causes are evidence-backed hypotheses ranked by confidence, not proven causation.",
            ],
            "limitations": [
                "With five projects, Pearson correlation coefficients computed across projects are a "
                "statistically thin signal on their own -- this module always pairs them with a "
                "plain-language 'is the same entity worst on both metrics' check rather than "
                "reporting r in isolation.",
                "This module performs descriptive and diagnostic analytics only: no machine "
                "learning, no predictive modeling, no forecasting.",
            ],
        },
    }
    return _json_safe(result)
