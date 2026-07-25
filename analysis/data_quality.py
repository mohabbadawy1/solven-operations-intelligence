"""Data Quality Intelligence engine.

An executive report that cites specific numbers is only as credible
as the data underneath it. This module runs a fixed battery of
integrity checks across the platform's relational dataset -- missing
keys, duplicate leads, invalid date sequences, orphaned payments,
double-booked units, negative monetary values, missing attribution --
and produces a single 0-100 Data Quality Score plus a named list of
findings, so the executive report can honestly disclose whether any
specific finding might be weakened by an underlying data issue rather
than silently assuming the data is perfect.

This module never repairs data; it only measures and reports.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis._shared import _json_safe

MAX_EXAMPLES_PER_CHECK = 5

# --- Data Quality Score ---------------------------------------------------
# Each check below contributes a penalty proportional to the affected
# row share, capped so no single check can single-handedly zero the
# score (a data quality issue is a disclosure, not evidence the whole
# platform is untrustworthy).
CHECK_WEIGHT = 100.0 / 12  # 12 checks, evenly weighted
MAX_PENALTY_PER_CHECK = CHECK_WEIGHT


def _check(name: str, description: str, affected_count: int, total_count: int, examples: list[Any], severity: str) -> dict[str, Any]:
    affected_pct = round(100 * affected_count / total_count, 2) if total_count else 0.0
    penalty = min(MAX_PENALTY_PER_CHECK, affected_pct * MAX_PENALTY_PER_CHECK / 10) if affected_count else 0.0
    return {
        "check": name,
        "description": description,
        "affected_count": int(affected_count),
        "affected_pct": affected_pct,
        "severity": severity if affected_count else "PASS",
        "examples": examples[:MAX_EXAMPLES_PER_CHECK],
        "score_penalty": round(penalty, 2),
    }


def analyze_data_quality(datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Run the full Data Quality battery across the platform's relational dataset.

    Args:
        datasets: A dict of {dataset_name: DataFrame} for every CSV the
            platform loads (projects, units, customers, brokers,
            sales_agents, campaigns, leads, sales, payment_schedules,
            collections, construction_milestones, customer_cases,
            handovers, snagging). Missing keys are treated as "check
            not applicable" rather than an error, so this module
            degrades gracefully if called with a subset.

    Returns:
        A JSON-serializable dictionary shaped as::

            {
                "data_quality_score": float,
                "status": "Healthy" | "Watch" | "At Risk" | "Critical",
                "checks": [...],
                "any_findings_may_be_weakened": bool,
                "methodology": {...},
            }
    """
    checks: list[dict[str, Any]] = []

    sales = datasets.get("sales")
    leads = datasets.get("leads")
    units = datasets.get("units")
    payment_schedules = datasets.get("payment_schedules")
    customers = datasets.get("customers")
    campaigns = datasets.get("campaigns")
    construction = datasets.get("construction_milestones")

    if sales is not None:
        missing_project = sales[sales["project_id"].isna()]
        checks.append(_check(
            "missing_project_id_on_sales", "Sales rows with no project_id attribution",
            len(missing_project), len(sales), missing_project["sale_id"].tolist(), "HIGH",
        ))

        bad_sequence = sales[
            sales["contract_date"].notna() & sales["reservation_date"].notna()
            & (pd.to_datetime(sales["contract_date"], errors="coerce") < pd.to_datetime(sales["reservation_date"], errors="coerce"))
        ]
        checks.append(_check(
            "contract_before_reservation", "Sales where contract_date precedes reservation_date",
            len(bad_sequence), len(sales), bad_sequence["sale_id"].tolist(), "HIGH",
        ))

        negative_values = sales[(sales["net_sales_value"] < 0) | (sales["gross_price"] < 0)]
        checks.append(_check(
            "negative_monetary_values_sales", "Sales rows with a negative gross_price or net_sales_value",
            len(negative_values), len(sales), negative_values["sale_id"].tolist(), "HIGH",
        ))

        duplicate_active_units = sales[
            (sales["contract_date"].notna()) & (~sales["cancellation_flag"])
        ].groupby("unit_id").filter(lambda g: len(g) > 1)
        affected_units = sorted(duplicate_active_units["unit_id"].unique().tolist())
        checks.append(_check(
            "units_with_multiple_active_contracts", "Units assigned to more than one non-cancelled contract",
            len(affected_units), int(sales["unit_id"].nunique()), affected_units, "HIGH",
        ))

        missing_broker_on_broker_channel = sales[(sales["sales_channel"] == "Broker") & (sales["broker_id"].isna())]
        checks.append(_check(
            "missing_broker_attribution", "Sales marked sales_channel='Broker' with no broker_id",
            len(missing_broker_on_broker_channel), len(sales), missing_broker_on_broker_channel["sale_id"].tolist(), "MEDIUM",
        ))

        if campaigns is not None:
            valid_campaigns = set(campaigns["campaign_id"])
            sales_with_campaign = sales[sales["campaign_id"].notna()]
            missing_campaign_attribution = sales_with_campaign[~sales_with_campaign["campaign_id"].isin(valid_campaigns)]
            checks.append(_check(
                "invalid_campaign_attribution", "Sales referencing a campaign_id not present in campaigns.csv",
                len(missing_campaign_attribution), len(sales_with_campaign) or 1,
                missing_campaign_attribution["sale_id"].tolist(), "MEDIUM",
            ))

    if leads is not None:
        duplicate_leads = leads[leads["duplicate_flag"] == True]  # noqa: E712
        checks.append(_check(
            "duplicate_leads_flagged", "Leads explicitly flagged as duplicates by the CRM",
            len(duplicate_leads), len(leads), duplicate_leads["lead_id"].tolist(), "LOW",
        ))

        valid_stages = {"new", "contacted", "qualified", "appointment_booked", "site_visit_completed",
                         "negotiation", "reservation", "contract", "lost", "dormant"}
        invalid_stage = leads[~leads["current_stage"].isin(valid_stages)]
        checks.append(_check(
            "invalid_funnel_stage", "Leads with a current_stage outside the platform's known funnel vocabulary",
            len(invalid_stage), len(leads), invalid_stage["lead_id"].tolist(), "HIGH",
        ))

        stale_records = leads[
            pd.to_datetime(leads["created_at"], errors="coerce")
            > pd.to_datetime(leads["last_activity_at"], errors="coerce")
        ]
        checks.append(_check(
            "last_activity_before_creation", "Leads where last_activity_at precedes created_at",
            len(stale_records), len(leads), stale_records["lead_id"].tolist(), "MEDIUM",
        ))

    if payment_schedules is not None and sales is not None:
        valid_sales = set(sales["sale_id"])
        orphaned_payments = payment_schedules[~payment_schedules["sale_id"].isin(valid_sales)]
        checks.append(_check(
            "payments_without_matching_sale", "Payment schedule rows referencing a sale_id not present in sales.csv",
            len(orphaned_payments), len(payment_schedules), orphaned_payments["installment_id"].tolist(), "HIGH",
        ))

    if customers is not None:
        dup_customer_ids = customers[customers["customer_id"].duplicated(keep=False)]
        checks.append(_check(
            "duplicate_customer_ids", "customer_id values appearing more than once in customers.csv",
            len(dup_customer_ids["customer_id"].unique()) if len(dup_customer_ids) else 0,
            int(customers["customer_id"].nunique()) or 1, sorted(dup_customer_ids["customer_id"].unique().tolist()), "HIGH",
        ))

    if construction is not None:
        impossible_pct = construction[
            (construction["actual_completion_pct"] < 0) | (construction["actual_completion_pct"] > 100)
            | (construction["planned_completion_pct"] < 0) | (construction["planned_completion_pct"] > 100)
        ]
        checks.append(_check(
            "impossible_construction_percentage", "Milestones with a completion percentage outside 0-100",
            len(impossible_pct), len(construction), impossible_pct["milestone_id"].tolist(), "HIGH",
        ))

    if units is not None:
        future_release = units[
            pd.to_datetime(units["release_date"], errors="coerce") > pd.Timestamp.now() + pd.Timedelta(days=3650)
        ]
        checks.append(_check(
            "implausible_future_dates", "Units with a release_date implausibly far in the future",
            len(future_release), len(units), future_release["unit_id"].tolist(), "LOW",
        ))

    total_penalty = sum(c["score_penalty"] for c in checks)
    score = round(max(0.0, 100.0 - total_penalty), 1)
    status = "Healthy" if score >= 90 else "Watch" if score >= 75 else "At Risk" if score >= 60 else "Critical"
    any_high_or_medium = any(c["severity"] in ("HIGH", "MEDIUM") for c in checks)

    result = {
        "data_quality_score": score,
        "status": status,
        "checks": checks,
        "any_findings_may_be_weakened": any_high_or_medium,
        "methodology": {
            "scoring": (
                f"Each of the {len(checks)} checks run contributes a penalty proportional to the "
                f"percentage of affected rows, capped at {MAX_PENALTY_PER_CHECK:.2f} points per check "
                "(so no single check can single-handedly zero the score) and summed against a 100-point "
                "baseline. This module measures and discloses data quality; it never repairs data."
            ),
            "disclosure": (
                "any_findings_may_be_weakened is true whenever at least one HIGH- or MEDIUM-severity "
                "check found affected rows -- the executive report's appendix should disclose this "
                "flag so a reader knows to treat specific figures with appropriate caution."
            ),
        },
    }
    return _json_safe(result)
