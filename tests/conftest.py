"""Shared pytest fixtures: load the committed synthetic dataset once per session.

Tests read data/*.csv (already generated and committed, exactly what
app.py/CI use) rather than regenerating it on every test run -- keeps
the suite fast while still exercising the real, committed dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASET_NAMES = [
    "projects", "units", "customers", "brokers", "sales_agents", "campaigns", "leads", "sales",
    "payment_schedules", "collections", "construction_milestones", "customer_cases", "handovers", "snagging",
]
DATE_COLUMNS_BY_DATASET = {
    "units": ["release_date", "reservation_date", "contract_date", "cancellation_date"],
    "leads": ["created_at", "first_response_at", "first_contact_at", "last_activity_at"],
    "sales": ["reservation_date", "contract_date", "cancellation_date"],
    "payment_schedules": ["due_date", "payment_date"],
    "collections": ["event_date"],
    "construction_milestones": ["baseline_start_date", "baseline_end_date", "forecast_end_date", "actual_end_date"],
    "customer_cases": ["created_at", "first_response_at", "resolved_at"],
    "handovers": ["original_handover_date", "forecast_handover_date", "actual_handover_date"],
}


@pytest.fixture(scope="session")
def datasets() -> dict[str, pd.DataFrame]:
    result = {}
    for name in DATASET_NAMES:
        path = DATA_DIR / f"{name}.csv"
        if not path.is_file():
            pytest.skip(f"data/{name}.csv not found -- run `python generate_real_estate_data.py` first.")
        date_cols = DATE_COLUMNS_BY_DATASET.get(name, [])
        result[name] = pd.read_csv(path, parse_dates=date_cols) if date_cols else pd.read_csv(path)
    return result


@pytest.fixture(scope="session")
def analytics(datasets):
    """Every domain analytics module's output, computed once for the whole test session."""
    from analysis.cancellations import analyze_cancellations
    from analysis.collections_risk import analyze_collections_risk
    from analysis.construction_handover import analyze_construction_handover
    from analysis.correlations import analyze_correlations
    from analysis.customer_experience import analyze_customer_experience
    from analysis.data_quality import analyze_data_quality
    from analysis.executive_scoring import compute_executive_scores
    from analysis.inventory_velocity import analyze_inventory_velocity
    from analysis.lead_funnel import analyze_lead_funnel
    from analysis.marketing_efficiency import analyze_marketing_efficiency
    from analysis.sales_performance import analyze_sales_performance
    from analysis.sales_team_broker_performance import analyze_sales_team_broker_performance

    sales_performance = analyze_sales_performance(datasets["sales"], datasets["units"])
    lead_funnel = analyze_lead_funnel(datasets["leads"])

    net = datasets["sales"][datasets["sales"]["contract_date"].notna() & ~datasets["sales"]["cancellation_flag"]].copy()
    net["month"] = pd.to_datetime(net["reservation_date"]).dt.to_period("M").astype(str)
    monthly_velocity = net.groupby(["project_id", "month"]).size().groupby("project_id").mean().to_dict()

    inventory_velocity = analyze_inventory_velocity(datasets["units"], monthly_velocity)
    marketing_efficiency = analyze_marketing_efficiency(datasets["campaigns"])
    sales_team_broker_performance = analyze_sales_team_broker_performance(datasets["sales"], datasets["brokers"])
    collections_risk = analyze_collections_risk(datasets["payment_schedules"], datasets["collections"], datasets["sales"])
    cancellations = analyze_cancellations(datasets["sales"])
    construction_handover = analyze_construction_handover(
        datasets["construction_milestones"], datasets["handovers"], datasets["snagging"],
        datasets["sales"], datasets["payment_schedules"],
    )
    customer_experience = analyze_customer_experience(datasets["customer_cases"])
    data_quality = analyze_data_quality(datasets)
    executive_scores = compute_executive_scores(
        sales_performance, lead_funnel, inventory_velocity, marketing_efficiency,
        collections_risk, cancellations, construction_handover, customer_experience, data_quality,
    )
    health_scores_flat = {k: v["score"] for k, v in executive_scores["domains"].items()}
    correlation_analysis = analyze_correlations(
        sales_performance, lead_funnel, inventory_velocity, marketing_efficiency,
        sales_team_broker_performance, collections_risk, cancellations, construction_handover,
        customer_experience, health_scores_flat, datasets["sales"],
    )

    return {
        "sales_performance": sales_performance, "lead_funnel": lead_funnel, "inventory_velocity": inventory_velocity,
        "marketing_efficiency": marketing_efficiency, "sales_team_broker_performance": sales_team_broker_performance,
        "collections_risk": collections_risk, "cancellations": cancellations,
        "construction_handover": construction_handover, "customer_experience": customer_experience,
        "data_quality": data_quality, "executive_scores": executive_scores, "correlation_analysis": correlation_analysis,
    }
