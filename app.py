"""Application entry point for Solven Real Estate Intelligence.

Loads Horizon Developments' synthetic relational dataset, runs the
full nine-domain analytics layer plus data quality, executive scoring,
and cross-functional correlation, and triggers AI-powered executive
report generation.

The entire pipeline is orchestrated by `run_pipeline()` below -- the
single place that sequences data loading, the analytics engines, and
report generation. `python app.py` (via `main()`) and api.py's
POST /run-analysis both call this same function, so there is exactly
one orchestration path and no duplicated pipeline logic.

Usage:
    python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

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
from ai.pdf_report_renderer import PDFRenderError, render_pdf
from ai.report_generator import ConfigurationError, ReportGenerationError, generate_executive_report

DATA_DIR = Path(__file__).parent / "data"

DATASET_NAMES = [
    "projects", "units", "customers", "brokers", "sales_agents", "campaigns", "leads", "sales",
    "payment_schedules", "collections", "construction_milestones", "customer_cases", "handovers", "snagging",
]

DATE_COLUMNS_BY_DATASET: dict[str, list[str]] = {
    "projects": ["launch_date", "expected_completion_date"],
    "units": ["release_date", "reservation_date", "contract_date", "cancellation_date", "expected_handover_date", "forecast_handover_date"],
    "customers": ["customer_since"],
    "brokers": ["onboarding_date"],
    "sales_agents": ["hire_date"],
    "campaigns": ["start_date", "end_date"],
    "leads": ["created_at", "first_response_at", "first_contact_at", "last_activity_at", "appointment_date", "site_visit_date"],
    "sales": ["reservation_date", "contract_date", "cancellation_date", "expected_handover_date", "forecast_handover_date"],
    "payment_schedules": ["due_date", "payment_date"],
    "collections": ["event_date", "promise_to_pay_date"],
    "construction_milestones": ["baseline_start_date", "baseline_end_date", "forecast_end_date", "actual_end_date"],
    "customer_cases": ["created_at", "first_response_at", "resolved_at"],
    "handovers": ["original_handover_date", "forecast_handover_date", "actual_handover_date"],
    "snagging": ["created_at", "target_resolution_date", "resolved_at"],
}


def load_data() -> dict[str, pd.DataFrame]:
    """Load every raw CSV dataset the platform uses, with dates parsed.

    Returns:
        A dict of {dataset_name: DataFrame} for all 14 datasets in data/.
    """
    datasets: dict[str, pd.DataFrame] = {}
    for name in DATASET_NAMES:
        path = DATA_DIR / f"{name}.csv"
        date_columns = DATE_COLUMNS_BY_DATASET.get(name, [])
        datasets[name] = pd.read_csv(path, parse_dates=date_columns) if date_columns else pd.read_csv(path)
    return datasets


def _average_monthly_net_sales_by_project(sales_df: pd.DataFrame) -> dict[str, float]:
    """Average monthly net-contracted unit count per project -- the one figure
    analysis.inventory_velocity needs from sales_performance's own domain
    (months-of-supply), computed here rather than inside either module so
    neither module has to import the other."""
    net = sales_df[sales_df["contract_date"].notna() & ~sales_df["cancellation_flag"]].copy()
    net["month"] = pd.to_datetime(net["reservation_date"]).dt.to_period("M").astype(str)
    monthly_counts = net.groupby(["project_id", "month"]).size()
    return monthly_counts.groupby("project_id").mean().to_dict()


def run_pipeline() -> dict[str, Any]:
    """Run the real estate operations intelligence pipeline end-to-end and return the executive report.

    Loads the fourteen relational datasets, runs the nine domain
    analytics engines, data quality, executive scoring, cross-domain
    correlation, and AI executive report generation, in that order.
    This is the one orchestration entry point for the whole platform --
    it contains no analytics of its own, only calls into analysis/*
    and ai.report_generator.

    Returns:
        The final report dict returned by
        `ai.report_generator.generate_executive_report`.

    Raises:
        ConfigurationError: If required AI provider configuration
            (e.g. GROQ_API_KEY) is missing.
        ReportGenerationError: If the AI executive report fails to
            generate for any other reason.
    """
    print("Solven Real Estate Intelligence -- Horizon Developments")
    print("=" * 60)

    print("\nLoading datasets...")
    data = load_data()
    for name in DATASET_NAMES:
        print(f"  {name}: {len(data[name]):,} rows")

    print("\nRunning commercial performance analysis...")
    sales_performance = analyze_sales_performance(data["sales"], data["units"])
    print(f"  Net contracted sales: {sales_performance['summary']['net_contracted_sales_value']:,.0f}")
    print(f"  Health score: {sales_performance['health_score']['overall_score']}/100")

    print("\nRunning lead funnel analysis...")
    lead_funnel = analyze_lead_funnel(data["leads"])
    print(f"  Lead-to-contract conversion: {lead_funnel['summary']['lead_to_reservation_conversion_pct']}%")
    print(f"  Health score: {lead_funnel['health_score']['overall_score']}/100")

    print("\nRunning inventory & pricing analysis...")
    monthly_velocity = _average_monthly_net_sales_by_project(data["sales"])
    inventory_velocity = analyze_inventory_velocity(data["units"], monthly_velocity)
    print(f"  Available inventory value: {inventory_velocity['summary']['available_inventory_value']:,.0f}")
    print(f"  Health score: {inventory_velocity['health_score']['overall_score']}/100")

    print("\nRunning marketing efficiency analysis...")
    marketing_efficiency = analyze_marketing_efficiency(data["campaigns"])
    print(f"  Marketing efficiency ratio: {marketing_efficiency['summary']['overall_marketing_efficiency_ratio']}x")
    print(f"  Health score: {marketing_efficiency['health_score']['overall_score']}/100")

    print("\nRunning sales team & broker performance analysis...")
    sales_team_broker_performance = analyze_sales_team_broker_performance(data["sales"], data["brokers"])
    print(f"  Broker share of reservations: {sales_team_broker_performance['summary']['broker_share_of_reservations_pct']}%")
    print(f"  Health score: {sales_team_broker_performance['health_score']['overall_score']}/100")

    print("\nRunning collections & receivables risk analysis...")
    collections_risk = analyze_collections_risk(data["payment_schedules"], data["collections"], data["sales"])
    print(f"  Total overdue: {collections_risk['summary']['total_overdue_amount']:,.0f}")
    print(f"  Health score: {collections_risk['health_score']['overall_score']}/100")

    print("\nRunning cancellations & revenue leakage analysis...")
    cancellations = analyze_cancellations(data["sales"])
    print(f"  Contract cancellation rate: {cancellations['summary']['contract_cancellation_rate_pct']}%")
    print(f"  Health score: {cancellations['health_score']['overall_score']}/100")

    print("\nRunning construction & handover risk analysis...")
    construction_handover = analyze_construction_handover(
        data["construction_milestones"], data["handovers"], data["snagging"], data["sales"], data["payment_schedules"],
    )
    print(f"  Handover delay rate: {construction_handover['summary']['handover_delay_rate_pct']}%")
    print(f"  Health score: {construction_handover['health_score']['overall_score']}/100")

    print("\nRunning customer experience analysis...")
    customer_experience = analyze_customer_experience(data["customer_cases"])
    print(f"  Negative sentiment: {customer_experience['summary']['negative_sentiment_pct']}%")
    print(f"  Health score: {customer_experience['health_score']['overall_score']}/100")

    print("\nRunning data quality checks...")
    data_quality = analyze_data_quality(data)
    print(f"  Data Quality Score: {data_quality['data_quality_score']}/100 ({data_quality['status']})")

    print("\nComputing executive health scores...")
    executive_scores = compute_executive_scores(
        sales_performance, lead_funnel, inventory_velocity, marketing_efficiency,
        collections_risk, cancellations, construction_handover, customer_experience, data_quality,
    )
    print(f"  Overall Business Health: {executive_scores['overall_business_health']['score']}/100 "
          f"({executive_scores['overall_business_health']['status']})")

    print("\nRunning cross-functional root cause & correlation analysis...")
    health_scores_flat = {k: v["score"] for k, v in executive_scores["domains"].items()}
    correlation_analysis = analyze_correlations(
        sales_performance, lead_funnel, inventory_velocity, marketing_efficiency,
        sales_team_broker_performance, collections_risk, cancellations, construction_handover,
        customer_experience, health_scores_flat, data["sales"],
    )
    print(f"  Overall business risk level: {correlation_analysis['executive_summary']['overall_business_risk_level']}")
    print(f"  Root causes identified: {len(correlation_analysis['root_causes'])}")
    for rc in correlation_analysis["root_causes"]:
        print(f"    [{rc['confidence']}] {rc['title']}")
    print(f"  Priority actions: {len(correlation_analysis['priority_actions'])}")

    print("\nGenerating AI executive report...")
    report = generate_executive_report(
        sales_performance, lead_funnel, inventory_velocity, marketing_efficiency,
        sales_team_broker_performance, collections_risk, cancellations, construction_handover,
        customer_experience, data_quality, executive_scores, correlation_analysis,
    )
    print(f"  Report ID: {report['metadata']['report_id']}")
    print(f"  Saved: {report['metadata']['saved_json_path']}")
    print(f"  Saved: {report['metadata']['saved_markdown_path']}")
    print(f"  Saved: {report['metadata']['saved_html_path']}")

    print("\nRendering PDF from HTML report...")
    html_path = Path(report["metadata"]["saved_html_path"])
    pdf_path = render_pdf(html_path, html_path.with_suffix(".pdf"))
    report["metadata"]["saved_pdf_path"] = str(pdf_path)
    print(f"  Saved: {pdf_path}")

    return report


def main() -> None:
    """CLI entry point: run the pipeline and print a friendly outcome on failure.

    PDF rendering failures exit non-zero so CI (and any other caller
    that checks the exit code) fails loudly instead of reporting
    success with a missing PDF. Callers that need the report itself
    (e.g. api.py) should call `run_pipeline()` directly instead.
    """
    try:
        run_pipeline()
    except ConfigurationError as exc:
        print(f"  Skipped -- {exc}")
    except ReportGenerationError as exc:
        print(f"  Failed -- {exc}")
    except PDFRenderError as exc:
        print(f"  Failed -- {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
