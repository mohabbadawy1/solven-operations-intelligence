"""Application entry point for the Solven Operations Intelligence Platform.

Loads raw operational data, runs the full analysis layer, and triggers
AI-powered executive report generation.

The entire pipeline is orchestrated by `run_pipeline()` below -- the
single place that sequences data loading, the four analytics engines,
and report generation. `python app.py` (via `main()`) and api.py's
POST /run-analysis both call this same function, so there is exactly
one orchestration path and no duplicated pipeline logic.

Usage:
    python app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from analysis.complaints import analyze_complaints
from analysis.correlations import analyze_correlations
from analysis.delivery import analyze_deliveries
from analysis.inventory import analyze_inventory
from ai.report_generator import ConfigurationError, ReportGenerationError, generate_executive_report

DATA_DIR = Path(__file__).parent / "data"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw CSV datasets used by the platform.

    Returns:
        A tuple of (shipments_df, complaints_df, inventory_df).
    """
    shipments_df = pd.read_csv(
        DATA_DIR / "shipments.csv",
        parse_dates=["order_date", "delivery_date", "promised_delivery_date", "actual_delivery_date"],
    )
    complaints_df = pd.read_csv(DATA_DIR / "complaints.csv", parse_dates=["complaint_date"])
    inventory_df = pd.read_csv(DATA_DIR / "inventory.csv")
    return shipments_df, complaints_df, inventory_df


def run_pipeline() -> dict[str, Any]:
    """Run the operations intelligence pipeline end-to-end and return the executive report.

    Loads the raw datasets, runs delivery/complaints/inventory analysis,
    cross-domain correlation, and AI executive report generation, in
    that order. This is the one orchestration entry point for the whole
    platform -- it contains no analytics of its own, only calls into
    analysis/* and ai.report_generator.

    Returns:
        The final report dict returned by
        `ai.report_generator.generate_executive_report` (see that
        function's docstring for its exact shape).

    Raises:
        ConfigurationError: If required AI provider configuration
            (e.g. GROQ_API_KEY) is missing.
        ReportGenerationError: If the AI executive report fails to
            generate for any other reason (invalid analytics input, AI
            provider failure, response validation failure).
    """
    print("Solven Operations Intelligence Platform")
    print("=" * 50)

    print("\nLoading datasets...")
    shipments_df, complaints_df, inventory_df = load_data()
    print(f"  Shipments:  {len(shipments_df):,} rows")
    print(f"  Complaints: {len(complaints_df):,} rows")
    print(f"  Inventory:  {len(inventory_df):,} rows")

    print("\nRunning delivery analysis...")
    delivery_insights = analyze_deliveries(shipments_df)
    kpis = delivery_insights["executive_kpis"]
    health = delivery_insights["operations_health_score"]
    print(f"  On-time delivery: {kpis['on_time_delivery_percentage']}%")
    print(f"  Delay rate: {kpis['delay_rate_percentage']}%")
    print(f"  Average driver rating: {kpis['average_driver_rating']}")
    print(f"  Best warehouse: {delivery_insights['warehouse_intelligence']['best_warehouse']}")
    print(f"  Worst warehouse: {delivery_insights['warehouse_intelligence']['worst_warehouse']}")
    print(f"  Operations Health Score: {health['overall_score']}/100")
    print(f"  Anomalies detected: {len(delivery_insights['anomalies'])}")
    for anomaly in delivery_insights["anomalies"]:
        print(f"    [{anomaly['severity']}] {anomaly['title']}")

    print("\nRunning complaints analysis...")
    complaint_insights = analyze_complaints(complaints_df, shipments_df)
    summary = complaint_insights["summary"]
    cx_health = complaint_insights["health_score"]
    print(f"  Complaint rate: {summary['complaint_rate_percentage']}%")
    print(f"  Resolution rate: {summary['resolution_rate_percentage']}%")
    print(f"  Negative sentiment: {summary['negative_sentiment_percentage']}%")
    print(f"  Best warehouse: {complaint_insights['rankings']['warehouse']['best_warehouse']}")
    print(f"  Worst warehouse: {complaint_insights['rankings']['warehouse']['worst_warehouse']}")
    print(f"  Customer Health Score: {cx_health['overall_score']}/100")
    print(f"  Anomalies detected: {len(complaint_insights['anomalies'])}")
    for anomaly in complaint_insights["anomalies"]:
        print(f"    [{anomaly['severity']}] {anomaly['title']}")

    print("\nRunning inventory analysis...")
    inventory_insights = analyze_inventory(inventory_df, shipments_df)
    inv_summary = inventory_insights["summary"]
    inv_health = inventory_insights["health_score"]
    print(f"  Total inventory value: {inv_summary['total_inventory_value']:,}")
    print(f"  Below reorder threshold: {inv_summary['percentage_below_reorder_threshold']}%")
    print(f"  Avg days of inventory remaining: {inv_summary['average_days_of_inventory_remaining']}")
    print(f"  Best warehouse: {inventory_insights['rankings']['warehouse']['best_warehouse']}")
    print(f"  Highest-risk warehouse: {inventory_insights['rankings']['warehouse']['highest_risk_warehouse']}")
    print(f"  Inventory Health Score: {inv_health['overall_score']}/100")
    print(f"  Anomalies detected: {len(inventory_insights['anomalies'])}")
    for anomaly in inventory_insights["anomalies"]:
        print(f"    [{anomaly['severity']}] {anomaly['title']}")

    print("\nRunning cross-domain correlation analysis...")
    correlation_insights = analyze_correlations(
        delivery_insights, complaint_insights, inventory_insights,
        shipments_df, complaints_df, inventory_df,
    )
    exec_summary = correlation_insights["executive_summary"]
    print(f"  Overall business risk level: {exec_summary['overall_business_risk_level']}")
    print(f"  Executive priority: {exec_summary['executive_priority']}")
    print(f"  Root causes identified: {len(correlation_insights['root_causes'])}")
    for root_cause in correlation_insights["root_causes"]:
        print(f"    [{root_cause['confidence']}] {root_cause['title']}")
    print(f"  Priority actions: {len(correlation_insights['priority_actions'])}")
    for action in correlation_insights["priority_actions"]:
        print(f"    {action['priority']}. {action['title']} ({action['difficulty']}, {action['estimated_timeframe']})")

    print("\nGenerating AI executive report...")
    report = generate_executive_report(
        delivery_insights, complaint_insights, inventory_insights, correlation_insights,
    )
    print(f"  Report ID: {report['metadata']['report_id']}")
    print(f"  Saved: {report['metadata']['saved_json_path']}")
    print(f"  Saved: {report['metadata']['saved_markdown_path']}")
    print(f"  Saved: {report['metadata']['saved_html_path']}")

    return report


def main() -> None:
    """CLI entry point: run the pipeline and print a friendly outcome on failure.

    Mirrors the previous behavior of this script exactly -- errors from
    report generation are caught and summarized rather than raised as a
    traceback. Callers that need the report itself (e.g. api.py) should
    call `run_pipeline()` directly instead.
    """
    try:
        run_pipeline()
    except ConfigurationError as exc:
        print(f"  Skipped -- {exc}")
    except ReportGenerationError as exc:
        print(f"  Failed -- {exc}")


if __name__ == "__main__":
    main()
