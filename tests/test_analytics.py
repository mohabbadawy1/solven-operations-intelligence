"""Tests for the domain analytics engines: score bounds, KPI correctness, structural invariants."""

from __future__ import annotations

import pytest

ALL_HEALTH_SCORE_DOMAINS = [
    "sales_performance", "lead_funnel", "inventory_velocity", "marketing_efficiency",
    "sales_team_broker_performance", "collections_risk", "cancellations",
    "construction_handover", "customer_experience",
]


@pytest.mark.parametrize("domain", ALL_HEALTH_SCORE_DOMAINS)
def test_health_scores_are_bounded_0_to_100(analytics, domain):
    score = analytics[domain]["health_score"]["overall_score"]
    assert score is not None
    assert 0.0 <= score <= 100.0


@pytest.mark.parametrize("domain", ALL_HEALTH_SCORE_DOMAINS)
def test_health_score_components_sum_to_overall_via_weights(analytics, domain):
    health = analytics[domain]["health_score"]
    weighted_sum = sum(health["components"][k] * health["weights"][k] for k in health["weights"])
    assert health["overall_score"] == pytest.approx(round(weighted_sum, 1), abs=0.15)


def test_overall_business_health_is_bounded(analytics):
    overall = analytics["executive_scores"]["overall_business_health"]
    assert 0.0 <= overall["score"] <= 100.0
    assert overall["status"] in ("Healthy", "Watch", "At Risk", "Critical")


def test_overall_business_health_is_not_a_naive_average(analytics):
    """The overall score must differ from a plain mean of the nine domain scores --
    otherwise the materiality weighting in config/real_estate_demo.yml isn't doing anything."""
    domains = analytics["executive_scores"]["domains"]
    scores = [v["score"] for v in domains.values()]
    naive_average = round(sum(scores) / len(scores), 1)
    weighted = analytics["executive_scores"]["overall_business_health"]["score"]
    # Only assert they differ if the weights are actually non-uniform (they are, by config) --
    # a tiny coincidental match is fine, an exact match every domain would indicate no weighting.
    assert weighted != naive_average or len(set(scores)) <= 1


def test_confidence_scores_are_bounded_and_never_certain(analytics):
    for rc in analytics["correlation_analysis"]["root_causes"]:
        assert 0.0 < rc["confidence"] <= 0.97


def test_funnel_waterfall_is_monotonically_non_increasing(analytics):
    waterfall = analytics["lead_funnel"]["kpis"]["funnel_waterfall"]
    counts = [stage["count"] for stage in waterfall]
    assert counts == sorted(counts, reverse=True)


def test_funnel_stage_order_matches_canonical_sequence(analytics):
    expected_order = ["New Leads", "Contacted", "Qualified", "Appointment Booked", "Site Visit Completed", "Converted to Sale"]
    actual_order = [stage["stage"] for stage in analytics["lead_funnel"]["kpis"]["funnel_waterfall"]]
    assert actual_order == expected_order


def test_receivables_aging_buckets_sum_to_total_overdue(analytics):
    aging = analytics["collections_risk"]["kpis"]["aging"]["aging_buckets"]
    overdue_total = analytics["collections_risk"]["summary"]["total_overdue_amount"]
    bucket_sum = sum(row["amount"] for row in aging if row["aging_bucket"] != "Current")
    assert bucket_sum == pytest.approx(overdue_total, abs=1.0)


def test_receivables_aging_uses_canonical_bucket_order(analytics):
    expected_order = ["Current", "1-30 days", "31-60 days", "61-90 days", "91-180 days", "180+ days"]
    actual_order = [row["aging_bucket"] for row in analytics["collections_risk"]["kpis"]["aging"]["aging_buckets"]]
    assert actual_order == expected_order


def test_cancellation_rate_is_internally_consistent(analytics):
    summary = analytics["cancellations"]["summary"]
    expected_rate = round(100 * summary["total_cancellations"] / summary["total_reservations"], 1)
    assert summary["reservation_cancellation_rate_pct"] == pytest.approx(expected_rate, abs=0.1)


def test_construction_variance_is_signed_days_not_percentage(analytics):
    summary = analytics["construction_handover"]["summary"]
    assert summary["average_schedule_variance_days"] is not None
    # A schedule variance of thousands of days would indicate a units/format bug.
    assert -365 <= summary["average_schedule_variance_days"] <= 365


def test_financial_exposure_line_items_are_non_negative(analytics):
    for item in analytics["correlation_analysis"]["financial_exposure"]["line_items"]:
        assert item["amount"] is None or item["amount"] >= 0


def test_project_ranking_is_sorted_correctly(analytics):
    by_project = analytics["sales_performance"]["rankings"]["by_project"]
    values = [row["net_sales_value"] for row in by_project]
    assert values == sorted(values, reverse=True)


def test_data_quality_score_is_bounded(analytics):
    assert 0.0 <= analytics["data_quality"]["data_quality_score"] <= 100.0


def test_expected_engineered_root_causes_are_found(analytics):
    """This platform's demo data engineers five specific business problems (see
    generate_real_estate_data.py's module docstring); the correlation engine
    should discover all five without any of them being hardcoded into it."""
    titles = " ".join(rc["title"] for rc in analytics["correlation_analysis"]["root_causes"])
    assert "HVW" in titles  # affordability-driven collections & cancellation risk
    assert "AUR" in titles  # sales capacity bottleneck
    assert "CST" in titles  # seasonal demand concentration
    assert "MER" in titles  # construction delay exposure
    assert any("Broker" in rc["title"] or "Realty" in rc["title"] for rc in analytics["correlation_analysis"]["root_causes"])
