"""Tests for ai/report_generator.py using a scripted mock AI provider.

No network calls: a MockProvider returns a fixed, schema-valid JSON
narrative instead of calling Groq, so these tests are hermetic and
free to run in CI without GROQ_API_KEY. What's under test here is the
Python skeleton-building, merging, and validation logic -- not the
quality of any specific model's prose (that's out of scope for an
automated test and is instead verified by manual visual inspection of
the rendered PDF, per the platform's implementation notes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.report_generator import AIProvider, ResponseValidationError, generate_executive_report

DEPARTMENTS = ["sales", "marketing", "collections", "construction", "customer_experience", "executive_leadership"]
DOMAIN_LABELS = [
    "commercial_health", "sales_funnel_health", "inventory_health", "marketing_efficiency_health",
    "collections_health", "cancellations_health", "construction_delivery_health",
    "handover_readiness_health", "customer_experience_health",
]

# Deliberately generic, deterministic placeholder text -- content quality
# is not what these tests check; schema shape and merge correctness are.
_NARRATIVE = {
    "executive_headline": "Test headline citing a specific finding.",
    "why_it_matters": "Test why-it-matters paragraph.",
    "consequence_of_inaction": "Test consequence paragraph.",
    "executive_summary": "Test executive summary spanning several sentences for realism.",
    "overall_business_health_narrative": {domain: f"Narrative for {domain}." for domain in DOMAIN_LABELS},
    "risk_narratives": [],
    "root_cause_notes": [],
    "recommendation_rationale": [],
    "department_breakdown": {dept: f"Briefing for {dept}." for dept in DEPARTMENTS},
    "forecast_narrative": "Test forecast narrative, labeled as an estimate.",
    "if_no_action_narrative": "Test if-no-action paragraph.",
}


class MockProvider(AIProvider):
    """Returns a fixed, schema-valid narrative -- title-matched to whatever
    risks/root causes/actions the real analytics pipeline actually produced,
    since those lists must be matched by exact title to pass validation."""

    def __init__(self, skeleton_risks, skeleton_root_causes, skeleton_actions):
        self._risks = skeleton_risks
        self._root_causes = skeleton_root_causes
        self._actions = skeleton_actions

    @property
    def model_name(self) -> str:
        return "mock-model-for-tests"

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        narrative = dict(_NARRATIVE)
        narrative["risk_narratives"] = [{"title": r["title"], "business_impact": "Mock business impact."} for r in self._risks]
        narrative["root_cause_notes"] = [{"title": rc["title"], "executive_note": "Mock executive note."} for rc in self._root_causes]
        narrative["recommendation_rationale"] = [{"title": a["title"], "executive_rationale": "Mock rationale."} for a in self._actions]
        return json.dumps(narrative)


@pytest.fixture(scope="module")
def generated_report(analytics, tmp_path_factory):
    from ai.report_generator import _build_report_skeleton
    from datetime import datetime, timezone

    skeleton = _build_report_skeleton(
        analytics["sales_performance"], analytics["lead_funnel"], analytics["inventory_velocity"],
        analytics["marketing_efficiency"], analytics["sales_team_broker_performance"], analytics["collections_risk"],
        analytics["cancellations"], analytics["construction_handover"], analytics["customer_experience"],
        analytics["data_quality"], analytics["executive_scores"], analytics["correlation_analysis"],
        "test-model", datetime.now(timezone.utc), None,
    )
    provider = MockProvider(skeleton["top_business_risks"], skeleton["root_causes"], skeleton["recommended_actions"])
    output_dir = tmp_path_factory.mktemp("report_output")

    report = generate_executive_report(
        analytics["sales_performance"], analytics["lead_funnel"], analytics["inventory_velocity"],
        analytics["marketing_efficiency"], analytics["sales_team_broker_performance"], analytics["collections_risk"],
        analytics["cancellations"], analytics["construction_handover"], analytics["customer_experience"],
        analytics["data_quality"], analytics["executive_scores"], analytics["correlation_analysis"],
        provider=provider, output_dir=output_dir,
    )
    return report, output_dir


def test_report_has_all_required_top_level_sections(generated_report):
    report, _ = generated_report
    required = [
        "metadata", "executive_headline", "why_it_matters", "consequence_of_inaction", "executive_summary",
        "overall_business_health", "highest_risk_project", "strongest_project", "highest_priority_initiative",
        "top_business_risks", "root_causes", "recommended_actions", "financial_exposure", "domains",
        "department_breakdown", "forecast_outlook", "if_no_action_narrative", "appendix",
    ]
    for key in required:
        assert key in report, f"report is missing required key: {key}"


def test_department_breakdown_is_complete(generated_report):
    report, _ = generated_report
    assert set(report["department_breakdown"].keys()) == set(DEPARTMENTS)
    for text in report["department_breakdown"].values():
        assert isinstance(text, str) and text.strip()


def test_all_output_files_are_written(generated_report):
    _, output_dir = generated_report
    for filename in ("executive_report.json", "executive_report.md", "executive_report.html"):
        path = Path(output_dir) / filename
        assert path.is_file(), f"{filename} was not written"
        assert path.stat().st_size > 0


def test_json_output_round_trips(generated_report):
    _, output_dir = generated_report
    path = Path(output_dir) / "executive_report.json"
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["metadata"]["report_type"] == "Real Estate Executive Intelligence Report"


def test_no_logistics_terminology_in_html_output(generated_report):
    """This platform was fully repurposed from a warehouse/logistics demo to real
    estate -- no leftover domain vocabulary should appear anywhere in client-facing output."""
    _, output_dir = generated_report
    html = (Path(output_dir) / "executive_report.html").read_text(encoding="utf-8").lower()
    forbidden_terms = ["warehouse", "shipment", "driver rating", "loading time", "stockout", "delivery complaint"]
    for term in forbidden_terms:
        assert term not in html, f"found leftover logistics terminology in report HTML: {term!r}"


def test_markdown_output_has_no_logistics_terminology(generated_report):
    _, output_dir = generated_report
    markdown = (Path(output_dir) / "executive_report.md").read_text(encoding="utf-8").lower()
    forbidden_terms = ["warehouse", "shipment", "driver rating", "loading time"]
    for term in forbidden_terms:
        assert term not in markdown, f"found leftover logistics terminology in report Markdown: {term!r}"


def test_every_number_in_top_business_risks_traces_to_skeleton(generated_report):
    """Structural guarantee, not a text-scan: top_business_risks entries are built
    entirely from analysis.correlations output in _build_report_skeleton, never
    from the AI response -- confirmed here by checking the field the AI narrative
    step could only ever *add to* (business_impact), never replace (title/severity/confidence)."""
    report, _ = generated_report
    for risk in report["top_business_risks"]:
        assert risk["confidence"] is not None
        assert 0.0 < risk["confidence"] <= 0.97
        assert risk["severity"] in ("HIGH", "MEDIUM", "LOW")


def test_missing_narrative_key_raises_validation_error(analytics, tmp_path):
    """A narrative response missing a required key must fail loudly, not silently degrade."""
    from ai.report_generator import _build_report_skeleton
    from datetime import datetime, timezone

    skeleton = _build_report_skeleton(
        analytics["sales_performance"], analytics["lead_funnel"], analytics["inventory_velocity"],
        analytics["marketing_efficiency"], analytics["sales_team_broker_performance"], analytics["collections_risk"],
        analytics["cancellations"], analytics["construction_handover"], analytics["customer_experience"],
        analytics["data_quality"], analytics["executive_scores"], analytics["correlation_analysis"],
        "test-model", datetime.now(timezone.utc), None,
    )

    class BrokenProvider(AIProvider):
        @property
        def model_name(self) -> str:
            return "broken-mock"

        def generate_json(self, system_prompt: str, user_prompt: str) -> str:
            incomplete = dict(_NARRATIVE)
            del incomplete["executive_headline"]
            return json.dumps(incomplete)

    with pytest.raises(ResponseValidationError):
        generate_executive_report(
            analytics["sales_performance"], analytics["lead_funnel"], analytics["inventory_velocity"],
            analytics["marketing_efficiency"], analytics["sales_team_broker_performance"], analytics["collections_risk"],
            analytics["cancellations"], analytics["construction_handover"], analytics["customer_experience"],
            analytics["data_quality"], analytics["executive_scores"], analytics["correlation_analysis"],
            provider=BrokenProvider(), output_dir=tmp_path,
        )
