"""AI Orchestration Layer: turns computed real-estate analytics into an executive report.

This is the fourth and final layer of the platform:

    CSV Data -> Domain Analytics Layer -> Correlation/Scoring Layer -> AI Orchestration Layer

The nine domain analytics engines (analysis/sales_performance.py,
lead_funnel.py, inventory_velocity.py, marketing_efficiency.py,
sales_team_broker_performance.py, collections_risk.py, cancellations.py,
construction_handover.py, customer_experience.py), plus
analysis/data_quality.py, analysis/executive_scoring.py, and
analysis/correlations.py, have already done every calculation this
platform performs: every KPI, every health score, every root cause,
every confidence score, and every financial exposure figure. Nothing
in this module recomputes any of that. This module's only job is
communication -- taking findings that already exist and writing them
up the way a seasoned real-estate executive analyst would brief a
board.

Why the AI performs communication, not analytics
--------------------------------------------------
Identical rationale to the platform's original logistics-era design:
an LLM is unreliable at arithmetic and prone to inventing a
plausible-sounding number. So this module builds a "report skeleton"
in pure Python first -- every number, ranking, project/broker/agent
name, confidence score, and financial figure that appears in the final
report is copied directly from the analytics layer's dicts before the
AI is ever called. The single Groq call that follows is asked to
return *only* narrative text (an executive summary, department
briefings, one-sentence risk/root-cause framings) keyed to match the
skeleton, which Python then merges in. The model literally cannot
corrupt a number, because no number ever depends on the model
reproducing it correctly.

Section bodies (the detailed KPI tables, rankings, and charts inside
each domain section) are rendered directly from the skeleton's own
Python data by ai/html_report_renderer.py -- they carry no AI-written
prose at all, only a report-wide narrative layer (headline, summary,
per-domain one-liners, root-cause/action rationale, department
briefings, forecast commentary, and the "if no action" projection).
This keeps the report genuinely data-led rather than prose-heavy, and
keeps the AI's surface area small and auditable.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import groq
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from analysis._config import CONFIG
from ai.html_report_renderer import HTMLRenderError, _fmt_currency, render_html

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PLATFORM_NAME = CONFIG["company"]["platform_label"]
REPORT_TYPE = CONFIG["company"]["report_type"]
COMPANY_NAME = CONFIG["company"]["name"]
CURRENCY = CONFIG["company"]["currency"]

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.3
# Groq's token-per-minute rate limit counts input tokens *and*
# max_output_tokens against the same budget for a single request (the
# API reserves max_output_tokens up front) -- this platform's free/
# on-demand tier caps at 12,000 TPM, so this is sized to leave
# comfortable headroom next to the curated (not dumped) AI context
# this module sends, which runs several thousand tokens on its own
# given nine analytics domains.
DEFAULT_MAX_OUTPUT_TOKENS = 4000
DEFAULT_TIMEOUT_SECONDS = 75.0
DEFAULT_MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
REPORT_JSON_FILENAME = "executive_report.json"
REPORT_MARKDOWN_FILENAME = "executive_report.md"
REPORT_HTML_FILENAME = "executive_report.html"

DEPARTMENTS = ["sales", "marketing", "collections", "construction", "customer_experience", "executive_leadership"]
DOMAIN_LABELS = [
    "commercial_health", "sales_funnel_health", "inventory_health", "marketing_efficiency_health",
    "collections_health", "cancellations_health", "construction_delivery_health",
    "handover_readiness_health", "customer_experience_health",
]

TOP_RISK_COUNT = 5  # root causes shown as top_business_risks
NUMBER_GROUNDING_IGNORE = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
NUMBER_PATTERN = re.compile(r"\d[\d,]*\.?\d*%?")

REQUIRED_NARRATIVE_KEYS = [
    "executive_headline", "why_it_matters", "consequence_of_inaction", "executive_summary",
    "overall_business_health_narrative", "risk_narratives", "root_cause_notes",
    "recommendation_rationale", "department_breakdown", "forecast_narrative", "if_no_action_narrative",
]
REQUIRED_REPORT_KEYS = [
    "metadata", "executive_headline", "why_it_matters", "consequence_of_inaction", "executive_summary",
    "overall_business_health", "highest_risk_project", "strongest_project", "highest_priority_initiative",
    "top_business_risks", "root_causes", "recommended_actions", "financial_exposure", "domains",
    "department_breakdown", "forecast_outlook", "if_no_action_narrative", "appendix",
]
# Top-level string fields on the assembled report dict. forecast_outlook's
# own "narrative" text is validated/scanned separately since it lives
# nested under report["forecast_outlook"]["narrative"], not as a
# top-level key -- see _validate_final_report and _find_unverified_numbers.
NARRATIVE_TEXT_REPORT_KEYS = ["executive_headline", "why_it_matters", "consequence_of_inaction", "executive_summary", "if_no_action_narrative"]


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class ReportGenerationError(Exception):
    """Base class for every error this module raises. Never fails silently."""


class ConfigurationError(ReportGenerationError):
    """Required configuration (an environment variable, typically) is missing or invalid."""


class AnalyticsInputError(ReportGenerationError):
    """One of the analytics inputs is missing, empty, or malformed."""


class AIProviderError(ReportGenerationError):
    """The AI provider failed to produce a usable response (auth, network, rate limit, timeout)."""


class ResponseValidationError(ReportGenerationError):
    """The AI's response, or the assembled report, failed structural validation."""


# --------------------------------------------------------------------------
# AI provider abstraction (domain-agnostic -- unchanged from the platform's original design)
# --------------------------------------------------------------------------

class AIProvider(ABC):
    """Interface every AI backend must implement to plug into this module."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class AIConfig:
    api_key: str
    model: str
    timeout_seconds: float
    max_retries: int
    temperature: float
    max_output_tokens: int

    @classmethod
    def from_env(cls) -> "AIConfig":
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Add it to a .env file at the project root "
                "(see .env.example) or export it in your environment before generating a report."
            )
        return cls(
            api_key=api_key,
            model=os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL,
            timeout_seconds=_float_env("GROQ_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            max_retries=int(_float_env("GROQ_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
            temperature=_float_env("GROQ_TEMPERATURE", DEFAULT_TEMPERATURE),
            max_output_tokens=int(_float_env("GROQ_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
        )


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _call_with_retry(call: Callable[[], str], max_retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call()
        except groq.AuthenticationError as exc:
            raise AIProviderError(f"Groq authentication failed -- check GROQ_API_KEY: {exc}") from exc
        except groq.BadRequestError as exc:
            raise AIProviderError(f"Groq rejected the request as malformed: {exc}") from exc
        except (
            groq.RateLimitError, groq.APITimeoutError,
            groq.APIConnectionError, groq.InternalServerError, groq.APIError,
        ) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
                continue
    raise AIProviderError(f"Groq API call failed after {max_retries + 1} attempt(s): {last_error}") from last_error


class GroqProvider(AIProvider):
    def __init__(self, config: AIConfig | None = None) -> None:
        self._config = config or AIConfig.from_env()
        self._client = Groq(api_key=self._config.api_key, timeout=self._config.timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._config.model

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        def _call() -> str:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self._config.temperature,
                max_tokens=self._config.max_output_tokens,
            )
            content = response.choices[0].message.content if response.choices else None
            if not content or not content.strip():
                raise AIProviderError("Groq returned an empty response.")
            return content

        return _call_with_retry(_call, self._config.max_retries)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def _require(analysis: Any, name: str) -> dict[str, Any]:
    if analysis is None or not isinstance(analysis, dict):
        raise AnalyticsInputError(f"generate_executive_report: {name} must be a non-empty dict")
    return analysis


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------

def _trend(previous_score: float | None, current_score: float | None) -> dict[str, Any]:
    if previous_score is None or current_score is None:
        return {"available": False, "direction": None, "delta": None}
    delta = round(current_score - previous_score, 1)
    direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
    return {"available": True, "direction": direction, "delta": delta}


def _load_previous_report(json_path: Path) -> dict[str, Any] | None:
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _match_by_title(items: list[dict[str, Any]], title: str, key: str) -> str:
    for item in items:
        if isinstance(item, dict) and item.get("title") == title:
            value = item.get(key, "")
            return value if isinstance(value, str) else ""
    return ""


def _severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.75:
        return "HIGH"
    if confidence >= 0.55:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------
# Highest risk / strongest project (convergence across domain "worst" flags)
# --------------------------------------------------------------------------

def _highest_risk_project(
    cancellations: dict, collections_risk: dict, construction_handover: dict, lead_funnel: dict, sales_performance: dict,
) -> str | None:
    """The project flagged as worst/highest-risk by the most independent domain engines.

    Reads only project names each engine already ranked worst -- no new
    comparison or statistic is computed here. Ties are broken by a
    fixed source order.
    """
    cancel_rows = cancellations.get("kpis", {}).get("cuts", {}).get("by_project", [])
    worst_cancel = cancel_rows[0]["project_id"] if cancel_rows else None
    worst_collections = collections_risk.get("kpis", {}).get("aging", {}).get("highest_overdue_exposure_project")
    worst_construction = construction_handover.get("kpis", {}).get("delay_concentration", {}).get("most_delayed_building_project")
    worst_funnel = lead_funnel.get("rankings", {}).get("weakest_converting_project")
    worst_sales = sales_performance.get("rankings", {}).get("weakest_project")

    candidates = {
        "cancellations": worst_cancel, "collections": worst_collections, "construction": worst_construction,
        "funnel": worst_funnel, "sales": worst_sales,
    }
    named = {k: v for k, v in candidates.items() if v}
    if not named:
        return None
    counts: dict[str, int] = {}
    for v in named.values():
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    tied = {p for p, c in counts.items() if c == max_count}
    for source in ("cancellations", "collections", "construction", "funnel", "sales"):
        if named.get(source) in tied:
            return named[source]
    return next(iter(tied))


def _strongest_project(sales_performance: dict, lead_funnel: dict) -> str | None:
    return sales_performance.get("rankings", {}).get("strongest_project") or lead_funnel.get("rankings", {}).get("strongest_converting_project")


# --------------------------------------------------------------------------
# Forecast (transparent run-rate projections only -- no ML, no black box)
# --------------------------------------------------------------------------

def _compute_forecast(sales_performance: dict, collections_risk: dict, cancellations: dict) -> dict[str, Any]:
    """Simple, transparent run-rate/rolling-average projections. Every figure is labeled an estimate."""
    min_months = CONFIG["forecast"]["min_months_history"]
    window = CONFIG["forecast"]["recent_trend_window_months"]

    target_attainment = sales_performance.get("kpis", {}).get("target_attainment", {})
    monthly_sales = sales_performance.get("trends", {}).get("monthly", [])
    forecast: dict[str, Any] = {"available": len(monthly_sales) >= min_months}
    if not forecast["available"]:
        forecast["note"] = f"Fewer than {min_months} months of sales history available; forecast withheld."
        return forecast

    as_of_str = target_attainment.get("as_of_date")
    as_of = pd.Timestamp(as_of_str) if as_of_str else None
    current_month_value = monthly_sales[-1]["net_sales_value"] if monthly_sales else None

    if as_of is not None and current_month_value is not None and as_of.day:
        run_rate_multiplier = as_of.days_in_month / as_of.day
        forecast["expected_month_end_net_sales"] = round(current_month_value * run_rate_multiplier, 2)
    else:
        forecast["expected_month_end_net_sales"] = None

    ytd_value = target_attainment.get("ytd_net_contracted_sales_value")
    full_year_target = target_attainment.get("full_year_annual_sales_target")
    if as_of is not None and ytd_value is not None and full_year_target:
        day_of_year = as_of.dayofyear
        days_in_year = 366 if as_of.is_leap_year else 365
        projected_full_year = round(ytd_value / (day_of_year / days_in_year), 2)
        forecast["projected_full_year_net_sales"] = projected_full_year
        forecast["projected_full_year_target_attainment_pct"] = round(100 * projected_full_year / full_year_target, 1)
    else:
        forecast["projected_full_year_net_sales"] = None
        forecast["projected_full_year_target_attainment_pct"] = None

    collections_monthly = collections_risk.get("trends", {}).get("monthly", [])
    recent_collections = [m["amount_paid"] for m in collections_monthly[-window:] if m.get("amount_paid") is not None]
    forecast["expected_next_month_collections"] = round(sum(recent_collections) / len(recent_collections), 2) if recent_collections else None

    cancellations_monthly = cancellations.get("trends", {}).get("monthly", [])
    recent_cancellations = [m["cancellations"] for m in cancellations_monthly[-window:] if m.get("cancellations") is not None]
    forecast["expected_next_month_cancellations"] = round(sum(recent_cancellations) / len(recent_cancellations), 1) if recent_cancellations else None

    overdue_amount = collections_risk.get("summary", {}).get("total_overdue_amount")
    forecast["current_overdue_exposure"] = overdue_amount

    forecast["methodology"] = (
        f"expected_month_end_net_sales is a straight-line run-rate projection (current month-to-date "
        "value scaled by days-in-month / day-of-month elapsed). projected_full_year_net_sales scales "
        "year-to-date value by days-in-year / day-of-year elapsed. expected_next_month_collections and "
        f"expected_next_month_cancellations are simple rolling averages of the most recent "
        f"{window} months. No machine learning or trend-fitting is used; every figure here is a "
        "transparent, disclosed estimate, not a prediction of certainty, and is withheld entirely if "
        f"fewer than {min_months} months of history are available."
    )
    return forecast


# --------------------------------------------------------------------------
# Report skeleton: every number and fact, built entirely in Python
# --------------------------------------------------------------------------

def _build_appendix(
    sales_performance: dict, lead_funnel: dict, inventory_velocity: dict, marketing_efficiency: dict,
    sales_team_broker_performance: dict, collections_risk: dict, cancellations: dict,
    construction_handover: dict, customer_experience: dict, data_quality: dict, correlation_analysis: dict,
) -> dict[str, Any]:
    """Assemble the appendix entirely from the domain engines' own methodology fields. No AI involved."""
    limitations: list[str] = []
    domain_sources = [
        ("sales_performance", sales_performance), ("lead_funnel", lead_funnel), ("inventory_velocity", inventory_velocity),
        ("marketing_efficiency", marketing_efficiency), ("sales_team_broker_performance", sales_team_broker_performance),
        ("collections_risk", collections_risk), ("cancellations", cancellations),
        ("construction_handover", construction_handover), ("customer_experience", customer_experience),
    ]
    for source_name, analysis in domain_sources:
        methodology = analysis.get("methodology", {})
        if not isinstance(methodology, dict):
            continue
        for key in ("limitations",):
            for value in methodology.get(key, []) or []:
                limitations.append(f"[{source_name}] {value}")

    for value in correlation_analysis.get("methodology", {}).get("limitations", []) or []:
        limitations.append(f"[correlations] {value}")

    if data_quality.get("any_findings_may_be_weakened"):
        limitations.append(
            f"[data quality] The platform's Data Quality Score is {data_quality.get('data_quality_score')}/100 "
            f"({data_quality.get('status')}); at least one integrity check found affected records -- see the "
            "Data Quality section for the full check list before treating any single figure as final."
        )

    limitations.append(
        "This executive report's narrative text (headline, summary, department briefings, root-cause "
        "and recommendation framing) is generated by an LLM from the structured analytics above; every "
        "number, ranking, project/broker/agent name, and confidence score in this report is sourced "
        "directly from Python-computed analytics and is never generated, recalculated, or altered by the AI."
    )

    return {
        "methodology_summary": (
            "Commercial, funnel, inventory, marketing, sales-team/broker, collections, cancellations, "
            "construction/handover, and customer-experience metrics are each computed independently by "
            "a dedicated analytics engine from Horizon Developments' operational data (projects, units, "
            "leads, sales, payment schedules, collections activity, construction milestones, customer "
            "cases, handovers, and snagging records -- all synthetic demo data). The correlation layer "
            "cross-references those nine engines' own outputs to identify root causes, financial "
            "exposure, and a prioritized action plan using descriptive statistics (Pearson correlation, "
            "z-scores, composite risk scoring, coefficient of variation) -- never machine learning or "
            "forecasting. Executive Health Scores blend each domain's own documented 0-100 score using "
            "materiality-aware weights (config/real_estate_demo.yml). This report's sole narrative role "
            "is to communicate those pre-computed findings in board-ready language; it performs no "
            "analysis of its own."
        ),
        "confidence_explanation": correlation_analysis.get("methodology", {}).get("confidence_methodology", "Not available."),
        "data_quality_summary": {
            "score": data_quality.get("data_quality_score"), "status": data_quality.get("status"),
            "checks_with_findings": [c for c in data_quality.get("checks", []) if c.get("severity") != "PASS"],
        },
        "data_limitations": limitations,
        "generated_by": f"{PLATFORM_NAME} AI Orchestration Layer",
    }


def _build_report_skeleton(
    sales_performance: dict, lead_funnel: dict, inventory_velocity: dict, marketing_efficiency: dict,
    sales_team_broker_performance: dict, collections_risk: dict, cancellations: dict,
    construction_handover: dict, customer_experience: dict, data_quality: dict,
    executive_scores: dict, correlation_analysis: dict,
    model_name: str, generation_started_at: datetime, previous_report: dict[str, Any] | None,
) -> dict[str, Any]:
    previous_health = (previous_report or {}).get("overall_business_health", {})
    overall_business_health = {}
    for domain_key, block in executive_scores["domains"].items():
        overall_business_health[domain_key] = {
            "score": block["score"], "status": block["status"],
            "trend": _trend(previous_health.get(domain_key, {}).get("score"), block["score"]),
        }
    overall_business_health["overall"] = {
        "score": executive_scores["overall_business_health"]["score"],
        "status": executive_scores["overall_business_health"]["status"],
        "trend": _trend(previous_health.get("overall", {}).get("score"), executive_scores["overall_business_health"]["score"]),
    }

    root_causes_raw = correlation_analysis.get("root_causes", [])
    top_business_risks = [
        {
            "rank": index + 1,
            "title": rc["title"],
            "severity": _severity_from_confidence(rc["confidence"]),
            "confidence": rc["confidence"],
            "evidence": "; ".join((rc.get("statistical_evidence") or []) + (rc.get("operational_evidence") or [])[:2]),
            "project_id": rc.get("project_id"), "broker_id": rc.get("broker_id"),
        }
        for index, rc in enumerate(root_causes_raw[:TOP_RISK_COUNT])
    ]

    root_causes = [
        {
            "title": rc["title"], "category": rc["category"], "confidence": rc["confidence"],
            "description": rc["description"], "statistical_evidence": rc.get("statistical_evidence", []),
            "operational_evidence": rc.get("operational_evidence", []),
            "financial_exposure": rc.get("financial_exposure"), "customer_impact": rc.get("customer_impact"),
            "owner": rc["recommended_owner"], "horizon": rc["recommended_horizon"],
        }
        for rc in root_causes_raw
    ]

    recommended_actions = correlation_analysis.get("priority_actions", [])

    kpi_dashboard = _build_kpi_dashboard(
        sales_performance, lead_funnel, inventory_velocity, collections_risk, cancellations,
        construction_handover, executive_scores,
    )

    metadata = {
        "report_id": str(uuid.uuid4()),
        "generated_at": generation_started_at.isoformat(),
        "platform": PLATFORM_NAME,
        "report_type": REPORT_TYPE,
        "company_name": COMPANY_NAME,
        "currency": CURRENCY,
        "ai_model": model_name,
        "data_sources": [
            "sales_performance", "lead_funnel", "inventory_velocity", "marketing_efficiency",
            "sales_team_broker_performance", "collections_risk", "cancellations",
            "construction_handover", "customer_experience", "data_quality", "correlations",
        ],
    }

    return {
        "metadata": metadata,
        "overall_business_health": overall_business_health,
        "kpi_dashboard": kpi_dashboard,
        "highest_risk_project": _highest_risk_project(cancellations, collections_risk, construction_handover, lead_funnel, sales_performance),
        "strongest_project": _strongest_project(sales_performance, lead_funnel),
        "highest_priority_initiative": recommended_actions[0]["title"] if recommended_actions else None,
        "top_business_risks": top_business_risks,
        "root_causes": root_causes,
        "recommended_actions": recommended_actions,
        "financial_exposure": correlation_analysis.get("financial_exposure", {}),
        "relationships": correlation_analysis.get("relationships", []),
        "domains": {
            "commercial_performance": sales_performance,
            "lead_funnel": lead_funnel,
            "inventory_pricing": inventory_velocity,
            "marketing_performance": marketing_efficiency,
            "sales_team_broker": sales_team_broker_performance,
            "collections_risk": collections_risk,
            "cancellations": cancellations,
            "construction_handover": construction_handover,
            "customer_experience": customer_experience,
        },
        "data_quality": data_quality,
        "forecast_outlook": _compute_forecast(sales_performance, collections_risk, cancellations),
        "appendix": _build_appendix(
            sales_performance, lead_funnel, inventory_velocity, marketing_efficiency,
            sales_team_broker_performance, collections_risk, cancellations,
            construction_handover, customer_experience, data_quality, correlation_analysis,
        ),
    }


def _build_kpi_dashboard(
    sales_performance: dict, lead_funnel: dict, inventory_velocity: dict, collections_risk: dict,
    cancellations: dict, construction_handover: dict, executive_scores: dict,
) -> list[dict[str, Any]]:
    """The ~9 headline dashboard cards -- curated, not exhaustive. Every value copied from a domain engine."""
    ta = sales_performance.get("kpis", {}).get("target_attainment", {})
    sp_summary = sales_performance.get("summary", {})
    lf_summary = lead_funnel.get("summary", {})
    iv_summary = inventory_velocity.get("summary", {})
    cr_summary = collections_risk.get("summary", {})
    cx_summary = cancellations.get("summary", {})
    domains = executive_scores["domains"]

    return [
        {"label": "Net Contracted Sales", "value": sp_summary.get("net_contracted_sales_value"), "unit": "currency",
         "target": ta.get("full_year_annual_sales_target"), "status_ref": "commercial_health"},
        {"label": "Sales Target Attainment", "value": ta.get("sales_target_attainment_pct"), "unit": "percent",
         "target": 100.0, "status_ref": "commercial_health"},
        {"label": "Lead-to-Contract Conversion", "value": lf_summary.get("lead_to_reservation_conversion_pct"), "unit": "percent",
         "target": None, "status_ref": "sales_funnel_health"},
        {"label": "Available Inventory Value", "value": iv_summary.get("available_inventory_value"), "unit": "currency",
         "target": None, "status_ref": "inventory_health"},
        {"label": "Overdue Receivables", "value": cr_summary.get("total_overdue_amount"), "unit": "currency",
         "target": None, "status_ref": "collections_health"},
        {"label": "Cancellation Rate", "value": cx_summary.get("contract_cancellation_rate_pct"), "unit": "percent",
         "target": CONFIG["targets"]["cancellation_rate_ceiling_pct"], "status_ref": "cancellations_health"},
        {"label": "Construction Delivery Health", "value": domains.get("construction_delivery_health", {}).get("score"), "unit": "score",
         "target": None, "status_ref": "construction_delivery_health"},
        {"label": "Handover Readiness", "value": domains.get("handover_readiness_health", {}).get("score"), "unit": "score",
         "target": None, "status_ref": "handover_readiness_health"},
        {"label": "Overall Business Health", "value": executive_scores["overall_business_health"]["score"], "unit": "score",
         "target": None, "status_ref": "overall"},
    ]


# --------------------------------------------------------------------------
# AI context: the curated (not dumped) data the model is allowed to see
# --------------------------------------------------------------------------

def _build_ai_context(skeleton: dict[str, Any]) -> dict[str, Any]:
    """Curate exactly the fields the AI needs to write narrative -- not the full raw domain dicts.

    Two deliberate trims beyond simple curation:

    - recommended_actions is slimmed to title/business_issue/owner/
      horizon/confidence only: its evidence and financial_exposure
      fields duplicate what root_causes already carries in full (every
      action is built 1:1 from a root cause -- see
      analysis/correlations.py::_build_priority_actions), and repeating
      that evidence a second time was the single largest contributor to
      this prompt exceeding the AI provider's tokens-per-minute budget.
    - Every raw currency float that could end up quoted verbatim in
      AI-authored prose (root_causes' financial_exposure/customer_impact,
      financial_exposure's line-item amounts, kpi_dashboard's currency
      values) is either dropped (the already-formatted evidence/
      description text already conveys it in words) or pre-formatted
      into an "EGP 8.4M"-style display string -- the model is instructed
      to copy figures exactly as given, so a raw float like
      1075599619.1 in the context becomes exactly that ugly string in
      the report if it's ever handed one.
    """
    domains = skeleton["domains"]
    slim_actions = [
        {"title": a["title"], "business_issue": a["business_issue"][:220], "owner": a["owner"],
         "horizon": a["horizon"], "confidence": a["confidence"]}
        for a in skeleton["recommended_actions"]
    ]
    slim_root_causes = [
        {"title": rc["title"], "category": rc["category"], "confidence": rc["confidence"],
         "description": rc["description"], "statistical_evidence": rc.get("statistical_evidence", []),
         "operational_evidence": rc.get("operational_evidence", []),
         "owner": rc["owner"], "horizon": rc["horizon"]}
        for rc in skeleton["root_causes"]
    ]
    formatted_financial_exposure = {
        "line_items": [
            {"category": item["category"], "amount_display": _fmt_currency(item["amount"], CURRENCY), "description": item["description"]}
            for item in skeleton["financial_exposure"].get("line_items", [])
        ],
        "total_material_risk_exposure_display": _fmt_currency(skeleton["financial_exposure"].get("total_material_risk_exposure"), CURRENCY),
    }
    formatted_kpi_dashboard = [
        {**card, "value": _fmt_currency(card["value"], CURRENCY) if card["unit"] == "currency" and card["value"] is not None else card["value"]}
        for card in skeleton["kpi_dashboard"]
    ]
    forecast = skeleton["forecast_outlook"]
    formatted_forecast_outlook = dict(forecast)
    for money_key in ("expected_month_end_net_sales", "projected_full_year_net_sales", "expected_next_month_collections", "current_overdue_exposure"):
        if money_key in formatted_forecast_outlook:
            formatted_forecast_outlook[money_key] = _fmt_currency(formatted_forecast_outlook[money_key], CURRENCY)
    return {
        "company_name": COMPANY_NAME,
        "overall_business_health": skeleton["overall_business_health"],
        "kpi_dashboard": formatted_kpi_dashboard,
        "highest_risk_project": skeleton["highest_risk_project"],
        "strongest_project": skeleton["strongest_project"],
        "highest_priority_initiative": skeleton["highest_priority_initiative"],
        "top_business_risks": skeleton["top_business_risks"],
        "root_causes": slim_root_causes,
        "recommended_actions": slim_actions,
        "financial_exposure": formatted_financial_exposure,
        "forecast_outlook": formatted_forecast_outlook,
        "domain_summaries": {
            "commercial_performance": domains["commercial_performance"].get("summary", {}),
            "lead_funnel": domains["lead_funnel"].get("summary", {}),
            "inventory_pricing": domains["inventory_pricing"].get("summary", {}),
            "marketing_performance": domains["marketing_performance"].get("summary", {}),
            "sales_team_broker": domains["sales_team_broker"].get("summary", {}),
            "collections_risk": domains["collections_risk"].get("summary", {}),
            "cancellations": domains["cancellations"].get("summary", {}),
            "construction_handover": domains["construction_handover"].get("summary", {}),
            "customer_experience": domains["customer_experience"].get("summary", {}),
        },
        "data_quality_status": skeleton["data_quality"].get("status"),
    }


# --------------------------------------------------------------------------
# Prompt engineering
# --------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are the Chief Commercial Officer of {COMPANY_NAME}, a multi-project real-estate developer, writing the monthly executive intelligence report for the CEO, board, and department heads (Sales, Marketing, Collections, Construction, Customer Experience). This report is read in under five minutes by people who will act on it. Write the way McKinsey, a top-tier PE portfolio review, or an institutional investment memo would brief a board -- not the way a chatbot summarizes data.

Every number, ranking, root cause, confidence score, and financial figure in the context you are given has already been computed by the company's analytics platform. That platform is deterministic and authoritative. Your job is not to analyze -- it is to communicate what has already been found, and to translate it into business consequences (revenue, cash flow, margin, execution risk, reputation) a board immediately understands.

Rules you must follow without exception:
1. Never perform, estimate, or restate a calculation. Use only the figures given to you, copied exactly as provided (same digits, same units, same project/broker/agent/building names).
2. Never invent a root cause, risk, recommendation, or finding not present in the provided context. If a list is empty, say so plainly instead of manufacturing content.
3. Never fabricate a financial figure or forecast a number not already in the context. Forecasts given to you are pre-computed run-rate estimates, already labeled as such -- do not present them as certainties.
4. Write like a management-consulting executive briefing: short paragraphs (2-4 sentences), plain declarative sentences, varied structure. Do not open sentences with "This indicates," "We should," or "Our overall." Never repeat the phrase "poses a significant risk to the business" -- name the specific financial or operational stake instead (e.g. "EGP 184M of contracted value is tied to units currently forecast for delayed handover").
5. No buzzwords, no chatbot phrasing ("I'd be happy to," "Let's dive in"), no first-person voice, no unnecessary quotation marks around your own interpretation text, never say "AI has discovered" or similar.
6. Distinguish revenue risk, cash-flow risk, margin risk, execution risk, and reputational risk explicitly where the evidence supports it -- do not force-fit a category that isn't grounded in the given evidence.
7. Return ONLY a single JSON object matching the schema you are given. No markdown code fences, no prose outside the JSON, no trailing commentary."""

NARRATIVE_JSON_SCHEMA_EXAMPLE = """{
  "executive_headline": "string, one sentence naming the single most important business issue this period, citing its financial or operational stake",
  "why_it_matters": "string, 1-3 sentences on the business stakes of that issue",
  "consequence_of_inaction": "string, 1-3 sentences, qualitative and/or citing the given financial exposure figures, on what happens if it is not addressed",
  "executive_summary": "string, 3-5 short paragraphs covering: the headline issue, overall business health across domains, the strongest-performing project, and the CCO's top recommendation",
  "overall_business_health_narrative": {
    "commercial_health": "string", "sales_funnel_health": "string", "inventory_health": "string",
    "marketing_efficiency_health": "string", "collections_health": "string", "cancellations_health": "string",
    "construction_delivery_health": "string", "handover_readiness_health": "string", "customer_experience_health": "string"
  },
  "risk_narratives": [
    {"title": "string, must exactly match a title from top_business_risks", "business_impact": "string, executive framing of the financial/operational stake"}
  ],
  "root_cause_notes": [
    {"title": "string, must exactly match a title from root_causes", "executive_note": "string, 1-2 sentences distinguishing measured finding from hypothesis where relevant"}
  ],
  "recommendation_rationale": [
    {"title": "string, must exactly match a title from recommended_actions", "executive_rationale": "string, 1 sentence"}
  ],
  "department_breakdown": {
    "sales": "string", "marketing": "string", "collections": "string", "construction": "string",
    "customer_experience": "string", "executive_leadership": "string"
  },
  "forecast_narrative": "string, 2-3 sentences on outlook, grounded only in the forecast_outlook figures given -- label them as estimates",
  "if_no_action_narrative": "string, 2-3 short paragraphs on the likely business consequences of leaving top_business_risks and root_causes unaddressed"
}"""


def _build_user_prompt(context: dict[str, Any]) -> str:
    context_json = json.dumps(context, indent=2, default=str)
    return f"""Here is this period's real estate operations intelligence, already fully computed by the analytics platform:

```json
{context_json}
```

Write the narrative content for this period's executive report. Return a single JSON object with exactly these keys and shapes:

{NARRATIVE_JSON_SCHEMA_EXAMPLE}

Requirements for each field:
- "executive_headline": names the #1 ranked item in "top_business_risks" (or "highest_priority_initiative" if risks are empty) in one sentence, citing its financial/operational stake if the context provides one.
- "why_it_matters": grounded in "highest_risk_project" and the evidence in "top_business_risks"/"root_causes".
- "consequence_of_inaction": qualitative and/or citing "financial_exposure" figures already in the context. No new numbers, no forecasts beyond what's given.
- "executive_summary": open with the issue in "executive_headline," cover overall business health (reference "overall_business_health"), name "strongest_project" as a positive counterpoint, and state your top recommendation from "recommended_actions".
- "overall_business_health_narrative": one plain-business-terms sentence per domain, using that domain's summary in "domain_summaries" and its score/status in "overall_business_health".
- "risk_narratives": exactly one entry per item in "top_business_risks," matched by "title" exactly.
- "root_cause_notes": exactly one entry per item in "root_causes," matched by "title" exactly. If empty, return an empty list and say so in the executive summary.
- "recommendation_rationale": exactly one entry per item in "recommended_actions," matched by "title" exactly.
- "department_breakdown": one paragraph each for sales, marketing, collections, construction, customer_experience, and executive_leadership -- what that specific audience needs to know from "domain_summaries," "root_causes," and "recommended_actions".
- "forecast_narrative": grounded only in "forecast_outlook" -- if forecast_outlook.available is false, say plainly that insufficient history exists for a forecast this period.
- "if_no_action_narrative": grounded only in "top_business_risks," "root_causes," and "financial_exposure" already given."""


# --------------------------------------------------------------------------
# Response validation
# --------------------------------------------------------------------------

def _validate_narrative_response(narrative: Any) -> dict[str, Any]:
    if not isinstance(narrative, dict):
        raise ResponseValidationError("AI response was not a JSON object.")
    missing = [key for key in REQUIRED_NARRATIVE_KEYS if key not in narrative]
    if missing:
        raise ResponseValidationError(f"AI response is missing required keys: {missing}")

    for text_key in ["executive_headline", "why_it_matters", "consequence_of_inaction", "executive_summary", "forecast_narrative", "if_no_action_narrative"]:
        if not isinstance(narrative.get(text_key), str) or not narrative[text_key].strip():
            raise ResponseValidationError(f"AI response's '{text_key}' is empty or not a string.")

    health_narrative = narrative.get("overall_business_health_narrative")
    if not isinstance(health_narrative, dict):
        raise ResponseValidationError("AI response's overall_business_health_narrative is not an object.")
    missing_domains = [d for d in DOMAIN_LABELS if not isinstance(health_narrative.get(d), str) or not health_narrative[d].strip()]
    if missing_domains:
        raise ResponseValidationError(f"AI response's overall_business_health_narrative is missing content for: {missing_domains}")

    department_breakdown = narrative.get("department_breakdown")
    if not isinstance(department_breakdown, dict):
        raise ResponseValidationError("AI response's department_breakdown is not an object.")
    missing_departments = [
        dept for dept in DEPARTMENTS
        if not isinstance(department_breakdown.get(dept), str) or not department_breakdown[dept].strip()
    ]
    if missing_departments:
        raise ResponseValidationError(f"AI response's department_breakdown is missing content for: {missing_departments}")

    for list_key in ("risk_narratives", "root_cause_notes", "recommendation_rationale"):
        if not isinstance(narrative.get(list_key), list):
            raise ResponseValidationError(f"AI response's '{list_key}' must be a list.")

    return narrative


def _validate_final_report(report: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    if missing:
        raise ResponseValidationError(f"Assembled report is missing required sections: {missing}")
    for text_key in NARRATIVE_TEXT_REPORT_KEYS:
        if not isinstance(report[text_key], str) or not report[text_key].strip():
            raise ResponseValidationError(f"Assembled report has an empty '{text_key}'.")
    forecast_narrative = report.get("forecast_outlook", {}).get("narrative")
    if not isinstance(forecast_narrative, str) or not forecast_narrative.strip():
        raise ResponseValidationError("Assembled report has an empty forecast_outlook.narrative.")
    if not report["overall_business_health"]:
        raise ResponseValidationError("Assembled report has an empty overall_business_health.")
    if not report["department_breakdown"]:
        raise ResponseValidationError("Assembled report has an empty department_breakdown.")
    if not report["appendix"].get("data_limitations"):
        raise ResponseValidationError("Assembled report's appendix is missing data_limitations.")
    try:
        json.dumps(report)
    except (TypeError, ValueError) as exc:
        raise ResponseValidationError(f"Assembled report is not JSON-serializable: {exc}") from exc


# --------------------------------------------------------------------------
# Hallucinated-number heuristic (best-effort, non-fatal, disclosed)
# --------------------------------------------------------------------------

def _extract_numbers(text: str) -> set[str]:
    return {token.replace(",", "") for token in NUMBER_PATTERN.findall(text)}


def _collect_context_numbers(context: dict[str, Any]) -> set[str]:
    numbers: set[str] = set()

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, list):
            for item in value:
                _walk(item)
        elif isinstance(value, (int, float)):
            numbers.add(str(value))
        elif isinstance(value, str):
            numbers.update(_extract_numbers(value))

    _walk(context)
    return numbers


def _find_unverified_numbers(report: dict[str, Any], context_numbers: set[str]) -> list[str]:
    narrative_text = " ".join([
        *(report.get(key, "") for key in NARRATIVE_TEXT_REPORT_KEYS),
        report.get("forecast_outlook", {}).get("narrative", ""),
        " ".join(block.get("narrative", "") for block in report.get("overall_business_health", {}).values()),
        " ".join(risk.get("business_impact", "") for risk in report.get("top_business_risks", [])),
        " ".join(report.get("department_breakdown", {}).values()),
    ])
    found = _extract_numbers(narrative_text)
    return sorted(n for n in found if n not in context_numbers and n not in NUMBER_GROUNDING_IGNORE)


# --------------------------------------------------------------------------
# Merge: narrative text + factual skeleton -> final report
# --------------------------------------------------------------------------

def _merge_narrative_into_skeleton(skeleton: dict[str, Any], narrative: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "metadata": dict(skeleton["metadata"]),
        "executive_headline": narrative["executive_headline"].strip(),
        "why_it_matters": narrative["why_it_matters"].strip(),
        "consequence_of_inaction": narrative["consequence_of_inaction"].strip(),
        "executive_summary": narrative["executive_summary"].strip(),
        "overall_business_health": {},
        "kpi_dashboard": skeleton["kpi_dashboard"],
        "highest_risk_project": skeleton["highest_risk_project"],
        "strongest_project": skeleton["strongest_project"],
        "highest_priority_initiative": skeleton["highest_priority_initiative"],
        "top_business_risks": [],
        "root_causes": [],
        "recommended_actions": [],
        "financial_exposure": skeleton["financial_exposure"],
        "relationships": skeleton["relationships"],
        "domains": skeleton["domains"],
        "data_quality": skeleton["data_quality"],
        "forecast_outlook": {**skeleton["forecast_outlook"], "narrative": narrative["forecast_narrative"].strip()},
        "department_breakdown": dict(narrative["department_breakdown"]),
        "if_no_action_narrative": narrative["if_no_action_narrative"].strip(),
        "appendix": {**skeleton["appendix"], "data_limitations": list(skeleton["appendix"]["data_limitations"])},
    }

    health_narrative = narrative.get("overall_business_health_narrative", {})
    for domain, block in skeleton["overall_business_health"].items():
        report["overall_business_health"][domain] = {
            **block, "narrative": health_narrative.get(domain, "") if isinstance(health_narrative, dict) else "",
        }

    risk_narratives = narrative.get("risk_narratives", [])
    for risk in skeleton["top_business_risks"]:
        report["top_business_risks"].append({**risk, "business_impact": _match_by_title(risk_narratives, risk["title"], "business_impact")})

    root_cause_notes = narrative.get("root_cause_notes", [])
    for rc in skeleton["root_causes"]:
        report["root_causes"].append({**rc, "executive_note": _match_by_title(root_cause_notes, rc["title"], "executive_note")})

    rationale_notes = narrative.get("recommendation_rationale", [])
    for action in skeleton["recommended_actions"]:
        report["recommended_actions"].append({**action, "executive_rationale": _match_by_title(rationale_notes, action["title"], "executive_rationale")})

    return report


# --------------------------------------------------------------------------
# Markdown rendering (deterministic -- not a second AI call)
# --------------------------------------------------------------------------

def _label(key: str) -> str:
    return key.replace("_", " ").title()


def _fmt_score(score: float | None) -> str:
    return "N/A" if score is None else str(score)


def _fmt_confidence(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0%}"


def _fmt_currency_md(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{CURRENCY} {value:,.0f}"


def _render_trend_cell(trend: dict[str, Any]) -> str:
    if not trend.get("available"):
        return "No historical comparison available"
    direction = trend.get("direction")
    if direction == "flat":
        return "Flat vs. last report"
    arrow = "Up" if direction == "up" else "Down"
    return f"{arrow} {trend['delta']:+.1f} pts vs. last report"


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the final report dict to professionally formatted Markdown. Deterministic, not a second AI call."""
    metadata = report["metadata"]
    divider = "---"
    lines: list[str] = [
        f"# {metadata['report_type']}",
        f"*{metadata['platform']} for {metadata['company_name']} -- Generated {metadata['generated_at']}*",
        "", divider, "",
        "## Executive Alert", "",
        f"> **{report['executive_headline']}**", ">", f"> {report['why_it_matters']}", ">",
        f"> **If nothing changes:** {report['consequence_of_inaction']}", "",
        divider, "",
        "## Executive Dashboard", "",
        "| Metric | Value | Target |", "|---|---|---|",
    ]
    for card in report["kpi_dashboard"]:
        value = _fmt_currency_md(card["value"]) if card["unit"] == "currency" else (
            f"{card['value']}%" if card["unit"] == "percent" and card["value"] is not None else _fmt_score(card["value"])
        )
        target = (_fmt_currency_md(card["target"]) if card["unit"] == "currency" else f"{card['target']}%") if card.get("target") else "--"
        lines.append(f"| {card['label']} | {value} | {target} |")
    lines.extend(["", f"**Highest Risk Project:** {report['highest_risk_project'] or 'Not identified this period.'}",
                  "", f"**Strongest Performing Project:** {report['strongest_project'] or 'Not identified this period.'}",
                  "", f"**Highest Priority Initiative:** {report['highest_priority_initiative'] or 'None identified this period.'}", "",
                  divider, "", "## Executive Summary", "", report["executive_summary"], ""])

    for domain, block in report["overall_business_health"].items():
        if block.get("narrative"):
            lines.append(f"**{_label(domain)}:** {block['score']}/100 ({block['status']}) -- {block['narrative']}")
    lines.extend(["", divider, "", "## Cross-Functional Root Causes", ""])
    if not report["root_causes"]:
        lines.append("_No cross-domain root cause met the platform's evidence threshold this period._")
    for rc in report["root_causes"]:
        lines.append(f"### {rc['title']} (confidence: {_fmt_confidence(rc['confidence'])})")
        if rc.get("executive_note"):
            lines.append(rc["executive_note"])
        lines.append("")
        lines.append(f"**Owner:** {rc['owner']} &middot; **Horizon:** {rc['horizon']}")
        lines.append("")

    lines.extend([divider, "", "## Financial Exposure", ""])
    for item in report["financial_exposure"].get("line_items", []):
        lines.append(f"- **{item['category']}:** {_fmt_currency_md(item['amount'])} -- {item['description']}")
    lines.append("")

    lines.extend([divider, "", "## 90-Day Action Plan", ""])
    for action in report["recommended_actions"]:
        lines.append(f"**P{action['priority']}. {action['title']}** ({action['horizon']})")
        lines.append(f"Owner: {action['owner']} &middot; Difficulty: {action['difficulty']} &middot; Confidence: {_fmt_confidence(action['confidence'])}")
        lines.append("")
        lines.append(action.get("business_issue", ""))
        if action.get("executive_rationale"):
            lines.append(action["executive_rationale"])
        lines.append(f"**Recommended action:** {action.get('recommended_action', '')}")
        lines.append("")

    lines.extend([divider, "", "## Department Accountability", ""])
    for department, text in report["department_breakdown"].items():
        lines.append(f"### {_label(department)}")
        lines.append(text)
        lines.append("")

    lines.extend([divider, "", "## Forecast & Outlook", "", report["forecast_outlook"].get("narrative", ""), ""])
    lines.extend([divider, "", "## If No Action Is Taken", "", report["if_no_action_narrative"], ""])

    appendix = report["appendix"]
    lines.extend([divider, "", "## Appendix", "", "### Methodology", appendix.get("methodology_summary", ""), ""])
    lines.append("### Confidence Methodology")
    lines.append(appendix.get("confidence_explanation", ""))
    lines.append("")
    dq = appendix.get("data_quality_summary", {})
    lines.append(f"### Data Quality: {dq.get('score')}/100 ({dq.get('status')})")
    lines.append("")
    lines.append("### Data Limitations")
    for limitation in appendix.get("data_limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append(f"*Generated by {appendix.get('generated_by', '')}*")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def _save_report(report: dict[str, Any], markdown: str, html: str, output_dir: Path, json_path: Path, md_path: Path, html_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def generate_executive_report(
    sales_performance: dict[str, Any],
    lead_funnel: dict[str, Any],
    inventory_velocity: dict[str, Any],
    marketing_efficiency: dict[str, Any],
    sales_team_broker_performance: dict[str, Any],
    collections_risk: dict[str, Any],
    cancellations: dict[str, Any],
    construction_handover: dict[str, Any],
    customer_experience: dict[str, Any],
    data_quality: dict[str, Any],
    executive_scores: dict[str, Any],
    correlation_analysis: dict[str, Any],
    *,
    provider: AIProvider | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Turn nine completed real-estate analytics outputs (+ scoring/correlation layers) into a persisted executive report.

    Pipeline: validate inputs -> build a fully-computed report skeleton
    in Python -> curate an AI context -> call the AI provider for
    narrative text only -> validate that response -> merge narrative
    into the skeleton -> validate the assembled report -> render
    Markdown and HTML (both pure functions over the same report dict,
    no further AI calls) -> persist JSON, Markdown, and HTML to
    `output_dir` -> return the report dict.

    Args:
        sales_performance, lead_funnel, inventory_velocity,
            marketing_efficiency, sales_team_broker_performance,
            collections_risk, cancellations, construction_handover,
            customer_experience: outputs of the corresponding
            analysis.* module.
        data_quality: output of analysis.data_quality.analyze_data_quality.
        executive_scores: output of analysis.executive_scoring.compute_executive_scores.
        correlation_analysis: output of analysis.correlations.analyze_correlations.
        provider: The AI backend to use. Defaults to `GroqProvider()`.
        output_dir: Directory to write executive_report.json/.md/.html
            into. Defaults to `outputs/` at the project root.

    Returns:
        The final report dict.

    Raises:
        AnalyticsInputError: If any analytics input is missing or malformed.
        ConfigurationError: If required AI provider configuration is missing.
        AIProviderError: If the AI provider fails after retries are exhausted.
        ResponseValidationError: If the AI's response, or the final
            assembled report, fails structural validation.
    """
    started_at = datetime.now(timezone.utc)

    sales_performance = _require(sales_performance, "sales_performance")
    lead_funnel = _require(lead_funnel, "lead_funnel")
    inventory_velocity = _require(inventory_velocity, "inventory_velocity")
    marketing_efficiency = _require(marketing_efficiency, "marketing_efficiency")
    sales_team_broker_performance = _require(sales_team_broker_performance, "sales_team_broker_performance")
    collections_risk = _require(collections_risk, "collections_risk")
    cancellations = _require(cancellations, "cancellations")
    construction_handover = _require(construction_handover, "construction_handover")
    customer_experience = _require(customer_experience, "customer_experience")
    data_quality = _require(data_quality, "data_quality")
    executive_scores = _require(executive_scores, "executive_scores")
    correlation_analysis = _require(correlation_analysis, "correlation_analysis")

    provider = provider or GroqProvider()

    output_directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    json_path = output_directory / REPORT_JSON_FILENAME
    md_path = output_directory / REPORT_MARKDOWN_FILENAME
    html_path = output_directory / REPORT_HTML_FILENAME
    previous_report = _load_previous_report(json_path)

    skeleton = _build_report_skeleton(
        sales_performance, lead_funnel, inventory_velocity, marketing_efficiency,
        sales_team_broker_performance, collections_risk, cancellations, construction_handover,
        customer_experience, data_quality, executive_scores, correlation_analysis,
        provider.model_name, started_at, previous_report,
    )
    context = _build_ai_context(skeleton)

    raw_response = provider.generate_json(SYSTEM_PROMPT, _build_user_prompt(context))
    try:
        narrative = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"AI response was not valid JSON: {exc}") from exc
    narrative = _validate_narrative_response(narrative)

    report = _merge_narrative_into_skeleton(skeleton, narrative)

    unverified_numbers = _find_unverified_numbers(report, _collect_context_numbers(context))
    if unverified_numbers:
        report["appendix"]["data_limitations"].append(
            "The following figures appear in this report's narrative text but could not be "
            f"automatically verified against source analytics; spot-check before distribution: {unverified_numbers}."
        )

    report["metadata"]["generation_duration_seconds"] = round((datetime.now(timezone.utc) - started_at).total_seconds(), 2)
    report["metadata"]["saved_json_path"] = str(json_path)
    report["metadata"]["saved_markdown_path"] = str(md_path)
    report["metadata"]["saved_html_path"] = str(html_path)

    _validate_final_report(report)

    markdown = _render_markdown(report)
    try:
        html = render_html(report)
    except HTMLRenderError as exc:
        raise ResponseValidationError(f"Failed to render the HTML report: {exc}") from exc
    _save_report(report, markdown, html, output_directory, json_path, md_path, html_path)

    return report
