"""AI Orchestration Layer: turns computed analytics into an executive report.

This is the fourth and final layer of the platform:

    CSV Data -> Analytics Layer -> Business Intelligence Layer -> AI Orchestration Layer

The three analytics engines (analysis/delivery.py, analysis/complaints.py,
analysis/inventory.py) and the correlation engine (analysis/correlations.py)
have already done every calculation this platform performs: every rate,
every ranking, every correlation coefficient, every confidence score,
every root cause and priority action. Nothing in this module recomputes
any of that. This module's only job is communication -- taking findings
that already exist and writing them up the way a Chief Operations
Officer would brief company leadership.

Why the AI performs communication, not analytics
--------------------------------------------------
An LLM is unreliable at arithmetic and prone to quietly restating a
number wrong, rounding differently, or inventing a plausible-sounding
figure that was never computed. A deterministic pandas pipeline is not.
So the division of labor here is architectural, not just a prompt
instruction: this module builds a "report skeleton" in pure Python
first -- every number, ranking, warehouse name, confidence score, and
piece of evidence that appears in the final report is copied directly
from the four analytics dicts before the AI is ever called. The single
Groq call that follows is asked to return *only* narrative text (an
executive summary, one-sentence framings, department briefings) keyed
to match the skeleton, which Python then merges in. The model
literally cannot corrupt a number, because no number ever depends on
the model reproducing it correctly.

Why JSON is the source of truth
--------------------------------
Every analytics engine already emits a hierarchical, JSON-serializable
dict as its contract with the rest of the platform. This module keeps
that contract all the way through: the AI is asked to return JSON (not
prose or markdown) constrained to a fixed schema, so the merge step is
a matter of dict lookups, not text parsing; the final report is a JSON
dict validated the same way the analytics engines validate their own
output; and Markdown is a *rendering* of that JSON, generated
deterministically in Python, not a second thing the model has to get
right independently. The JSON file persisted to outputs/ is the
platform's durable, API-and-dashboard-ready artifact; the Markdown
file is one presentation of it.

Why structured prompts and a provider abstraction
----------------------------------------------------
The system prompt fixes the persona (a COO writing a weekly briefing,
not a chatbot) and a small set of hard rules (never calculate, never
invent, never fabricate a number, match the given entity names
exactly, return only JSON). The user prompt supplies a curated -- not
dumped -- context: only the fields a report actually cites, so the
model isn't tempted to editorialize on data it wasn't asked to
discuss. All of this lives behind the `AIProvider` interface
(`generate_json(system_prompt, user_prompt) -> str`); `GroqProvider`
is the only concrete implementation today, but nothing in
`generate_executive_report` imports the `groq` package directly, so
adding a second provider (Anthropic, a local model, a different
OpenAI-compatible endpoint) means implementing that one interface, not
touching orchestration, prompting, merging, validation, or rendering.
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
from dotenv import load_dotenv
from groq import Groq

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PLATFORM_NAME = "Solven Operations Intelligence Platform"
REPORT_TYPE = "Weekly Executive Operations Report"

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUTPUT_TOKENS = 4000
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
REPORT_JSON_FILENAME = "executive_report.json"
REPORT_MARKDOWN_FILENAME = "executive_report.md"

# How many candidates Python selects (deterministically, by severity/
# confidence already computed upstream) before asking the AI to write
# narrative for them.
TOP_RISK_COUNT = 3
TOP_RELATIONSHIP_COUNT = 5

# score >= HIGH -> "Healthy"; >= LOW -> "At Risk"; below LOW -> "Critical".
HEALTH_STATUS_THRESHOLDS = (60.0, 80.0)  # (at_risk_floor, healthy_floor)

URGENCY_BY_SEVERITY = {"HIGH": "Immediate", "MEDIUM": "This Quarter", "LOW": "Monitor"}

REQUIRED_DEPARTMENTS = ["operations", "logistics", "customer_experience", "supply_chain", "executive_leadership"]

# Presentation-layer lookups below (health icons, action ownership, action
# horizons). These are display metadata, not analytics: they classify
# values the analytics engines already computed -- they never compute a
# score, rate, or confidence figure of their own.

HEALTH_STATUS_ICON = {"Healthy": "\U0001F7E2", "At Risk": "\U0001F7E1", "Critical": "\U0001F534", "Unknown": "⚪"}
SEVERITY_ICON = {"HIGH": "\U0001F534", "MEDIUM": "\U0001F7E0", "LOW": "\U0001F7E1"}

# Which department a recommended action lands on, keyed by the same root-
# cause category vocabulary analysis/correlations.py already assigns
# (Warehouse Bottleneck / Supplier Risk / Process Design). Unrecognized or
# future categories fall back to DEFAULT_OWNER rather than raising.
OWNER_BY_ROOT_CAUSE_CATEGORY = {
    "Warehouse Bottleneck": "Operations & Logistics",
    "Supplier Risk": "Supply Chain & Procurement",
    "Process Design": "Operations Leadership",
}
DEFAULT_OWNER = "Executive Leadership"

# Buckets an action's existing estimated_timeframe ("14-30 days", "30-60
# days", "60-120 days") into an executive-readable horizon, by the leading
# day count. (exclusive_day_ceiling, label) evaluated in order with a
# strict "<" so the three known, contiguous ranges never collide at their
# shared boundary (e.g. "60-120 days" lands in Strategic, not Near-Term,
# even though "30-60 days" also mentions 60). The last entry is the
# catch-all for anything longer or unparsed.
ACTION_HORIZONS = [
    (30, "Immediate (0-30 Days)"),
    (60, "Near-Term (30-60 Days)"),
    (float("inf"), "Strategic (60-120 Days)"),
]

# Priority order used to break ties when more than one warehouse is
# equally implicated across delivery, customer experience, and inventory.
RISK_LOCATION_SOURCES = ("delivery", "customer_experience", "inventory")

# Columns/keys each analytics engine's output must have for this module
# to trust it. delivery_analysis keeps the flatter shape analysis/delivery.py
# has always returned; complaints/inventory/correlations share the newer
# {summary/kpis/rankings/...} or {executive_summary/root_causes/...} shapes.
REQUIRED_DELIVERY_KEYS = ["executive_kpis", "warehouse_intelligence", "anomalies", "operations_health_score"]
REQUIRED_COMPLAINTS_KEYS = ["summary", "rankings", "anomalies", "health_score"]
REQUIRED_INVENTORY_KEYS = ["summary", "rankings", "anomalies", "health_score"]
REQUIRED_CORRELATION_KEYS = ["executive_summary", "root_causes", "business_impacts", "priority_actions", "confidence_scores", "methodology"]

REQUIRED_NARRATIVE_KEYS = [
    "executive_headline", "why_it_matters", "consequence_of_inaction", "executive_summary",
    "overall_business_health_narrative", "risk_narratives", "root_cause_notes",
    "recommendation_rationale", "expected_business_impact", "department_breakdown", "if_no_action_narrative",
]
REQUIRED_REPORT_KEYS = [
    "metadata", "executive_headline", "why_it_matters", "consequence_of_inaction", "executive_summary",
    "overall_business_health", "highest_risk_location", "highest_priority_initiative", "top_business_risks",
    "root_causes", "recommended_actions", "expected_business_impact", "department_breakdown",
    "if_no_action_narrative", "appendix",
]

# AI-authored free-text fields walked by the hallucinated-number heuristic
# (_find_unverified_numbers). Kept as one list so every new narrative field
# added to the schema is grounding-checked by construction, not by memory.
NARRATIVE_TEXT_REPORT_KEYS = ["executive_headline", "why_it_matters", "consequence_of_inaction", "executive_summary", "if_no_action_narrative"]

# Numbers this small are almost always a rank, a count of items, or a
# list index in narrative prose ("the top 3 risks"), not a metric --
# excluded from the hallucinated-number heuristic to avoid false positives.
NUMBER_GROUNDING_IGNORE = {"1", "2", "3", "4", "5"}
NUMBER_PATTERN = re.compile(r"\d[\d,]*\.?\d*%?")


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class ReportGenerationError(Exception):
    """Base class for every error this module raises. Never fails silently."""


class ConfigurationError(ReportGenerationError):
    """Required configuration (an environment variable, typically) is missing or invalid."""


class AnalyticsInputError(ReportGenerationError):
    """One of the four analytics inputs is missing, empty, or malformed."""


class AIProviderError(ReportGenerationError):
    """The AI provider failed to produce a usable response (auth, network, rate limit, timeout)."""


class ResponseValidationError(ReportGenerationError):
    """The AI's response, or the assembled report, failed structural validation."""


# --------------------------------------------------------------------------
# AI provider abstraction
# --------------------------------------------------------------------------

class AIProvider(ABC):
    """Interface every AI backend must implement to plug into this module.

    Deliberately minimal: "given a system prompt and a user prompt,
    return a JSON string." Nothing about prompt construction, response
    merging, or validation lives here or depends on any one vendor's
    SDK -- only `GroqProvider` below imports `groq`.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """A human-readable identifier for the model in use, recorded in report metadata."""
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        """Return a raw JSON string produced by the model for the given prompts."""
        raise NotImplementedError


@dataclass(frozen=True)
class AIConfig:
    """Groq configuration, read entirely from environment variables. Never hardcoded."""

    api_key: str
    model: str
    timeout_seconds: float
    max_retries: int
    temperature: float
    max_output_tokens: int

    @classmethod
    def from_env(cls) -> "AIConfig":
        """Load configuration from the environment (via a .env file if present).

        Raises:
            ConfigurationError: If GROQ_API_KEY is not set.
        """
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
    """Read a numeric environment variable, falling back to `default` if unset or invalid."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _call_with_retry(call: Callable[[], str], max_retries: int) -> str:
    """Retry transient Groq failures with exponential backoff; fail fast on non-retryable errors.

    Authentication and malformed-request errors are never retried --
    retrying them cannot succeed. Rate limits, timeouts, connection
    errors, and 5xx server errors are retried up to `max_retries`
    times with exponential backoff before giving up.
    """
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
    """Default AI provider: Groq chat completions in JSON mode.

    Configuration comes entirely from AIConfig (i.e. entirely from
    environment variables) -- no API key or model name is ever
    hardcoded here.
    """

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

def _require_analysis(analysis: Any, required_keys: list[str], name: str) -> dict[str, Any]:
    """Validate that an upstream engine's output is a usable dict before reading from it."""
    if analysis is None or not isinstance(analysis, dict):
        raise AnalyticsInputError(f"generate_executive_report: {name} must be a non-empty dict")
    missing = [k for k in required_keys if k not in analysis]
    if missing:
        raise AnalyticsInputError(f"generate_executive_report: {name} is missing expected keys: {missing}")
    return analysis


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------

def _status_from_score(score: float | None) -> str:
    """Classify a 0-100 health score into a plain-language status. Never AI-derived."""
    if score is None:
        return "Unknown"
    at_risk_floor, healthy_floor = HEALTH_STATUS_THRESHOLDS
    if score >= healthy_floor:
        return "Healthy"
    if score >= at_risk_floor:
        return "At Risk"
    return "Critical"


def _top_items(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """First `n` items from an already-sorted (by the source engine) list."""
    return items[:n] if items else []


def _match_by_title(items: list[dict[str, Any]], title: str, key: str) -> str:
    """Look up a narrative field by matching "title" -- graceful fallback if the AI omitted it."""
    for item in items:
        if isinstance(item, dict) and item.get("title") == title:
            value = item.get(key, "")
            return value if isinstance(value, str) else ""
    return ""


def _action_owner(root_cause_category: str | None) -> str:
    """Map a root-cause category (already computed upstream) to an owning department."""
    return OWNER_BY_ROOT_CAUSE_CATEGORY.get(root_cause_category or "", DEFAULT_OWNER)


def _action_horizon(estimated_timeframe: str | None) -> str:
    """Bucket an action's existing estimated_timeframe string into an executive-readable horizon."""
    match = re.search(r"\d+", estimated_timeframe or "")
    leading_days = int(match.group()) if match else None
    if leading_days is None:
        return ACTION_HORIZONS[-1][1]
    for ceiling, label in ACTION_HORIZONS:
        if leading_days < ceiling:
            return label
    return ACTION_HORIZONS[-1][1]


def _highest_risk_location(
    delivery_analysis: dict[str, Any], complaints_analysis: dict[str, Any], inventory_analysis: dict[str, Any]
) -> str | None:
    """The warehouse implicated as worst-performing by the most analytics engines.

    Reads only warehouse names each engine already ranked worst/highest-risk
    -- no new comparison, scoring, or statistic is computed here. Ties are
    broken by RISK_LOCATION_SOURCES order (delivery, then customer
    experience, then inventory).
    """
    candidates = {
        "delivery": delivery_analysis.get("warehouse_intelligence", {}).get("worst_warehouse"),
        "customer_experience": complaints_analysis.get("rankings", {}).get("warehouse", {}).get("worst_warehouse"),
        "inventory": inventory_analysis.get("rankings", {}).get("warehouse", {}).get("highest_risk_warehouse"),
    }
    named = {source: warehouse for source, warehouse in candidates.items() if warehouse}
    if not named:
        return None

    counts: dict[str, int] = {}
    for warehouse in named.values():
        counts[warehouse] = counts.get(warehouse, 0) + 1
    max_count = max(counts.values())
    tied = {warehouse for warehouse, count in counts.items() if count == max_count}
    for source in RISK_LOCATION_SOURCES:
        if named.get(source) in tied:
            return named[source]
    return next(iter(tied))


def _trend(previous_score: float | None, current_score: float | None) -> dict[str, Any]:
    """Compare a health score to the same score from the last persisted report, if any.

    Purely a diff of two numbers the analytics engines already computed on
    two separate runs -- no forecasting, no new metric. `available` is
    False whenever there is no prior report or the prior report didn't
    carry this domain, in which case the caller shows "No historical
    comparison available" rather than a synthesized trend.
    """
    if previous_score is None or current_score is None:
        return {"available": False, "direction": None, "delta": None}
    delta = round(current_score - previous_score, 1)
    if delta > 0:
        direction = "up"
    elif delta < 0:
        direction = "down"
    else:
        direction = "flat"
    return {"available": True, "direction": direction, "delta": delta}


def _load_previous_report(json_path: Path) -> dict[str, Any] | None:
    """Load the last executive report this module persisted, for trend comparison.

    Returns None (not an error) if no prior report exists yet or it can't
    be parsed -- a missing history is the expected steady state for a
    brand-new deployment, not a failure.
    """
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------
# Report skeleton: every number and fact, built entirely in Python
# --------------------------------------------------------------------------

def _build_appendix(
    complaints_analysis: dict[str, Any], inventory_analysis: dict[str, Any], correlation_analysis: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the appendix entirely from the engines' own methodology fields. No AI involved.

    This is deliberately the one section with zero AI-generated
    content: it is where an executive would look to audit the report,
    so it is built exclusively from text the analytics engines already
    wrote about themselves.
    """
    limitations: list[str] = []
    for source_name, analysis in (("complaints", complaints_analysis), ("inventory", inventory_analysis), ("correlations", correlation_analysis)):
        methodology = analysis.get("methodology", {})
        if not isinstance(methodology, dict):
            continue
        for value in methodology.get("limitations", []) or []:
            limitations.append(f"[{source_name}] {value}")
        for value in methodology.get("assumptions", []) or []:
            limitations.append(f"[{source_name} assumption] {value}")

    limitations.append(
        "This executive report's narrative text is generated by an LLM from the structured "
        "analytics above; every number, ranking, and entity name in this report is sourced "
        "directly from Python-computed analytics and is never generated, recalculated, or "
        "altered by the AI."
    )

    return {
        "methodology_summary": (
            "Delivery, customer experience, and inventory metrics are computed independently by "
            "dedicated analytics engines from operational data (shipments, complaints, and "
            "inventory snapshots). The correlation layer cross-references those three engines' "
            "own outputs to identify root causes and prioritized actions using descriptive "
            "statistics (Pearson correlation, z-scores, convergence counting across independent "
            "engines) -- never machine learning or forecasting. This report's sole role is to "
            "communicate those pre-computed findings in executive language; it performs no "
            "analysis of its own."
        ),
        "confidence_explanation": correlation_analysis.get("methodology", {}).get(
            "confidence_methodology", "Confidence methodology not available from the correlation engine."
        ),
        "data_limitations": limitations,
        "generated_by": f"{PLATFORM_NAME} AI Orchestration Layer",
    }


def _build_report_skeleton(
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
    correlation_analysis: dict[str, Any],
    model_name: str,
    generation_started_at: datetime,
    previous_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the report's numeric/factual skeleton -- everything except narrative prose.

    Every value here is copied or directly derived from the four
    analytics dicts (plus, for `trend`, the previous persisted report).
    Narrative string fields (executive_summary, per-item "why it
    matters" framing, department paragraphs, etc.) are intentionally
    absent; `_merge_narrative_into_skeleton` adds them after the AI call.
    """
    delivery_health = delivery_analysis.get("operations_health_score", {})
    complaints_health = complaints_analysis.get("health_score", {})
    inventory_health = inventory_analysis.get("health_score", {})

    scores = {
        "delivery": delivery_health.get("overall_score"),
        "customer_experience": complaints_health.get("overall_score"),
        "inventory": inventory_health.get("overall_score"),
    }
    valid_scores = [s for s in scores.values() if s is not None]
    scores["operations_overall"] = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else None

    previous_health = (previous_report or {}).get("overall_business_health", {})
    overall_business_health = {
        domain: {
            "score": score,
            "status": _status_from_score(score),
            "trend": _trend(previous_health.get(domain, {}).get("score"), score),
        }
        for domain, score in scores.items()
    }

    business_impacts = correlation_analysis.get("business_impacts", [])
    top_business_risks = [
        {
            "rank": index + 1,
            "title": impact["impact"],
            "severity": impact["severity"],
            "urgency": URGENCY_BY_SEVERITY.get(impact["severity"], "Monitor"),
            "evidence": impact["evidence"],
        }
        for index, impact in enumerate(_top_items(business_impacts, TOP_RISK_COUNT))
    ]

    root_causes = [
        {
            "title": root_cause["title"],
            "category": root_cause.get("category"),
            "confidence": root_cause["confidence"],
            "evidence": root_cause.get("evidence", []),
            "business_impact": root_cause.get("business_impact"),
        }
        for root_cause in correlation_analysis.get("root_causes", [])
    ]
    category_by_title = {root_cause["title"]: root_cause["category"] for root_cause in root_causes}

    recommended_actions = [
        {
            "priority": action["priority"],
            "title": action["title"],
            "reason": action["reason"],
            "expected_business_impact": action["expected_business_impact"],
            "difficulty": action["difficulty"],
            "estimated_timeframe": action["estimated_timeframe"],
            "confidence": action["confidence"],
            "owner": _action_owner(category_by_title.get(action["title"])),
            "horizon": _action_horizon(action["estimated_timeframe"]),
        }
        for action in correlation_analysis.get("priority_actions", [])
    ]

    metadata = {
        "report_id": str(uuid.uuid4()),
        "generated_at": generation_started_at.isoformat(),
        "platform": PLATFORM_NAME,
        "report_type": REPORT_TYPE,
        "ai_model": model_name,
        "data_sources": ["delivery_analysis", "complaints_analysis", "inventory_analysis", "correlation_analysis"],
    }

    return {
        "metadata": metadata,
        "overall_business_health": overall_business_health,
        "highest_risk_location": _highest_risk_location(delivery_analysis, complaints_analysis, inventory_analysis),
        "highest_priority_initiative": recommended_actions[0]["title"] if recommended_actions else None,
        "top_business_risks": top_business_risks,
        "root_causes": root_causes,
        "recommended_actions": recommended_actions,
        "appendix": _build_appendix(complaints_analysis, inventory_analysis, correlation_analysis),
    }


# --------------------------------------------------------------------------
# AI context: the curated (not dumped) data the model is allowed to see
# --------------------------------------------------------------------------

def _build_ai_context(
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
    correlation_analysis: dict[str, Any],
    skeleton: dict[str, Any],
) -> dict[str, Any]:
    """Curate exactly the fields the AI needs to write narrative -- not the full raw dicts.

    Passing the complete nested output of all four engines (which
    includes, e.g., 300 individual product rows) would waste tokens
    and invite the model to comment on data no report section actually
    covers. This selects only what each report section is grounded in.
    """
    delivery_warehouses = delivery_analysis.get("warehouse_intelligence", {})
    complaints_warehouses = complaints_analysis.get("rankings", {}).get("warehouse", {})
    inventory_warehouses = inventory_analysis.get("rankings", {}).get("warehouse", {})

    return {
        "overall_business_health": skeleton["overall_business_health"],
        "highest_risk_location": skeleton["highest_risk_location"],
        "highest_priority_initiative": skeleton["highest_priority_initiative"],
        "delivery": {
            "kpis": delivery_analysis.get("executive_kpis", {}),
            "best_warehouse": delivery_warehouses.get("best_warehouse"),
            "worst_warehouse": delivery_warehouses.get("worst_warehouse"),
            "top_anomalies": _top_items(delivery_analysis.get("anomalies", []), TOP_RISK_COUNT),
        },
        "customer_experience": {
            "summary": complaints_analysis.get("summary", {}),
            "best_warehouse": complaints_warehouses.get("best_warehouse"),
            "worst_warehouse": complaints_warehouses.get("worst_warehouse"),
            "top_anomalies": _top_items(complaints_analysis.get("anomalies", []), TOP_RISK_COUNT),
        },
        "inventory": {
            "summary": inventory_analysis.get("summary", {}),
            "best_warehouse": inventory_warehouses.get("best_warehouse"),
            "highest_risk_warehouse": inventory_warehouses.get("highest_risk_warehouse"),
            "top_anomalies": _top_items(inventory_analysis.get("anomalies", []), TOP_RISK_COUNT),
        },
        "correlations": {
            "executive_summary": correlation_analysis.get("executive_summary", {}),
            "relationships": _top_items(correlation_analysis.get("relationships", []), TOP_RELATIONSHIP_COUNT),
        },
        "top_business_risks": skeleton["top_business_risks"],
        "root_causes": skeleton["root_causes"],
        "recommended_actions": skeleton["recommended_actions"],
    }


# --------------------------------------------------------------------------
# Prompt engineering
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Chief Operations Officer of a logistics company, writing the weekly executive operations report for the CEO, the executive leadership team, and department heads. This report is read Monday morning, in under five minutes, by people who will act on it. Write the way McKinsey, Amazon Operations, or Palantir Foundry would brief a COO -- not the way a chatbot summarizes data.

Every number, ranking, root cause, and recommendation in the context you are given has already been computed by the company's analytics platform. That platform is deterministic and authoritative: it has already run the statistics, correlations, and confidence scoring. Your job is not to analyze -- it is to communicate what has already been found, and to translate it into business consequences (revenue, customers, cost, reputation) an executive immediately understands.

Rules you must follow without exception:
1. Never perform, estimate, or restate a calculation. Use only the figures given to you, copied exactly as provided (same digits, same units, same warehouse/supplier/category names).
2. Never invent a root cause, risk, recommendation, or finding that is not present in the provided context. If a list in the context is empty, say so plainly instead of manufacturing content to fill space.
3. Never fabricate a dollar figure, percentage, or ROI estimate, and never forecast a future number. If the context does not state a number, describe the outcome in qualitative business terms only -- e.g. "elevated churn risk among high-volume accounts," not "a 12% improvement" or "an estimated $2M exposure" unless that figure already appears in the context.
4. Write like a management-consulting executive briefing: short paragraphs (2-4 sentences), plain declarative sentences, varied sentence structure. Do not open sentences with "This indicates," "We should," or "Our overall." Do not use the phrase "business health is at risk" or any other generic hedge -- name the specific issue instead. No buzzwords ("synergy," "leverage," "game-changing"), no chatbot phrasing ("I'd be happy to," "Great question," "Let's dive in").
5. Every business-impact sentence should read like a consultant naming the stakes -- customer churn, SLA penalties, margin/revenue exposure, carrying cost, warehouse throughput, reputational risk -- using only the categories the evidence actually supports. Do not force-fit a category that isn't grounded in the given evidence.
6. Return ONLY a single JSON object matching the schema you are given. No markdown code fences, no prose outside the JSON, no trailing commentary."""

NARRATIVE_JSON_SCHEMA_EXAMPLE = """{
  "executive_headline": "string, one sentence naming the single most important operational issue this period",
  "why_it_matters": "string, 1-3 sentences on the business stakes of that issue",
  "consequence_of_inaction": "string, 1-3 sentences, qualitative, on what happens if it is not addressed",
  "executive_summary": "string, 2-4 short paragraphs",
  "overall_business_health_narrative": {
    "delivery": "string", "customer_experience": "string", "inventory": "string", "operations_overall": "string"
  },
  "risk_narratives": [
    {"title": "string, must exactly match a title from top_business_risks", "business_impact": "string, executive business-impact framing (churn, SLA, revenue, cost, reputation)"}
  ],
  "root_cause_notes": [
    {"title": "string, must exactly match a title from root_causes", "executive_note": "string"}
  ],
  "recommendation_rationale": [
    {"title": "string, must exactly match a title from recommended_actions", "executive_rationale": "string"}
  ],
  "expected_business_impact": [
    {"area": "string", "expected_improvement": "string, qualitative only"}
  ],
  "department_breakdown": {
    "operations": "string", "logistics": "string", "customer_experience": "string",
    "supply_chain": "string", "executive_leadership": "string"
  },
  "if_no_action_narrative": "string, 2-3 short paragraphs on the likely operational consequences of leaving the current top risks and root causes unaddressed"
}"""


def _build_user_prompt(context: dict[str, Any]) -> str:
    """Build the user-turn prompt: the curated context plus exact output requirements."""
    context_json = json.dumps(context, indent=2, default=str)
    return f"""Here is this period's operations intelligence, already fully computed by the analytics platform:

```json
{context_json}
```

Write the narrative content for this week's executive report. Return a single JSON object with exactly these keys and shapes:

{NARRATIVE_JSON_SCHEMA_EXAMPLE}

Requirements for each field:
- "executive_headline": names the #1 ranked item in "top_business_risks" (or the top "highest_priority_initiative" if there are no risks) in a single sentence that would stop a CEO mid-scroll. No throat-clearing.
- "why_it_matters": grounded in "highest_risk_location" and the evidence already in "top_business_risks" / "root_causes" -- why this specific issue is a business problem, not just an operational one.
- "consequence_of_inaction": qualitative only, grounded in "business_impact" / "expected_business_impact" text already present in root_causes and recommended_actions. No new numbers, no forecasts.
- "executive_summary": open with the same issue named in "executive_headline," then cover overall business health, the single biggest opportunity, and your recommendation as COO -- grounded entirely in the context above. Do not restate "executive_headline" verbatim; build on it.
- "overall_business_health_narrative": one plain-business-terms sentence per domain explaining what that score means.
- "risk_narratives": exactly one entry per item in "top_business_risks" above, matched by "title" exactly; "business_impact" translates the risk into what it costs the business (customers, penalties, revenue, cost, reputation) -- pick only the framings the evidence supports.
- "root_cause_notes": exactly one entry per item in "root_causes" above, matched by "title" exactly. If "root_causes" is empty, return an empty list here and say so plainly in the executive summary instead of inventing one.
- "recommendation_rationale": exactly one entry per item in "recommended_actions" above, matched by "title" exactly.
- "expected_business_impact": 3-5 items describing, qualitatively, what improves if the recommended actions are executed -- based only on the business_impact / expected_business_impact text already present in root_causes and recommended_actions above. No new numbers.
- "department_breakdown": one paragraph each for operations, logistics, customer_experience, supply_chain, and executive_leadership -- what that specific audience needs to know from this report.
- "if_no_action_narrative": what happens across the business if "top_business_risks" and "root_causes" go unaddressed -- grounded only in the evidence and business_impact text already given. If both lists are empty, state plainly that no material risk requiring escalation was identified this period."""


# --------------------------------------------------------------------------
# Response validation
# --------------------------------------------------------------------------

def _validate_narrative_response(narrative: Any) -> dict[str, Any]:
    """Validate the AI's raw narrative JSON before it touches the report skeleton."""
    if not isinstance(narrative, dict):
        raise ResponseValidationError("AI response was not a JSON object.")
    missing = [key for key in REQUIRED_NARRATIVE_KEYS if key not in narrative]
    if missing:
        raise ResponseValidationError(f"AI response is missing required keys: {missing}")

    for text_key in NARRATIVE_TEXT_REPORT_KEYS:
        if not isinstance(narrative.get(text_key), str) or not narrative[text_key].strip():
            raise ResponseValidationError(f"AI response's '{text_key}' is empty or not a string.")

    department_breakdown = narrative.get("department_breakdown")
    if not isinstance(department_breakdown, dict):
        raise ResponseValidationError("AI response's department_breakdown is not an object.")
    missing_departments = [
        dept for dept in REQUIRED_DEPARTMENTS
        if not isinstance(department_breakdown.get(dept), str) or not department_breakdown[dept].strip()
    ]
    if missing_departments:
        raise ResponseValidationError(f"AI response's department_breakdown is missing content for: {missing_departments}")

    for list_key in ("risk_narratives", "root_cause_notes", "recommendation_rationale", "expected_business_impact"):
        if not isinstance(narrative.get(list_key), list):
            raise ResponseValidationError(f"AI response's '{list_key}' must be a list.")

    return narrative


def _validate_final_report(report: dict[str, Any]) -> None:
    """Validate the fully assembled report before it is persisted or returned.

    Raises:
        ResponseValidationError: If required sections are missing,
            empty, or the report cannot round-trip through JSON.
    """
    missing = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    if missing:
        raise ResponseValidationError(f"Assembled report is missing required sections: {missing}")
    for text_key in NARRATIVE_TEXT_REPORT_KEYS:
        if not isinstance(report[text_key], str) or not report[text_key].strip():
            raise ResponseValidationError(f"Assembled report has an empty '{text_key}'.")
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
    """Extract number-like tokens (including percentages) from free text, normalized."""
    return {token.replace(",", "") for token in NUMBER_PATTERN.findall(text)}


def _collect_context_numbers(context: dict[str, Any]) -> set[str]:
    """Every number that legitimately appears anywhere in the AI's provided context."""
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
    """Best-effort check: numbers in the AI-authored narrative not traceable to the context.

    This is a heuristic safety net, not a guarantee -- free text can
    reformat a number (rounding, units) in ways this simple matcher
    won't recognize as equivalent. It exists to catch clear
    fabrications and disclose them, not to replace the structural
    guarantee that every *skeleton* number is Python-sourced.
    """
    narrative_text = " ".join([
        *(report.get(key, "") for key in NARRATIVE_TEXT_REPORT_KEYS),
        " ".join(block.get("narrative", "") for block in report.get("overall_business_health", {}).values()),
        " ".join(risk.get("business_impact", "") for risk in report.get("top_business_risks", [])),
        " ".join(item.get("expected_improvement", "") for item in report.get("expected_business_impact", [])),
        " ".join(report.get("department_breakdown", {}).values()),
    ])
    found = _extract_numbers(narrative_text)
    return sorted(n for n in found if n not in context_numbers and n not in NUMBER_GROUNDING_IGNORE)


# --------------------------------------------------------------------------
# Merge: narrative text + factual skeleton -> final report
# --------------------------------------------------------------------------

def _merge_narrative_into_skeleton(skeleton: dict[str, Any], narrative: dict[str, Any]) -> dict[str, Any]:
    """Combine the Python-built skeleton with the AI's narrative text into the final report.

    Every dict here is newly constructed (not mutating shared inputs),
    so this function has no side effects on `skeleton`.
    """
    report: dict[str, Any] = {
        "metadata": dict(skeleton["metadata"]),
        "executive_headline": narrative["executive_headline"].strip(),
        "why_it_matters": narrative["why_it_matters"].strip(),
        "consequence_of_inaction": narrative["consequence_of_inaction"].strip(),
        "executive_summary": narrative["executive_summary"].strip(),
        "overall_business_health": {},
        "highest_risk_location": skeleton["highest_risk_location"],
        "highest_priority_initiative": skeleton["highest_priority_initiative"],
        "top_business_risks": [],
        "root_causes": [],
        "recommended_actions": [],
        "expected_business_impact": narrative.get("expected_business_impact", []),
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
        report["top_business_risks"].append({
            **risk, "business_impact": _match_by_title(risk_narratives, risk["title"], "business_impact"),
        })

    root_cause_notes = narrative.get("root_cause_notes", [])
    for root_cause in skeleton["root_causes"]:
        report["root_causes"].append({
            **root_cause, "executive_note": _match_by_title(root_cause_notes, root_cause["title"], "executive_note"),
        })

    rationale_notes = narrative.get("recommendation_rationale", [])
    for action in skeleton["recommended_actions"]:
        report["recommended_actions"].append({
            **action, "executive_rationale": _match_by_title(rationale_notes, action["title"], "executive_rationale"),
        })

    return report


# --------------------------------------------------------------------------
# Markdown rendering (deterministic -- not a second AI call)
# --------------------------------------------------------------------------
#
# Trend architecture (see requirement 7): every health block already
# carries a `trend` dict shaped {"available", "direction", "delta"} --
# `_trend()` fills it in from the previous persisted report when one
# exists, and leaves `available=False` otherwise. `_render_trend_cell`
# below is the only place that turns that dict into text, so adding a
# new trend presentation later (a sparkline, a color, an arrow icon
# swap) is a one-function change, not a rewrite of the report loop.

def _label(key: str) -> str:
    """"customer_experience" -> "Customer Experience". Used for every domain/department heading."""
    return key.replace("_", " ").title()


def _fmt_score(score: float | None) -> str:
    return "N/A" if score is None else str(score)


def _fmt_confidence(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0%}"


def _render_trend_cell(trend: dict[str, Any]) -> str:
    if not trend.get("available"):
        return "No historical comparison available"
    direction = trend.get("direction")
    if direction == "flat":
        return "▬ Flat vs. last report"
    arrow = "▲" if direction == "up" else "▼"
    return f"{arrow} {trend['delta']:+.1f} pts vs. last report"


def _render_executive_alert(report: dict[str, Any]) -> list[str]:
    """The callout box a COO reads in the first five seconds: issue, stakes, cost of inaction, top action."""
    top_action_title = report.get("highest_priority_initiative")
    top_action = next(
        (action for action in report["recommended_actions"] if action["title"] == top_action_title), None
    )
    lines = [
        f"> **{report['executive_headline']}**",
        ">",
        f"> {report['why_it_matters']}",
        ">",
        f"> **If nothing changes:** {report['consequence_of_inaction']}",
    ]
    if top_action:
        lines.append(">")
        lines.append(f"> **Top priority:** {top_action['title']} -- Owner: {top_action['owner']} ({top_action['horizon']})")
    return lines


def _render_dashboard(report: dict[str, Any]) -> list[str]:
    """The five-second-read KPI table plus the two headline facts that aren't scores."""
    lines = ["| Metric | Status | Score | Trend |", "|---|---|---|---|"]
    for domain, block in report["overall_business_health"].items():
        icon = HEALTH_STATUS_ICON.get(block["status"], "")
        lines.append(
            f"| {_label(domain)} | {icon} {block['status']} | {_fmt_score(block['score'])} | "
            f"{_render_trend_cell(block['trend'])} |"
        )
    lines.append("")
    lines.append(f"**Highest Risk Location:** {report['highest_risk_location'] or 'Not identified this period.'}")
    lines.append("")
    lines.append(f"**Highest Priority Initiative:** {report['highest_priority_initiative'] or 'No priority initiative identified this period.'}")
    return lines


def _render_risks(risks: list[dict[str, Any]]) -> list[str]:
    if not risks:
        return ["_No significant business risks were identified this period._"]
    lines: list[str] = []
    for risk in risks:
        icon = SEVERITY_ICON.get(risk["severity"], "")
        lines.append(f"### {risk['rank']}. {icon} {risk['title']} -- {risk['severity']} · {risk['urgency']}")
        lines.append(f"- **Evidence:** {risk['evidence']}")
        if risk.get("business_impact"):
            lines.append(f"- **Executive Business Impact:** {risk['business_impact']}")
        lines.append("")
    return lines


def _render_root_causes(root_causes: list[dict[str, Any]]) -> list[str]:
    if not root_causes:
        return ["_No cross-domain root cause met the platform's evidence threshold this period._"]
    lines: list[str] = []
    for root_cause in root_causes:
        lines.append(f"### {root_cause['title']} (confidence: {_fmt_confidence(root_cause['confidence'])})")
        if root_cause.get("executive_note"):
            lines.append(root_cause["executive_note"])
        lines.append("")
        if root_cause.get("evidence"):
            lines.append("**Evidence:**")
            for item in root_cause["evidence"]:
                lines.append(f"- {item}")
        lines.append(f"\n**Business impact:** {root_cause.get('business_impact', '')}")
        lines.append("")
    return lines


def _render_action_plan(actions: list[dict[str, Any]]) -> list[str]:
    """A scan table for triage, then the same actions grouped by horizon for execution."""
    if not actions:
        return ["_No priority actions were identified this period._", ""]

    lines = ["| Priority | Initiative | Owner | Timeline | Confidence |", "|---|---|---|---|---|"]
    for action in actions:
        lines.append(
            f"| {action['priority']} | {action['title']} | {action['owner']} | "
            f"{action['horizon']} | {_fmt_confidence(action['confidence'])} |"
        )
    lines.append("")

    for _, horizon_label in ACTION_HORIZONS:
        horizon_actions = [action for action in actions if action["horizon"] == horizon_label]
        if not horizon_actions:
            continue
        lines.append(f"### {horizon_label}")
        lines.append("")
        for action in horizon_actions:
            lines.append(f"**{action['priority']}. {action['title']}**  ")
            lines.append(f"Owner: {action['owner']} · Difficulty: {action['difficulty']} · Est. timeframe: {action['estimated_timeframe']} · Confidence: {_fmt_confidence(action['confidence'])}")
            lines.append("")
            lines.append(action.get("reason", ""))
            if action.get("executive_rationale"):
                lines.append(f"*{action['executive_rationale']}*")
            lines.append(f"**Expected outcome:** {action.get('expected_business_impact', '')}")
            lines.append("")
    return lines


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the final report dict to professionally formatted Markdown.

    Deterministic Python, not a second model call: the report's
    content is already finalized and validated by this point, so
    formatting it is a data-transformation problem, not a
    communication one, and Python does it more reliably and cheaply.
    """
    metadata = report["metadata"]
    divider = "---"
    lines: list[str] = [
        f"# {metadata['report_type']}",
        f"*{metadata['platform']} -- Generated {metadata['generated_at']}*",
        "",
        divider,
        "",
        "## Executive Alert",
        "",
        *_render_executive_alert(report),
        "",
        divider,
        "",
        "## Executive Dashboard",
        "",
        *_render_dashboard(report),
        "",
        divider,
        "",
        "## Executive Summary",
        "",
        report["executive_summary"],
        "",
    ]

    for domain, block in report["overall_business_health"].items():
        if block.get("narrative"):
            lines.append(f"**{_label(domain)}:** {block['narrative']}")
    lines.append("")

    lines.extend([divider, "", "## Top Business Risks", ""])
    lines.extend(_render_risks(report["top_business_risks"]))

    lines.extend([divider, "", "## Root Cause Analysis", ""])
    lines.extend(_render_root_causes(report["root_causes"]))

    lines.extend([divider, "", "## If No Action Is Taken", "", report["if_no_action_narrative"], "", divider, ""])

    lines.extend(["## 90-Day Action Plan", ""])
    lines.extend(_render_action_plan(report["recommended_actions"]))

    lines.extend([divider, "", "## Expected Business Impact", ""])
    for item in report["expected_business_impact"]:
        lines.append(f"- **{item.get('area', '')}:** {item.get('expected_improvement', '')}")
    lines.append("")

    lines.extend([divider, "", "## Department Breakdown", ""])
    for department, text in report["department_breakdown"].items():
        lines.append(f"### {_label(department)}")
        lines.append(text)
        lines.append("")

    appendix = report["appendix"]
    lines.extend([divider, "", "## Appendix", "", "### Methodology", appendix.get("methodology_summary", ""), ""])
    lines.append("### Confidence Methodology")
    lines.append(appendix.get("confidence_explanation", ""))
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

def _save_report(report: dict[str, Any], markdown: str, output_dir: Path, json_path: Path, md_path: Path) -> None:
    """Write both artifacts to disk, creating outputs/ if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def generate_executive_report(
    delivery_analysis: dict[str, Any],
    complaints_analysis: dict[str, Any],
    inventory_analysis: dict[str, Any],
    correlation_analysis: dict[str, Any],
    *,
    provider: AIProvider | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Turn four completed analytics outputs into a persisted executive report.

    Pipeline: validate inputs -> build a fully-computed report skeleton
    in Python -> curate an AI context -> call the AI provider for
    narrative text only -> validate that response -> merge narrative
    into the skeleton -> validate the assembled report -> render
    Markdown -> persist both JSON and Markdown to `output_dir` ->
    return the report dict.

    Args:
        delivery_analysis: Output of analysis.delivery.analyze_deliveries.
        complaints_analysis: Output of analysis.complaints.analyze_complaints.
        inventory_analysis: Output of analysis.inventory.analyze_inventory.
        correlation_analysis: Output of analysis.correlations.analyze_correlations.
        provider: The AI backend to use. Defaults to `GroqProvider()`,
            which reads its configuration from environment variables.
            Pass a different `AIProvider` implementation (or a test
            double) to use another backend without touching this
            function's logic.
        output_dir: Directory to write executive_report.json and
            executive_report.md into. Defaults to `outputs/` at the
            project root.

    Returns:
        The final report dict, shaped as::

            {
                "metadata": {...},
                "executive_headline": "...",
                "why_it_matters": "...",
                "consequence_of_inaction": "...",
                "executive_summary": "...",
                "overall_business_health": {...},
                "highest_risk_location": "..." | None,
                "highest_priority_initiative": "..." | None,
                "top_business_risks": [...],
                "root_causes": [...],
                "recommended_actions": [...],
                "expected_business_impact": [...],
                "department_breakdown": {...},
                "if_no_action_narrative": "...",
                "appendix": {...},
            }

    Raises:
        AnalyticsInputError: If any of the four analytics dicts is
            missing, empty, or malformed.
        ConfigurationError: If required AI provider configuration
            (e.g. GROQ_API_KEY) is missing.
        AIProviderError: If the AI provider fails (auth, timeout, rate
            limit, connection error) after retries are exhausted.
        ResponseValidationError: If the AI's response, or the final
            assembled report, fails structural validation.
    """
    started_at = datetime.now(timezone.utc)

    delivery_analysis = _require_analysis(delivery_analysis, REQUIRED_DELIVERY_KEYS, "delivery_analysis")
    complaints_analysis = _require_analysis(complaints_analysis, REQUIRED_COMPLAINTS_KEYS, "complaints_analysis")
    inventory_analysis = _require_analysis(inventory_analysis, REQUIRED_INVENTORY_KEYS, "inventory_analysis")
    correlation_analysis = _require_analysis(correlation_analysis, REQUIRED_CORRELATION_KEYS, "correlation_analysis")

    provider = provider or GroqProvider()

    output_directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    json_path = output_directory / REPORT_JSON_FILENAME
    md_path = output_directory / REPORT_MARKDOWN_FILENAME
    previous_report = _load_previous_report(json_path)

    skeleton = _build_report_skeleton(
        delivery_analysis, complaints_analysis, inventory_analysis, correlation_analysis,
        provider.model_name, started_at, previous_report,
    )
    context = _build_ai_context(delivery_analysis, complaints_analysis, inventory_analysis, correlation_analysis, skeleton)

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
            f"automatically verified against source analytics; spot-check before distribution: "
            f"{unverified_numbers}."
        )

    report["metadata"]["generation_duration_seconds"] = round((datetime.now(timezone.utc) - started_at).total_seconds(), 2)
    report["metadata"]["saved_json_path"] = str(json_path)
    report["metadata"]["saved_markdown_path"] = str(md_path)

    _validate_final_report(report)

    markdown = _render_markdown(report)
    _save_report(report, markdown, output_directory, json_path, md_path)

    return report
