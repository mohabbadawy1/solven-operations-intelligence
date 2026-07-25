"""Executive Health Scoring engine.

Every domain analytics module (sales_performance, lead_funnel,
inventory_velocity, marketing_efficiency, sales_team_broker_performance,
collections_risk, cancellations, construction_handover,
customer_experience) already computes and documents its own 0-100
health score. This module's only job is to collect those nine scores
plus data_quality's score, classify each into a status band, and blend
them into one Overall Business Health score using the
materiality-aware weights in config/real_estate_demo.yml -- never a
naive average, since a weak collections position is a harder, more
expensive problem than a soft marketing-efficiency quarter, and the
weights say so explicitly rather than leaving it implicit.

This module computes no new domain metric. It is a pure aggregation
layer over scores the nine domain modules already computed and
documented.
"""

from __future__ import annotations

from typing import Any

from analysis._config import CONFIG
from analysis._shared import _json_safe

STATUS_THRESHOLDS = CONFIG["health_status_thresholds"]
OVERALL_WEIGHTS = CONFIG["overall_health_weights"]


def _status_from_score(score: float | None) -> str:
    """Classify a 0-100 health score into a plain-language status, using the shared thresholds
    in config/real_estate_demo.yml so every score on the platform uses the same bands."""
    if score is None:
        return "Unknown"
    if score >= STATUS_THRESHOLDS["healthy_floor"]:
        return "Healthy"
    if score >= STATUS_THRESHOLDS["watch_floor"]:
        return "Watch"
    if score >= STATUS_THRESHOLDS["at_risk_floor"]:
        return "At Risk"
    return "Critical"


def _domain_block(score: float | None, methodology: str | None, source_module: str) -> dict[str, Any]:
    return {
        "score": score, "status": _status_from_score(score),
        "methodology": methodology, "source_module": source_module,
    }


def compute_executive_scores(
    sales_performance: dict[str, Any],
    lead_funnel: dict[str, Any],
    inventory_velocity: dict[str, Any],
    marketing_efficiency: dict[str, Any],
    collections_risk: dict[str, Any],
    cancellations: dict[str, Any],
    construction_handover: dict[str, Any],
    customer_experience: dict[str, Any],
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    """Collect every domain health score and blend them into Overall Business Health.

    Args:
        sales_performance, lead_funnel, inventory_velocity,
            marketing_efficiency, collections_risk, cancellations,
            construction_handover, customer_experience: outputs of the
            corresponding analysis.* module (each already carries its
            own `health_score` block).
        data_quality: output of analysis.data_quality (carries
            `data_quality_score` directly, not a `health_score` block).

    Returns:
        A JSON-serializable dictionary shaped as::

            {
                "domains": {
                    "commercial_health": {"score", "status", "methodology", "source_module"},
                    "sales_funnel_health": {...},
                    "inventory_health": {...},
                    "marketing_efficiency_health": {...},
                    "collections_health": {...},
                    "cancellations_health": {...},
                    "construction_delivery_health": {...},
                    "handover_readiness_health": {...},
                    "customer_experience_health": {...},
                },
                "overall_business_health": {"score", "status", "components", "weights", "methodology"},
                "data_quality": {"score", "status"},
            }
    """
    construction_components = construction_handover["health_score"]["components"]
    # construction_handover analyzes both facets in one module; its own
    # component breakdown (schedule_adherence/cost_control vs.
    # handover_delay_exposure/snagging_quality) is what lets this
    # aggregator honor config's two separate weighted line items
    # without recomputing anything -- see config's own comment.
    construction_delivery_score = round(
        (construction_components["schedule_adherence"] + construction_components["cost_control"]) / 2, 1
    )
    handover_readiness_score = round(
        (construction_components["handover_delay_exposure"] + construction_components["snagging_quality"]) / 2, 1
    )

    domains = {
        "commercial_health": _domain_block(
            sales_performance["health_score"]["overall_score"], sales_performance["health_score"]["methodology"], "sales_performance",
        ),
        "sales_funnel_health": _domain_block(
            lead_funnel["health_score"]["overall_score"], lead_funnel["health_score"]["methodology"], "lead_funnel",
        ),
        "inventory_health": _domain_block(
            inventory_velocity["health_score"]["overall_score"], inventory_velocity["health_score"]["methodology"], "inventory_velocity",
        ),
        "marketing_efficiency_health": _domain_block(
            marketing_efficiency["health_score"]["overall_score"], marketing_efficiency["health_score"]["methodology"], "marketing_efficiency",
        ),
        "collections_health": _domain_block(
            collections_risk["health_score"]["overall_score"], collections_risk["health_score"]["methodology"], "collections_risk",
        ),
        "cancellations_health": _domain_block(
            cancellations["health_score"]["overall_score"], cancellations["health_score"]["methodology"], "cancellations",
        ),
        "construction_delivery_health": _domain_block(
            construction_delivery_score,
            "Average of construction_handover's schedule_adherence and cost_control components.",
            "construction_handover",
        ),
        "handover_readiness_health": _domain_block(
            handover_readiness_score,
            "Average of construction_handover's handover_delay_exposure and snagging_quality components.",
            "construction_handover",
        ),
        "customer_experience_health": _domain_block(
            customer_experience["health_score"]["overall_score"], customer_experience["health_score"]["methodology"], "customer_experience",
        ),
    }

    weighted_sum = sum((domains[k]["score"] or 0.0) * OVERALL_WEIGHTS[k] for k in OVERALL_WEIGHTS)
    overall_score = round(weighted_sum, 1)

    result = {
        "domains": domains,
        "overall_business_health": {
            "score": overall_score,
            "status": _status_from_score(overall_score),
            "components": {k: domains[k]["score"] for k in OVERALL_WEIGHTS},
            "weights": OVERALL_WEIGHTS,
            "methodology": (
                "Overall Business Health is a materiality-weighted blend of nine domain health "
                "scores, each already computed and documented by its own analytics module (see each "
                "domain's own 'methodology' field for how that score is built). Weights are set in "
                "config/real_estate_demo.yml -- collections (0.18), construction delivery (0.16), "
                "cancellations (0.12), and customer experience (0.12) carry more weight than "
                "marketing efficiency (0.05) or inventory (0.06), reflecting that a receivables or "
                "delivery problem is more expensive and slower to unwind than a soft marketing or "
                "inventory-mix quarter. This is never a naive average."
            ),
        },
        "data_quality": {
            "score": data_quality.get("data_quality_score"),
            "status": data_quality.get("status"),
        },
        "status_thresholds": STATUS_THRESHOLDS,
    }
    return _json_safe(result)
