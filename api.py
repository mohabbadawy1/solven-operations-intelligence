"""Minimal HTTP API so n8n (or any HTTP caller) can trigger the existing
operations-intelligence pipeline and retrieve its result.

This module contains no analytics, scoring, or report-generation logic
of its own. It only calls `app.run_pipeline()` -- the same orchestration
function `python app.py` uses -- and translates its result (or failure)
into an HTTP response. Python continues to own every calculation; n8n's
role stays limited to scheduling, triggering this endpoint, and later
handling delivery/storage of the report it gets back.

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health         Liveness check.
    POST /run-analysis    Run the full pipeline synchronously and return
                          its result. Blocks until the pipeline finishes.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ai.report_generator import AIProviderError, ReportGenerationError
from app import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solven.api")

app = FastAPI(
    title="Solven Operations Intelligence API",
    description="Triggers the existing Python analytics/reporting pipeline over HTTP.",
    version="1.0.0",
)

# The pipeline reads and overwrites the same fixed files under outputs/
# on every run and isn't designed for concurrent execution. A single
# process-wide lock is the smallest solution that prevents two
# /run-analysis calls from racing on those files; a non-blocking
# acquire lets an overlapping call fail fast with 409 instead of
# silently queuing behind an unbounded wait.
_pipeline_lock = threading.Lock()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check -- does not touch the pipeline or any external service."""
    return {"status": "healthy", "service": "Solven Operations Intelligence"}


@app.post("/run-analysis")
def run_analysis() -> JSONResponse:
    """Run the full analytics + AI reporting pipeline and return its result.

    Synchronous by design: the caller (n8n) waits for the HTTP response,
    which is only sent once `run_pipeline()` has finished and the JSON,
    Markdown, and HTML reports have been persisted to disk.
    """
    if not _pipeline_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": "An analysis run is already in progress. Try again once it finishes.",
            },
        )
    try:
        report = run_pipeline()
    except AIProviderError as exc:
        # This one wraps a raw exception from the third-party Groq SDK,
        # which can itself embed account/billing details (seen live
        # during testing: an org ID and a billing URL came through in a
        # rate-limit error). Log the full message server-side; never
        # forward it as-is over HTTP.
        logger.error("AI provider error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "The AI provider failed to generate the report. Check server logs for details.",
            },
        )
    except ReportGenerationError as exc:
        # Every other exception this pipeline deliberately raises
        # (missing GROQ_API_KEY, malformed analytics input, response
        # validation failure) is a ReportGenerationError with a clean,
        # human-authored message -- safe to return as-is, and far more
        # useful to n8n than a generic failure string.
        logger.error("Pipeline failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc)},
        )
    except Exception:
        # Anything else (bad input data, an unexpected library error) is
        # unplanned: log the full traceback server-side for debugging,
        # but never echo it -- or any path/environment detail from it --
        # back over HTTP.
        logger.exception("Unexpected error while running the analysis pipeline")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "An unexpected error occurred while running the analysis. Check server logs for details.",
            },
        )
    finally:
        _pipeline_lock.release()

    return JSONResponse(status_code=200, content=_success_payload(report))


def _success_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Shape a successful pipeline result into the API's response contract.

    Every value here is read directly from the report the pipeline
    already produced -- report_id, timestamps, output paths, and the
    three summary figures are all copied, never recomputed or guessed.
    """
    metadata = report["metadata"]
    operations_overall = report.get("overall_business_health", {}).get("operations_overall", {})
    return {
        "success": True,
        "message": "Operations intelligence analysis completed",
        "report_id": metadata.get("report_id"),
        "generated_at": metadata.get("generated_at"),
        "outputs": {
            "json": metadata.get("saved_json_path"),
            "markdown": metadata.get("saved_markdown_path"),
            "html": metadata.get("saved_html_path"),
        },
        "summary": {
            "operations_health_score": operations_overall.get("score"),
            "highest_risk_location": report.get("highest_risk_location"),
            "highest_priority_initiative": report.get("highest_priority_initiative"),
        },
    }
