"""Client-facing HTML rendering for the executive report.

This module owns exactly one job: turn the final report dict --
already fully computed and validated by ai/report_generator.py -- into
a single self-contained HTML document. It performs no analytics, no
scoring, and no narrative generation; every value on the page is read
directly from the report dict, the same source of truth the JSON and
Markdown outputs are built from. If a value isn't already in `report`,
it doesn't appear on the page -- charts included: every chart here is
a plain HTML/CSS rendering of numbers analysis/*.py already computed
(delivery on-time rate per warehouse, the complaints engine's negative-
sentiment percentage, root-cause confidence, etc.), never a new
statistic and never an invented one.

The document is fully self-contained (all CSS inlined, no external
fonts/scripts/CDNs, no network calls) so it opens correctly from disk
with no internet connection, and is styled to be both screen-readable
and printable via an embedded `@media print` block.

All dynamic text is passed through `html.escape` before being placed
in the document -- the report contains free-text narrative from an LLM
and evidence strings copied from source data, neither of which is
trusted to be free of `<`, `&`, or `"` characters.

Layout is organized as one builder function per section (mirrors the
report's own section list): header, executive alert, KPI dashboard,
performance charts, executive summary, business risks, root causes,
if-no-action, action-plan timeline, expected impact, department
breakdown, appendix, footer. A small set of chart/component primitives
(`_hbar_row`, `_donut_html`, `_badge_html`, ...) are shared across
those builders so no section reimplements its own bar or badge markup.
"""

from __future__ import annotations

from html import escape
from typing import Any

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class HTMLRenderError(Exception):
    """The report dict was missing a field this renderer requires.

    Raised instead of letting a bare KeyError/TypeError escape, so a
    caller (or a future standalone re-render of an old JSON file) gets
    a message that names the report shape problem instead of a raw
    traceback into string-formatting code.
    """


# --------------------------------------------------------------------------
# Presentation vocabulary
# --------------------------------------------------------------------------
# These map values the analytics/orchestration layers already computed
# (a status string, a severity string) to a CSS class name. They select
# a look, never a number -- no score, confidence, or ranking is decided
# here.

STATUS_CLASS = {"Healthy": "status-healthy", "At Risk": "status-at-risk", "Critical": "status-critical"}
SEVERITY_CLASS = {"HIGH": "severity-high", "MEDIUM": "severity-medium", "LOW": "severity-low"}
DEFAULT_STATUS_CLASS = "status-unknown"
DEFAULT_SEVERITY_CLASS = "severity-low"

# The only three horizon labels report_generator._action_horizon can ever
# produce, in chronological display order. Not imported from
# report_generator.py to avoid a circular import (that module imports
# render_html from this one); if that vocabulary ever changes, update it
# in both places.
ACTION_HORIZON_ORDER = ("Immediate (0-30 Days)", "Near-Term (30-60 Days)", "Strategic (60-120 Days)")


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------


def _esc(value: Any) -> str:
    """Escape a value for safe inclusion in HTML text content or an attribute."""
    if value is None:
        return ""
    return escape(str(value), quote=True)


def _label(key: str) -> str:
    """"customer_experience" -> "Customer Experience"."""
    return key.replace("_", " ").title()


def _fmt_score(score: float | None) -> str:
    return "N/A" if score is None else str(score)


def _percent(value: float | None, of: float = 100.0) -> float:
    """Clamp a value onto a 0-100 CSS-safe percentage width."""
    if value is None:
        return 0.0
    return max(0.0, min(100.0, float(value) / of * 100))


def _fmt_confidence(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0%}"


def _brand_lockup(platform: str | None) -> tuple[str, str]:
    """Split "Solven Operations Intelligence Platform" into ("SOLVEN", "Operations Intelligence").

    Pure string derivation of the existing metadata.platform value --
    no new copy is introduced. First word becomes the wordmark; the
    interior words (excluding a trailing "Platform"-style suffix)
    become the sub-brand line. Degrades gracefully for shorter strings.
    """
    words = (platform or "").split()
    if not words:
        return "", ""
    if len(words) == 1:
        return words[0].upper(), ""
    tagline_words = words[1:-1] if len(words) > 2 else words[1:]
    return words[0].upper(), " ".join(tagline_words)


def _paragraphs_html(text: str, css_class: str = "") -> str:
    """Split narrative text on blank lines into separate escaped <p> blocks."""
    blocks = [block.strip() for block in (text or "").split("\n\n") if block.strip()]
    class_attr = f' class="{css_class}"' if css_class else ""
    if not blocks:
        return f"<p{class_attr}>&mdash;</p>"
    return "\n".join(f"<p{class_attr}>{_esc(block)}</p>" for block in blocks)


def _trend_html(trend: dict[str, Any] | None, compact: bool = False) -> str:
    if not trend or not trend.get("available"):
        label = "No prior report" if compact else "No historical comparison available"
        return f'<span class="trend trend-unavailable">{label}</span>'
    direction = trend.get("direction")
    delta = trend.get("delta")
    suffix = "" if compact else " vs. last report"
    if direction == "up":
        return f'<span class="trend trend-up">&#9650; {_esc(f"{delta:+.1f}")} pts{suffix}</span>'
    if direction == "down":
        return f'<span class="trend trend-down">&#9660; {_esc(f"{delta:+.1f}")} pts{suffix}</span>'
    return f'<span class="trend trend-flat">&#9644; Flat{suffix}</span>'


def _badge_html(text: Any, css_class: str = "") -> str:
    classes = f"badge {css_class}".strip()
    return f'<span class="{classes}">{_esc(text)}</span>'


# --------------------------------------------------------------------------
# Chart primitives (pure HTML/CSS -- no canvas, no SVG library, no JS)
# --------------------------------------------------------------------------


def _hbar_row(label: str, percent: float, display_value: str, bar_class: str = "", meta_html: str = "") -> str:
    """One labeled horizontal bar row, used by every bar-style chart on the page."""
    return f"""
    <div class="hbar-row">
      <div class="hbar-row-head">
        <span class="hbar-label">{_esc(label)}</span>
        <span class="hbar-value">{_esc(display_value)}</span>
      </div>
      <div class="hbar-track"><div class="hbar-fill {bar_class}" style="width:{percent}%"></div></div>
      {meta_html}
    </div>""".strip()


def _donut_html(primary_percent: float, primary_label: str, secondary_label: str, center_caption: str) -> str:
    """A two-segment donut built from a single CSS conic-gradient -- no SVG, no JS."""
    primary_percent = max(0.0, min(100.0, primary_percent))
    return f"""
    <div class="donut-wrap">
      <div class="donut" style="background: conic-gradient(var(--red) 0% {primary_percent}%, var(--surface-alt) {primary_percent}% 100%)">
        <div class="donut-center">
          <span class="donut-value">{_esc(f"{primary_percent:.0f}%")}</span>
          <span class="donut-caption">{_esc(center_caption)}</span>
        </div>
      </div>
      <div class="donut-legend">
        <div class="legend-item"><span class="legend-dot legend-dot-primary"></span>{_esc(primary_label)}</div>
        <div class="legend-item"><span class="legend-dot legend-dot-secondary"></span>{_esc(secondary_label)}</div>
      </div>
    </div>""".strip()


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------


def _render_header(metadata: dict[str, Any]) -> str:
    brand, tagline = _brand_lockup(metadata.get("platform"))
    monogram = _esc(brand[:1]) if brand else ""
    tagline_html = f'<p class="brand-tagline">{_esc(tagline)}</p>' if tagline else ""
    return f"""
<header class="report-header">
  <div class="brand-row">
    <span class="brand-mark">{monogram}</span>
    <div>
      <p class="brand-word">{_esc(brand)}</p>
      {tagline_html}
    </div>
  </div>
  <h1 class="title">{_esc(metadata.get('report_type'))}</h1>
  <p class="meta-line">Generated {_esc(metadata.get('generated_at'))} &middot; Report {_esc(metadata.get('report_id'))}</p>
</header>
""".strip()


# --------------------------------------------------------------------------
# Executive Alert
# --------------------------------------------------------------------------


def _render_executive_alert(report: dict[str, Any]) -> str:
    top_action_title = report.get("highest_priority_initiative")
    top_action = next(
        (action for action in report.get("recommended_actions", []) if action.get("title") == top_action_title), None
    )
    top_priority_html = ""
    if top_action:
        top_priority_html = f"""
    <div class="alert-row alert-row-action">
      <span class="alert-tag">Top Priority</span>
      <span>{_esc(top_action.get('title'))} &mdash; Owner: {_esc(top_action.get('owner'))} &middot; {_esc(top_action.get('horizon'))}</span>
    </div>"""

    return f"""
<section class="section">
  <div class="card alert-card">
    <p class="alert-headline">{_esc(report.get('executive_headline'))}</p>
    <p class="alert-body prose">{_esc(report.get('why_it_matters'))}</p>
    <div class="alert-row">
      <span class="alert-tag">If Nothing Changes</span>
      <span>{_esc(report.get('consequence_of_inaction'))}</span>
    </div>{top_priority_html}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Executive Dashboard (KPI cards)
# --------------------------------------------------------------------------


def _render_kpi_card(domain: str, block: dict[str, Any]) -> str:
    status = block.get("status") or "Unknown"
    status_class = STATUS_CLASS.get(status, DEFAULT_STATUS_CLASS)
    insight_html = (
        f'<p class="kpi-insight">&ldquo;{_esc(block["narrative"])}&rdquo;</p>' if block.get("narrative") else ""
    )
    return f"""
    <div class="card kpi-card">
      <p class="kpi-label">{_esc(_label(domain))}</p>
      <p class="kpi-score">{_esc(_fmt_score(block.get('score')))}</p>
      <div class="hbar-track kpi-track"><div class="hbar-fill {status_class}" style="width:{_percent(block.get('score'))}%"></div></div>
      <div class="kpi-meta-row">
        {_badge_html(status, status_class)}
        {_trend_html(block.get('trend', {}), compact=True)}
      </div>
      {insight_html}
    </div>""".strip()


def _render_dashboard(report: dict[str, Any]) -> str:
    kpi_cards = "\n".join(
        _render_kpi_card(domain, block) for domain, block in report.get("overall_business_health", {}).items()
    )
    highest_risk_location = report.get("highest_risk_location") or "Not identified this period."
    highest_priority_initiative = report.get("highest_priority_initiative") or "No priority initiative identified this period."
    return f"""
<section class="section">
  <h2 class="section-title">Executive Dashboard</h2>
  <div class="kpi-grid">
{kpi_cards}
  </div>
  <div class="fact-grid">
    <div class="card fact-card">
      <p class="fact-label">Highest Risk Location</p>
      <p class="fact-value">{_esc(highest_risk_location)}</p>
    </div>
    <div class="card fact-card">
      <p class="fact-label">Highest Priority Initiative</p>
      <p class="fact-value">{_esc(highest_priority_initiative)}</p>
    </div>
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Performance charts
# --------------------------------------------------------------------------


def _render_business_health_chart(overall_business_health: dict[str, Any]) -> str:
    if not overall_business_health:
        return ""
    rows = "\n".join(
        _hbar_row(
            _label(domain), _percent(block.get("score")), _fmt_score(block.get("score")),
            bar_class=STATUS_CLASS.get(block.get("status"), DEFAULT_STATUS_CLASS),
            meta_html=f'<div class="hbar-meta">{_trend_html(block.get("trend", {}), compact=True)}</div>',
        )
        for domain, block in overall_business_health.items()
    )
    return f"""
  <div class="card chart-card">
    <p class="chart-title">Business Health Comparison</p>
    {rows}
  </div>""".strip()


def _render_warehouse_chart(warehouse_performance: list[dict[str, Any]], highest_risk_location: str | None) -> str:
    if not warehouse_performance:
        return ""
    rows = []
    for row in warehouse_performance:
        warehouse = row.get("warehouse")
        rate = row.get("on_time_rate_percentage")
        is_highest_risk = warehouse is not None and warehouse == highest_risk_location
        bar_class = "hbar-flag" if is_highest_risk else "hbar-neutral"
        meta_html = '<div class="hbar-meta"><span class="hbar-flag-label">Highest risk location</span></div>' if is_highest_risk else ""
        rows.append(_hbar_row(warehouse, _percent(rate), "N/A" if rate is None else f"{rate}%", bar_class, meta_html))
    return f"""
  <div class="card chart-card">
    <p class="chart-title">Warehouse On-Time Delivery Rate</p>
    {chr(10).join(rows)}
  </div>""".strip()


def _render_sentiment_chart(customer_sentiment: dict[str, Any]) -> str:
    negative = customer_sentiment.get("negative_percentage") if customer_sentiment else None
    non_negative = customer_sentiment.get("non_negative_percentage") if customer_sentiment else None
    if negative is None or non_negative is None:
        return ""
    donut = _donut_html(negative, f"Negative &middot; {negative}%", f"Non-negative &middot; {non_negative}%", "Negative")
    return f"""
  <div class="card chart-card">
    <p class="chart-title">Customer Sentiment</p>
    {donut}
  </div>""".strip()


def _render_root_cause_confidence_chart(root_causes: list[dict[str, Any]]) -> str:
    if not root_causes:
        return ""
    rows = "\n".join(
        _hbar_row(root_cause.get("title", ""), _percent((root_cause.get("confidence") or 0) * 100), _fmt_confidence(root_cause.get("confidence")))
        for root_cause in root_causes
    )
    return f"""
  <div class="card chart-card">
    <p class="chart-title">Root Cause Confidence</p>
    {rows}
  </div>""".strip()


def _render_risk_severity_chart(risks: list[dict[str, Any]]) -> str:
    if not risks:
        return ""
    total = len(risks)
    counts: dict[str, int] = {}
    for risk in risks:
        severity = risk.get("severity") or "LOW"
        counts[severity] = counts.get(severity, 0) + 1
    rows = "\n".join(
        _hbar_row(severity.title(), _percent(count, of=total), f"{count} of {total}", SEVERITY_CLASS.get(severity, DEFAULT_SEVERITY_CLASS))
        for severity, count in counts.items()
    )
    return f"""
  <div class="card chart-card">
    <p class="chart-title">Top Risks by Severity</p>
    {rows}
  </div>""".strip()


def _render_charts(report: dict[str, Any]) -> str:
    charts = report.get("charts", {})
    panels = [
        _render_business_health_chart(report.get("overall_business_health", {})),
        _render_warehouse_chart(charts.get("warehouse_performance", []), report.get("highest_risk_location")),
        _render_sentiment_chart(charts.get("customer_sentiment", {})),
        _render_root_cause_confidence_chart(report.get("root_causes", [])),
        _render_risk_severity_chart(report.get("top_business_risks", [])),
    ]
    panels = [panel for panel in panels if panel]
    if not panels:
        return ""
    return f"""
<section class="section">
  <h2 class="section-title">Performance Overview</h2>
  <div class="chart-grid">
{chr(10).join(panels)}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------


def _render_summary(report: dict[str, Any]) -> str:
    narrative_rows = "\n".join(
        f'    <div class="summary-row"><span class="summary-domain">{_esc(_label(domain))}</span>'
        f'<span>{_esc(block.get("narrative"))}</span></div>'
        for domain, block in report.get("overall_business_health", {}).items()
        if block.get("narrative")
    )
    return f"""
<section class="section">
  <h2 class="section-title">Executive Summary</h2>
  <div class="card">
    {_paragraphs_html(report.get('executive_summary', ''), css_class="prose lead")}
  </div>
  <div class="summary-grid">
{narrative_rows}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Top Business Risks
# --------------------------------------------------------------------------


def _render_risks(risks: list[dict[str, Any]]) -> str:
    if not risks:
        body = '<p class="empty-state">No significant business risks were identified this period.</p>'
    else:
        cards = []
        for risk in risks:
            severity = risk.get("severity") or "LOW"
            severity_class = SEVERITY_CLASS.get(severity, DEFAULT_SEVERITY_CLASS)
            impact_html = (
                f'<p class="risk-impact"><strong>Executive Business Impact:</strong> {_esc(risk.get("business_impact"))}</p>'
                if risk.get("business_impact") else ""
            )
            cards.append(f"""
    <div class="card risk-card">
      <div class="risk-header">
        <span class="risk-rank">{_esc(risk.get('rank'))}</span>
        <span class="risk-title">{_esc(risk.get('title'))}</span>
        <div class="risk-badges">
          {_badge_html(severity, severity_class)}
          {_badge_html(risk.get('urgency'), 'badge-outline')}
        </div>
      </div>
      <p class="risk-evidence"><strong>Evidence:</strong> {_esc(risk.get('evidence'))}</p>
      {impact_html}
    </div>""".strip())
        body = "\n".join(cards)
    return f"""
<section class="section">
  <h2 class="section-title">Top Business Risks</h2>
  {body}
</section>
""".strip()


# --------------------------------------------------------------------------
# Root Cause Analysis
# --------------------------------------------------------------------------


def _render_root_causes(root_causes: list[dict[str, Any]]) -> str:
    if not root_causes:
        body = '<p class="empty-state">No cross-domain root cause met the platform&rsquo;s evidence threshold this period.</p>'
    else:
        cards = []
        for root_cause in root_causes:
            evidence_items = root_cause.get("evidence") or []
            evidence_html = (
                "<ul class=\"evidence-list\">" + "".join(f"<li>{_esc(item)}</li>" for item in evidence_items) + "</ul>"
                if evidence_items else ""
            )
            note_html = (
                f'<p class="root-cause-note">{_esc(root_cause.get("executive_note"))}</p>'
                if root_cause.get("executive_note") else ""
            )
            priority_badge = _badge_html(f"P{root_cause['priority']}", "badge-priority") if root_cause.get("priority") else ""
            owner_html = f'<p class="root-cause-owner">Owner: {_esc(root_cause.get("owner"))}</p>' if root_cause.get("owner") else ""
            cards.append(f"""
    <div class="card root-cause-card">
      <div class="root-cause-header">
        <div class="root-cause-heading">
          {priority_badge}
          <span class="risk-title">{_esc(root_cause.get('title'))}</span>
        </div>
        <span class="confidence-badge">Confidence: {_esc(_fmt_confidence(root_cause.get('confidence')))}</span>
      </div>
      <div class="hbar-track confidence-track"><div class="hbar-fill hbar-confidence" style="width:{_percent((root_cause.get('confidence') or 0) * 100)}%"></div></div>
      {note_html}
      {evidence_html}
      <p class="root-cause-impact"><strong>Business impact:</strong> {_esc(root_cause.get('business_impact'))}</p>
      {owner_html}
    </div>""".strip())
        body = "\n".join(cards)
    return f"""
<section class="section">
  <h2 class="section-title">Root Cause Analysis</h2>
  {body}
</section>
""".strip()


# --------------------------------------------------------------------------
# If No Action Is Taken
# --------------------------------------------------------------------------


def _render_no_action(report: dict[str, Any]) -> str:
    return f"""
<section class="section">
  <h2 class="section-title">If No Action Is Taken</h2>
  <div class="card no-action-card">
    {_paragraphs_html(report.get('if_no_action_narrative', ''), css_class="prose")}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# 90-Day Action Plan (timeline roadmap)
# --------------------------------------------------------------------------


def _render_action_card(action: dict[str, Any]) -> str:
    rationale_html = (
        f'<p class="action-rationale">{_esc(action.get("executive_rationale"))}</p>'
        if action.get("executive_rationale") else ""
    )
    return f"""
      <div class="card action-card">
        <div class="action-card-head">
          {_badge_html(f"P{action.get('priority')}", "badge-priority")}
          <p class="action-title">{_esc(action.get('title'))}</p>
        </div>
        <p class="action-meta">Owner: {_esc(action.get('owner'))} &middot; Difficulty: {_esc(action.get('difficulty'))} &middot; Timeline: {_esc(action.get('estimated_timeframe'))} &middot; Confidence: {_esc(_fmt_confidence(action.get('confidence')))}</p>
        <p class="action-reason">{_esc(action.get('reason'))}</p>
        {rationale_html}
        <p class="action-outcome"><strong>Expected outcome:</strong> {_esc(action.get('expected_business_impact'))}</p>
      </div>""".strip()


def _render_action_plan(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return """
<section class="section">
  <h2 class="section-title">90-Day Action Plan</h2>
  <p class="empty-state">No priority actions were identified this period.</p>
</section>
""".strip()

    columns_html = []
    for horizon_label in ACTION_HORIZON_ORDER:
        horizon_actions = [action for action in actions if action.get("horizon") == horizon_label]
        if not horizon_actions:
            continue
        cards = "\n".join(_render_action_card(action) for action in horizon_actions)
        columns_html.append(f"""
    <div class="timeline-column">
      <p class="timeline-title">{_esc(horizon_label)}</p>
{cards}
    </div>""".strip())

    return f"""
<section class="section">
  <h2 class="section-title">90-Day Action Plan</h2>
  <div class="timeline-grid">
{chr(10).join(columns_html)}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Expected Business Impact
# --------------------------------------------------------------------------


def _render_expected_impact(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    rows = "\n".join(
        f'    <li><strong>{_esc(item.get("area"))}:</strong> {_esc(item.get("expected_improvement"))}</li>'
        for item in items
    )
    return f"""
<section class="section">
  <h2 class="section-title">Expected Business Impact</h2>
  <div class="card">
    <ul class="impact-list">
{rows}
    </ul>
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Department Breakdown
# --------------------------------------------------------------------------


def _render_departments(department_breakdown: dict[str, str]) -> str:
    cards = "\n".join(
        f"""    <div class="card dept-card">
      <p class="dept-index">{_esc(f"{index:02d}")}</p>
      <p class="dept-title">{_esc(_label(department))}</p>
      <p class="dept-text">{_esc(text)}</p>
    </div>"""
        for index, (department, text) in enumerate(department_breakdown.items(), start=1)
    )
    return f"""
<section class="section">
  <h2 class="section-title">Department Breakdown</h2>
  <div class="dept-grid">
{cards}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Appendix
# --------------------------------------------------------------------------


def _render_appendix(appendix: dict[str, Any]) -> str:
    limitations = appendix.get("data_limitations") or []
    limitations_html = "".join(f"<li>{_esc(item)}</li>" for item in limitations)
    return f"""
<section class="section">
  <h2 class="section-title">Appendix</h2>
  <div class="card appendix-card">
    <h3 class="appendix-heading">Methodology</h3>
    <p class="prose">{_esc(appendix.get('methodology_summary'))}</p>
    <h3 class="appendix-heading">Confidence Methodology</h3>
    <p class="prose">{_esc(appendix.get('confidence_explanation'))}</p>
    <h3 class="appendix-heading">Data Limitations</h3>
    <ul class="limitations-list">{limitations_html}</ul>
    <p class="appendix-footnote">Generated by {_esc(appendix.get('generated_by'))}</p>
  </div>
</section>
""".strip()


def _render_footer(metadata: dict[str, Any]) -> str:
    return f"""
<footer class="report-footer">
  <p>{_esc(metadata.get('platform'))} &middot; AI model: {_esc(metadata.get('ai_model'))} &middot; Report ID {_esc(metadata.get('report_id'))}</p>
  <p class="confidential">Confidential &mdash; prepared for internal executive distribution.</p>
</footer>
""".strip()


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def render_html(report: dict[str, Any]) -> str:
    """Render the final report dict into a single self-contained HTML document.

    Deterministic Python, like `_render_markdown` in report_generator.py
    -- this is a data-transformation step over an already-validated
    report dict, not a second AI call and not a place any new number,
    ranking, or finding is decided.

    Raises:
        HTMLRenderError: If `report` is missing a field this renderer
            requires (e.g. it was not produced by
            `generate_executive_report` / `_merge_narrative_into_skeleton`).
    """
    try:
        metadata = report["metadata"]
        body = "\n".join([
            _render_header(metadata),
            _render_executive_alert(report),
            _render_dashboard(report),
            _render_charts(report),
            _render_summary(report),
            _render_risks(report.get("top_business_risks", [])),
            _render_root_causes(report.get("root_causes", [])),
            _render_no_action(report),
            _render_action_plan(report.get("recommended_actions", [])),
            _render_expected_impact(report.get("expected_business_impact", [])),
            _render_departments(report.get("department_breakdown", {})),
            _render_appendix(report.get("appendix", {})),
            _render_footer(metadata),
        ])
    except (KeyError, TypeError, AttributeError) as exc:
        raise HTMLRenderError(f"Report dict is missing a field the HTML renderer requires: {exc}") from exc

    title = escape(f"{metadata.get('report_type', 'Executive Report')} - {metadata.get('platform', '')}")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{_CSS}
</style>
</head>
<body>
  <div class="page">
    <div class="report">
{body}
    </div>
  </div>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Stylesheet (inlined -- no external CDN, no network dependency)
# --------------------------------------------------------------------------
#
# Design system: Solven black-and-gold, in the register of Palantir
# Foundry / Stripe / Linear rather than a printed document. Typography
# is "Inter" first (used if the viewer's OS has it installed) falling
# back to the platform's native UI font stack -- see TYPOGRAPHY below;
# no font file is embedded, keeping the document lightweight and fully
# offline. Spacing is an 8px system throughout. Gold is restricted to a
# small set of attention-guiding roles (brand mark, section labels, the
# executive alert, confidence bars, priority badges) rather than
# applied as general decoration.

_CSS = """
:root {
  --bg: #0C0C0C;
  --surface: #FAFAF8;
  --surface-alt: #EFEEE7;
  --ink: #16161A;
  --ink-muted: #6B6A63;
  --gold: #C9A84C;
  --gold-deep: #8E7229;
  --cream: #EBEAE3;
  --cream-muted: #97958A;
  --line-dark: rgba(201, 168, 76, 0.16);
  --line-light: #E7E5DC;
  --red: #A23B3B;
  --green: #3F6E48;
  --amber: #A8791E;
  --radius: 3px;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--cream);
  font-family: var(--font-sans);
  font-size: 15px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.page { padding: 64px 32px 128px; }

.report { max-width: 1120px; margin: 0 auto; }

.prose { max-width: 68ch; }

/* ---- Header ---- */
.report-header { padding: 0 0 40px; border-bottom: 1px solid var(--line-dark); margin-bottom: 56px; }
.brand-row { display: flex; align-items: center; gap: 12px; margin-bottom: 32px; }
.brand-mark {
  width: 28px; height: 28px; flex: 0 0 auto; border-radius: var(--radius);
  background: var(--gold); color: #14120C; font-weight: 700; font-size: 13px;
  display: inline-flex; align-items: center; justify-content: center;
}
.brand-word { margin: 0; font-size: 12px; font-weight: 700; letter-spacing: 0.14em; color: var(--gold); }
.brand-tagline { margin: 2px 0 0; font-size: 11px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: var(--cream-muted); }
.title { margin: 0 0 14px; font-size: 48px; font-weight: 600; letter-spacing: -0.015em; color: var(--cream); line-height: 1.12; }
.meta-line { margin: 0; font-size: 12px; color: var(--cream-muted); }

/* ---- Section rhythm ---- */
.section { margin-bottom: 56px; }
.section-title {
  font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--gold); margin: 0 0 20px; padding-bottom: 12px;
  border-bottom: 1px solid var(--line-dark);
}

/* ---- Warm-white content surfaces ---- */
.card {
  background: var(--surface); color: var(--ink); border-radius: var(--radius);
  padding: 28px; margin-bottom: 16px; border: 1px solid var(--line-light);
  transition: box-shadow 160ms ease, transform 160ms ease, border-color 160ms ease;
}
.card:hover { box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28); transform: translateY(-1px); border-color: rgba(201, 168, 76, 0.35); }
.card:last-child { margin-bottom: 0; }
.card p { margin: 0 0 12px; }
.card p:last-child { margin-bottom: 0; }

/* ---- Executive Alert ---- */
.alert-card { border-left: 2px solid var(--gold); padding: 32px 36px; }
.alert-headline { font-size: 24px; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); margin-bottom: 14px !important; line-height: 1.3; }
.alert-body { color: var(--ink-muted); font-size: 16px; }
.alert-row { display: flex; gap: 10px; align-items: baseline; padding-top: 14px; margin-top: 14px; border-top: 1px solid var(--line-light); font-size: 14px; }
.alert-row-action { color: var(--ink); }
.alert-tag {
  flex: 0 0 auto; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--gold-deep); white-space: nowrap;
}

/* ---- KPI dashboard ---- */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 16px; }
.kpi-card { text-align: left; }
.kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 10px !important; }
.kpi-score { font-size: 52px; font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; color: var(--ink); margin-bottom: 14px !important; line-height: 1; }
.kpi-track { margin-bottom: 14px; }
.kpi-meta-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.kpi-insight { font-size: 13px; color: var(--ink-muted); font-style: italic; margin-top: 16px !important; padding-top: 14px; border-top: 1px solid var(--line-light); }

.hbar-track { height: 5px; background: var(--surface-alt); border-radius: 2px; overflow: hidden; }
.hbar-fill { height: 100%; border-radius: 2px; background: var(--ink-muted); }
.hbar-fill.status-healthy { background: var(--green); }
.hbar-fill.status-at-risk { background: var(--amber); }
.hbar-fill.status-critical { background: var(--red); }
.hbar-fill.status-unknown { background: var(--ink-muted); }
.hbar-fill.severity-high { background: var(--red); }
.hbar-fill.severity-medium { background: var(--amber); }
.hbar-fill.severity-low { background: var(--ink-muted); }
.hbar-fill.hbar-confidence, .hbar-fill.hbar-neutral { background: var(--gold); }
.hbar-fill.hbar-flag { background: var(--red); }

.confidence-track { margin: 10px 0 14px; }

.fact-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.fact-card { margin-bottom: 0; }
.fact-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 8px !important; }
.fact-value { font-size: 19px; font-weight: 600; color: var(--ink); margin-bottom: 0 !important; }

/* ---- Charts ---- */
.chart-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.chart-card { margin-bottom: 0; }
.chart-title { font-size: 12px; font-weight: 700; letter-spacing: 0.04em; color: var(--ink); margin-bottom: 20px !important; }

.hbar-row { margin-bottom: 16px; }
.hbar-row:last-child { margin-bottom: 0; }
.hbar-row-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 6px; font-size: 13px; }
.hbar-label { color: var(--ink); font-weight: 500; }
.hbar-value { color: var(--ink-muted); font-variant-numeric: tabular-nums; font-size: 12px; }
.hbar-meta { margin-top: 6px; font-size: 11px; }
.hbar-flag-label { color: var(--red); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; font-size: 10px; }

.donut-wrap { display: flex; align-items: center; gap: 24px; }
.donut { position: relative; width: 116px; height: 116px; border-radius: 50%; flex: 0 0 auto; }
.donut::after { content: ""; position: absolute; inset: 16px; border-radius: 50%; background: var(--surface); }
.donut-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 1; }
.donut-value { font-size: 22px; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; }
.donut-caption { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-muted); }
.donut-legend { display: flex; flex-direction: column; gap: 10px; font-size: 13px; color: var(--ink-muted); }
.legend-item { display: flex; align-items: center; gap: 8px; }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; }
.legend-dot-primary { background: var(--red); }
.legend-dot-secondary { background: var(--surface-alt); border: 1px solid var(--line-light); }

/* ---- Badges ---- */
.badge {
  display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 4px 8px; border-radius: var(--radius); background: var(--surface-alt); color: var(--ink-muted);
}
.badge.status-healthy { background: rgba(63, 110, 72, 0.12); color: var(--green); }
.badge.status-at-risk { background: rgba(168, 121, 30, 0.14); color: var(--amber); }
.badge.status-critical { background: rgba(162, 59, 59, 0.12); color: var(--red); }
.badge.severity-high { background: rgba(162, 59, 59, 0.12); color: var(--red); }
.badge.severity-medium { background: rgba(168, 121, 30, 0.14); color: var(--amber); }
.badge.severity-low { background: rgba(107, 106, 99, 0.14); color: var(--ink-muted); }
.badge-outline { background: transparent; border: 1px solid var(--line-light); color: var(--ink-muted); }
.badge-priority { background: rgba(201, 168, 76, 0.14); color: var(--gold-deep); }
.confidence-badge { font-size: 12px; color: var(--gold-deep); font-weight: 600; white-space: nowrap; }

/* ---- Trend indicators ---- */
.trend { font-size: 11px; font-weight: 600; }
.trend-up { color: var(--green); }
.trend-down { color: var(--red); }
.trend-flat { color: var(--ink-muted); }
.trend-unavailable { color: var(--cream-muted); font-weight: 400; font-style: italic; }
.kpi-meta-row .trend-unavailable { color: var(--ink-muted); }

/* ---- Executive summary ---- */
.lead { font-size: 17px; color: var(--ink); }
.summary-grid { margin-top: 4px; }
.summary-row {
  display: grid; grid-template-columns: 180px 1fr; gap: 16px; padding: 12px 0;
  border-bottom: 1px solid var(--line-dark); font-size: 14px; color: var(--cream-muted);
}
.summary-row:last-child { border-bottom: none; }
.summary-domain { color: var(--gold); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }

/* ---- Risks ---- */
.risk-header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.risk-rank {
  flex: 0 0 auto; width: 24px; height: 24px; border-radius: var(--radius); background: var(--ink);
  color: var(--surface); font-size: 12px; font-weight: 700; display: inline-flex; align-items: center; justify-content: center;
}
.risk-title { font-size: 17px; font-weight: 600; letter-spacing: -0.005em; flex: 1 1 auto; }
.risk-badges { display: flex; gap: 6px; flex: 0 0 auto; }
.risk-evidence, .risk-impact { font-size: 14px; color: var(--ink-muted); }
.risk-impact strong, .risk-evidence strong { color: var(--ink); }

/* ---- Root causes ---- */
.root-cause-header { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 4px; }
.root-cause-heading { display: flex; align-items: center; gap: 10px; }
.root-cause-note { color: var(--ink-muted); font-size: 14px; }
.root-cause-impact { font-size: 14px; }
.root-cause-owner { font-size: 12px; color: var(--ink-muted); font-weight: 500; }
.evidence-list { margin: 0 0 12px; padding-left: 18px; font-size: 14px; color: var(--ink-muted); }
.evidence-list li { margin-bottom: 4px; }

/* ---- If No Action ---- */
.no-action-card { border-left: 2px solid var(--red); }

/* ---- Action plan timeline ---- */
.timeline-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; align-items: start; }
.timeline-column { display: flex; flex-direction: column; }
.timeline-title {
  font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--gold);
  margin: 0 0 14px; padding-bottom: 10px; border-bottom: 1px solid var(--line-dark);
}
.action-card { margin-bottom: 14px; }
.action-card-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px; }
.action-title { font-size: 15px; font-weight: 600; margin: 0 !important; }
.action-meta { font-size: 11px; color: var(--ink-muted); margin-bottom: 12px !important; }
.action-reason { font-size: 13px; }
.action-rationale { font-size: 13px; font-style: italic; color: var(--ink-muted); }
.action-outcome { font-size: 13px; }

/* ---- Expected impact ---- */
.impact-list { margin: 0; padding-left: 18px; }
.impact-list li { margin-bottom: 8px; }
.impact-list li:last-child { margin-bottom: 0; }

/* ---- Department breakdown ---- */
.dept-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.dept-card { margin-bottom: 0; position: relative; }
.dept-index { position: absolute; top: 24px; right: 26px; font-size: 12px; font-weight: 700; color: var(--line-light); font-variant-numeric: tabular-nums; }
.dept-title { font-size: 16px; font-weight: 600; margin-bottom: 10px !important; padding-right: 32px; }
.dept-text { font-size: 14px; color: var(--ink-muted); }

/* ---- Appendix ---- */
.appendix-heading { font-size: 13px; font-weight: 700; margin: 20px 0 8px; }
.appendix-heading:first-child { margin-top: 0; }
.appendix-card p { font-size: 14px; color: var(--ink-muted); }
.limitations-list { margin: 0; padding-left: 18px; font-size: 13px; color: var(--ink-muted); }
.limitations-list li { margin-bottom: 6px; }
.appendix-footnote { font-size: 12px; color: var(--ink-muted); font-style: italic; margin-top: 18px !important; }

.empty-state { color: var(--cream-muted); font-style: italic; font-size: 14px; }

/* ---- Footer ---- */
.report-footer { margin-top: 64px; padding-top: 24px; border-top: 1px solid var(--line-dark); font-size: 12px; color: var(--cream-muted); }
.report-footer p { margin: 0 0 4px; }
.confidential { letter-spacing: 0.04em; text-transform: uppercase; font-size: 11px; }

/* ==========================================================
   Responsive -- desktop-first: base styles above target desktop;
   tablet narrows the grids; mobile collapses to a single column.
   ========================================================== */

/* Tablet */
@media (max-width: 1024px) {
  .page { padding: 48px 24px 96px; }
  .chart-grid { grid-template-columns: 1fr; }
  .timeline-grid { grid-template-columns: 1fr 1fr; }
}

/* Mobile */
@media (max-width: 768px) {
  .page { padding: 32px 16px 64px; }
  .title { font-size: 32px; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .fact-grid { grid-template-columns: 1fr; }
  .dept-grid { grid-template-columns: 1fr; }
  .timeline-grid { grid-template-columns: 1fr; }
  .summary-row { grid-template-columns: 1fr; gap: 4px; }
  .donut-wrap { flex-direction: column; align-items: flex-start; }
}

@media (max-width: 480px) {
  .kpi-grid { grid-template-columns: 1fr; }
  .alert-card { padding: 24px; }
}

/* ---- Print ---- */
@media print {
  @page { margin: 1.6cm; }
  body { background: #ffffff; color: #16161A; font-size: 12px; }
  .page { padding: 0; background: #ffffff; }
  .report { max-width: 100%; }
  .report-header { border-bottom-color: #C9A84C; }
  .brand-word, .title { color: #16161A; }
  .meta-line, .report-footer { color: #4A4536; }
  .section-title { color: #8E7229; border-bottom-color: #C9A84C; }
  .card { background: #ffffff; border: 1px solid #D8CFB4; break-inside: avoid; box-shadow: none; transform: none; }
  .card:hover { box-shadow: none; transform: none; border-color: #D8CFB4; }
  .trend-unavailable { color: #6B6A63; }
  .chart-grid { grid-template-columns: repeat(2, 1fr); }
  .timeline-grid { grid-template-columns: repeat(3, 1fr); }
  .action-card, .risk-card, .kpi-card, .dept-card, .chart-card, .root-cause-card { break-inside: avoid; }
  .timeline-column { break-inside: avoid-page; }
}
"""
