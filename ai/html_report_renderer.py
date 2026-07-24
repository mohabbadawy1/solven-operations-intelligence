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

The document is designed document-first, not website-first: a single
portrait content column, a dark cover panel followed by a light
editorial body, and a typographic/spacing scale built for A4 print
(see ai/pdf_report_renderer.py for how it is printed). It is still a
plain self-contained HTML file with all CSS inlined and no external
fonts/scripts/CDNs, so it opens correctly from disk with no internet
connection and reads reasonably in a browser -- but the print
rendering in @media print is the primary target this design is built
for, not a secondary override of a desktop layout.

All dynamic text is passed through `html.escape` before being placed
in the document -- the report contains free-text narrative from an LLM
and evidence strings copied from source data, neither of which is
trusted to be free of `<`, `&`, or `"` characters.

Layout is organized as one builder function per section: cover
(branding + title + executive alert, dark), dashboard (KPI cards),
performance charts, executive summary, business risks, root causes,
if-no-action, action plan, expected impact, department breakdown,
appendix, executive conclusion, closing footer. A small set of
chart/component primitives (`_hbar_row`, `_donut_html`, `_badge_html`,
`_meta_row`, `_info_row`, ...) are shared across those builders so no
section reimplements its own bar, badge, or label/value row markup.
"""

from __future__ import annotations

from datetime import datetime
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
ACTION_HORIZON_CLASS = {
    "Immediate (0-30 Days)": "phase-immediate",
    "Near-Term (30-60 Days)": "phase-near-term",
    "Strategic (60-120 Days)": "phase-strategic",
}


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


def _fmt_datetime(value: str | None) -> str:
    """An ISO-8601 timestamp -> "24 Jul 2026, 09:15 UTC".

    Pure presentation of the existing metadata.generated_at value
    (always produced by `datetime.now(timezone.utc).isoformat()`
    upstream) -- never a new value. Falls back to the raw string if it
    can't be parsed, so a future change to how the timestamp is
    produced degrades gracefully instead of raising.
    """
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    formatted = parsed.strftime("%d %b %Y, %H:%M")
    if formatted.startswith("0"):
        formatted = formatted[1:]
    return f"{formatted} UTC"


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


def _meta_row(pairs: list[tuple[str, Any]]) -> str:
    """A compact row of small label/value chips (Owner, Difficulty, Timeline, Confidence, ...).

    Skips any pair whose value is empty/None so an incomplete action or
    root cause never renders a blank chip.
    """
    items = "".join(
        f'<div class="meta-item"><span class="meta-k">{_esc(key)}</span><span class="meta-v">{_esc(value)}</span></div>'
        for key, value in pairs
        if value not in (None, "")
    )
    return f'<div class="meta-row">{items}</div>' if items else ""


def _info_row(label: str, value: Any) -> str:
    """A single labeled row (e.g. "Evidence" / "Business Impact") -- no inline bold-prefix prose."""
    if value in (None, ""):
        return ""
    return f'<div class="info-row"><span class="info-label">{_esc(label)}</span><span class="info-value">{_esc(value)}</span></div>'


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
      <div class="donut" style="background: conic-gradient(var(--red) 0% {primary_percent}%, var(--line-soft) {primary_percent}% 100%)">
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
# Cover (branding + title + executive alert, dark panel, page 1 only)
# --------------------------------------------------------------------------


def _render_cover(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    brand, tagline = _brand_lockup(metadata.get("platform"))
    monogram = _esc(brand[:1]) if brand else ""
    tagline_html = f'<p class="cover-tagline">{_esc(tagline)}</p>' if tagline else ""

    top_action_title = report.get("highest_priority_initiative")
    top_action = next(
        (action for action in report.get("recommended_actions", []) if action.get("title") == top_action_title), None
    )
    top_priority_html = ""
    if top_action:
        top_priority_html = f"""
  <div class="cover-row">
    <span class="cover-tag">Top Priority</span>
    <span class="cover-row-text">{_esc(top_action.get('title'))} &mdash; Owner: {_esc(top_action.get('owner'))} &middot; {_esc(top_action.get('horizon'))}</span>
  </div>"""

    return f"""
<section class="cover">
  <div class="cover-brand">
    <span class="cover-mark">{monogram}</span>
    <div>
      <p class="cover-word">{_esc(brand)}</p>
      {tagline_html}
    </div>
  </div>
  <h1 class="cover-title">{_esc(metadata.get('report_type'))}</h1>
  <div class="cover-meta-row">
    <span class="cover-date">{_esc(_fmt_datetime(metadata.get('generated_at')))}</span>
    <span class="cover-id">Report {_esc(metadata.get('report_id'))}</span>
  </div>
  <div class="cover-rule"></div>
  <p class="cover-headline">{_esc(report.get('executive_headline'))}</p>
  <p class="cover-body">{_esc(report.get('why_it_matters'))}</p>
  <div class="cover-row">
    <span class="cover-tag">If Nothing Changes</span>
    <span class="cover-row-text">{_esc(report.get('consequence_of_inaction'))}</span>
  </div>{top_priority_html}
</section>
""".strip()


# --------------------------------------------------------------------------
# Executive Dashboard (KPI cards)
# --------------------------------------------------------------------------


def _render_kpi_card(domain: str, block: dict[str, Any]) -> str:
    status = block.get("status") or "Unknown"
    status_class = STATUS_CLASS.get(status, DEFAULT_STATUS_CLASS)
    score = block.get("score")
    score_unit_html = '<span class="kpi-score-unit"> / 100</span>' if score is not None else ""
    insight_html = f'<p class="kpi-insight">{_esc(block["narrative"])}</p>' if block.get("narrative") else ""
    return f"""
    <div class="card kpi-card">
      <p class="kpi-label">{_esc(_label(domain))}</p>
      <p class="kpi-score">{_esc(_fmt_score(score))}{score_unit_html}</p>
      <div class="hbar-track kpi-track"><div class="hbar-fill {status_class}" style="width:{_percent(score)}%"></div></div>
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
    donut = _donut_html(negative, f"Negative · {negative}%", f"Non-negative · {non_negative}%", "Negative")
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
        f'<span class="summary-text">{_esc(block.get("narrative"))}</span></div>'
        for domain, block in report.get("overall_business_health", {}).items()
        if block.get("narrative")
    )
    return f"""
<section class="section">
  <h2 class="section-title">Executive Summary</h2>
  <div class="card card-prose">
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
      <div class="info-stack">
        {_info_row("Evidence", risk.get('evidence'))}
        {_info_row("Business Impact", risk.get('business_impact'))}
      </div>
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
            meta_html = _meta_row([("Owner", root_cause.get("owner")), ("Confidence", _fmt_confidence(root_cause.get("confidence")))])
            cards.append(f"""
    <div class="card root-cause-card">
      <div class="root-cause-header">
        <div class="root-cause-heading">
          {priority_badge}
          <span class="risk-title">{_esc(root_cause.get('title'))}</span>
        </div>
      </div>
      <div class="hbar-track confidence-track"><div class="hbar-fill hbar-confidence" style="width:{_percent((root_cause.get('confidence') or 0) * 100)}%"></div></div>
      {note_html}
      {evidence_html}
      <div class="info-stack">
        {_info_row("Business Impact", root_cause.get('business_impact'))}
      </div>
      {meta_html}
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
  <div class="card card-prose no-action-card">
    {_paragraphs_html(report.get('if_no_action_narrative', ''), css_class="prose")}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# 90-Day Action Plan
# --------------------------------------------------------------------------


def _render_action_card(action: dict[str, Any]) -> str:
    rationale_html = (
        f'<p class="action-rationale">{_esc(action.get("executive_rationale"))}</p>'
        if action.get("executive_rationale") else ""
    )
    meta_html = _meta_row([
        ("Owner", action.get("owner")),
        ("Difficulty", action.get("difficulty")),
        ("Timeline", action.get("estimated_timeframe")),
        ("Confidence", _fmt_confidence(action.get("confidence"))),
    ])
    return f"""
      <div class="card action-card">
        <div class="action-card-head">
          {_badge_html(f"P{action.get('priority')}", "badge-priority")}
          <p class="action-title">{_esc(action.get('title'))}</p>
        </div>
        {meta_html}
        <p class="action-reason">{_esc(action.get('reason'))}</p>
        {rationale_html}
        <div class="info-stack">
          {_info_row("Expected Outcome", action.get('expected_business_impact'))}
        </div>
      </div>""".strip()


def _render_action_plan(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return """
<section class="section">
  <h2 class="section-title">90-Day Action Plan</h2>
  <p class="empty-state">No priority actions were identified this period.</p>
</section>
""".strip()

    phase_blocks = []
    for horizon_label in ACTION_HORIZON_ORDER:
        horizon_actions = [action for action in actions if action.get("horizon") == horizon_label]
        if not horizon_actions:
            continue
        cards = "\n".join(_render_action_card(action) for action in horizon_actions)
        phase_class = ACTION_HORIZON_CLASS.get(horizon_label, "")
        phase_blocks.append(f"""
    <div class="phase-block {phase_class}">
      <div class="phase-head">
        <span class="phase-dot"></span>
        <p class="phase-title">{_esc(horizon_label)}</p>
      </div>
      <div class="phase-cards">
{cards}
      </div>
    </div>""".strip())

    return f"""
<section class="section">
  <h2 class="section-title">90-Day Action Plan</h2>
  <div class="phase-stack">
{chr(10).join(phase_blocks)}
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
        f'    <li><span class="impact-area">{_esc(item.get("area"))}</span><span class="impact-text">{_esc(item.get("expected_improvement"))}</span></li>'
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
<section class="section section-appendix">
  <h2 class="section-title">Appendix</h2>
  <div class="card card-prose appendix-card">
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


# --------------------------------------------------------------------------
# Executive Conclusion
# --------------------------------------------------------------------------


def _render_conclusion(report: dict[str, Any]) -> str:
    """Closing executive summary, assembled from fields already on `report`.

    Like every other section here, this invents no new number or
    finding -- it re-composes overall_business_health, top_business_risks,
    root_causes, and recommended_actions (already computed and narrated
    upstream) into a final "so what" read for a reader who only has time
    for the first and last page.
    """
    health = report.get("overall_business_health", {}) or {}
    statuses = [block.get("status") for block in health.values() if block.get("status")]
    healthy = statuses.count("Healthy")
    at_risk = statuses.count("At Risk")
    critical = statuses.count("Critical")
    total = len(statuses)

    if total:
        health_line = (
            f"Across {total} monitored business domain{'s' if total != 1 else ''} this period, "
            f"{healthy} {'is' if healthy == 1 else 'are'} Healthy, {at_risk} At Risk, "
            f"and {critical} Critical."
        )
    else:
        health_line = "No business health domains were scored this period."

    findings_items = [rc.get("title") for rc in (report.get("root_causes") or [])[:3] if rc.get("title")]
    if not findings_items:
        findings_items = [risk.get("title") for risk in (report.get("top_business_risks") or [])[:3] if risk.get("title")]
    findings_html = "".join(f"<li>{_esc(item)}</li>" for item in findings_items) or (
        '<li class="empty-state">No material findings were identified this period.</li>'
    )

    risks = sorted(report.get("top_business_risks") or [], key=lambda risk: risk.get("rank") or 0)[:3]
    risks_html = "".join(
        f'<li>{_esc(risk.get("title"))} {_badge_html(risk.get("severity") or "LOW", SEVERITY_CLASS.get(risk.get("severity"), DEFAULT_SEVERITY_CLASS))}</li>'
        for risk in risks
    ) or '<li class="empty-state">No significant business risks were identified this period.</li>'

    actions = sorted(report.get("recommended_actions") or [], key=lambda action: action.get("priority") or 99)[:3]
    actions_html = "".join(
        f'<li><strong>{_esc(action.get("title"))}</strong> &mdash; Owner: {_esc(action.get("owner"))}</li>'
        for action in actions
    ) or '<li class="empty-state">No priority actions were identified this period.</li>'

    if critical:
        outlook = (
            f"Immediate executive attention is warranted: {critical} domain{'s' if critical != 1 else ''} "
            f"{'is' if critical == 1 else 'are'} rated Critical this period. Executing the highest-impact "
            "recommendations above is the primary lever to stabilize operations."
        )
    elif at_risk:
        outlook = (
            f"Operations are broadly stable, but {at_risk} domain{'s' if at_risk != 1 else ''} "
            f"{'requires' if at_risk == 1 else 'require'} close monitoring. Sustained execution of the "
            "recommended actions above should keep the business on a Healthy trajectory."
        )
    elif total:
        outlook = (
            "All monitored business domains are Healthy this period. Continued execution of the "
            "recommended actions above will help sustain this position."
        )
    else:
        outlook = "Insufficient data was available this period to project an overall outlook."

    return f"""
<section class="section">
  <h2 class="section-title">Executive Conclusion</h2>
  <div class="card conclusion-card">
    <p class="conclusion-headline">{_esc(health_line)}</p>
    <div class="conclusion-block">
      <p class="conclusion-label">Most Significant Findings</p>
      <ul class="conclusion-list">{findings_html}</ul>
    </div>
    <div class="conclusion-block">
      <p class="conclusion-label">Primary Business Risks</p>
      <ul class="conclusion-list">{risks_html}</ul>
    </div>
    <div class="conclusion-block">
      <p class="conclusion-label">Highest-Impact Recommendations</p>
      <ul class="conclusion-list">{actions_html}</ul>
    </div>
    <p class="conclusion-outlook"><strong>Overall Outlook:</strong> {_esc(outlook)}</p>
  </div>
</section>
""".strip()


def _render_footer(metadata: dict[str, Any]) -> str:
    return f"""
<footer class="report-footer">
  <p>{_esc(metadata.get('platform'))} &middot; AI model: {_esc(metadata.get('ai_model'))}</p>
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
            _render_cover(report),
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
            _render_conclusion(report),
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
  <main class="report">
{body}
  </main>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Stylesheet (inlined -- no external CDN, no network dependency)
# --------------------------------------------------------------------------
#
# Design system: a dark title panel (Solven black/gold) opens the
# report; every page after it is a light editorial document (dark ink
# on warm off-white) in the register of a McKinsey/Palantir/Stripe
# printed report rather than a dark web app. Typography is set in `pt`
# (an absolute, print-accurate unit) against the hierarchy: cover
# ~30pt, section titles ~16pt, subsection/item titles ~11.5-12pt, body
# ~10pt, labels/metadata ~7.5-9pt. Spacing sticks to a 4/8/12/16/24/32
# px scale throughout. "Inter" is used if the viewer's OS has it,
# falling back to the platform's native UI font stack; no font file is
# embedded, keeping the document lightweight and fully offline.
# Page geometry (A4 portrait, margins, footer page numbers) is owned by
# ai/pdf_report_renderer.py's Playwright call, not by this stylesheet
# -- this file only ever adds pagination *behavior* (break-inside,
# widows/orphans) inside @media print, never page size or margin.

_CSS = """
:root {
  --cover-bg: #0C0C0C;
  --cover-ink: #F5F3EC;
  --cover-ink-muted: #A6A398;
  --cover-gold: #C9A84C;
  --cover-line: rgba(201, 168, 76, 0.22);

  --page-bg: #FAF9F5;
  --card-bg: #FFFFFF;
  --ink: #17171B;
  --ink-muted: #5B5A54;
  --ink-faint: #85837A;
  --gold: #9C7A24;
  --gold-soft: rgba(156, 122, 36, 0.12);
  --line: #E7E4DA;
  --line-soft: #EFEDE4;
  --red: #9B3A34;
  --red-soft: rgba(155, 58, 52, 0.10);
  --green: #2F6B3F;
  --green-soft: rgba(47, 107, 63, 0.10);
  --amber: #93650F;
  --amber-soft: rgba(147, 101, 15, 0.12);
  --radius: 4px;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

* {
  box-sizing: border-box;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
  color-adjust: exact;
}

html { background: var(--page-bg); }

body {
  margin: 0;
  background: var(--page-bg);
  color: var(--ink);
  font-family: var(--font-sans);
  font-size: 10pt;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

.report { max-width: 760px; margin: 0 auto; padding: 32px 24px 48px; }
.prose { max-width: 64ch; }

h1, h2, h3, h4 { margin: 0; }

/* ---- Cover ---- */
.cover {
  background: var(--cover-bg);
  color: var(--cover-ink);
  border-radius: 6px;
  padding: 32px 32px 28px;
  margin: 24px 0 32px;
}
.cover-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 28px; }
.cover-mark {
  width: 26px; height: 26px; flex: 0 0 auto; border-radius: 4px;
  background: var(--cover-gold); color: #14120C; font-weight: 700; font-size: 8.5pt;
  display: inline-flex; align-items: center; justify-content: center;
}
.cover-word { margin: 0; font-size: 8.5pt; font-weight: 700; letter-spacing: 0.16em; color: var(--cover-gold); }
.cover-tagline { margin: 2px 0 0; font-size: 7.5pt; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--cover-ink-muted); }
.cover-title { font-size: 30pt; font-weight: 600; letter-spacing: -0.01em; line-height: 1.15; color: var(--cover-ink); max-width: 20ch; margin-bottom: 16px; }
.cover-meta-row { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.cover-date { font-size: 9pt; color: var(--cover-ink-muted); }
.cover-id { font-size: 7.5pt; color: var(--cover-ink-muted); opacity: 0.85; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.cover-rule { height: 1px; background: var(--cover-line); margin-bottom: 24px; }
.cover-headline { font-size: 15pt; font-weight: 600; line-height: 1.35; color: var(--cover-ink); max-width: 44ch; margin-bottom: 12px; }
.cover-body { font-size: 10pt; line-height: 1.55; color: var(--cover-ink-muted); max-width: 58ch; margin-bottom: 4px; }
.cover-row { display: flex; gap: 12px; padding-top: 16px; margin-top: 16px; border-top: 1px solid var(--cover-line); font-size: 9.5pt; color: var(--cover-ink); }
.cover-tag { flex: 0 0 auto; width: 118px; font-size: 7.5pt; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--cover-gold); }
.cover-row-text { flex: 1 1 auto; }

/* ---- Section rhythm ---- */
.section { margin-bottom: 24px; }
.section-title {
  font-size: 16pt; font-weight: 600; letter-spacing: -0.005em; color: var(--ink);
  margin: 0 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--line);
}

/* ---- Cards ---- */
.card {
  background: var(--card-bg); color: var(--ink); border-radius: var(--radius);
  padding: 16px; margin-bottom: 8px; border: 1px solid var(--line);
}
.card:last-child { margin-bottom: 0; }
.card p { margin: 0 0 8px; }
.card p:last-child { margin-bottom: 0; }
.card-prose { padding: 24px; }

/* ---- KPI dashboard ----
   Flexbox, not CSS Grid: Chromium's print pagination fragments a
   wrapped flex row far more predictably than a grid row, which tends
   to get pushed to the next page as one atomic unit even when it
   would fit in the remaining space. */
.kpi-grid { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; }
.kpi-card { flex: 1 1 calc(50% - 8px); }
.kpi-label { font-size: 8pt; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 8px; }
.kpi-score { font-size: 28pt; font-weight: 600; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; color: var(--ink); margin-bottom: 8px; line-height: 1; }
.kpi-score-unit { font-size: 10pt; font-weight: 500; color: var(--ink-faint); }
.kpi-track { margin-bottom: 8px; }
.kpi-meta-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.kpi-insight { font-size: 9pt; color: var(--ink-muted); margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--line-soft); }

.hbar-track { height: 5px; background: var(--line-soft); border-radius: 2px; overflow: hidden; }
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

.confidence-track { margin: 8px 0 12px; }

.fact-grid { display: flex; flex-direction: column; gap: 8px; }
.fact-card { margin-bottom: 0; }
.fact-label { font-size: 8pt; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 6px; }
.fact-value { font-size: 12pt; font-weight: 600; line-height: 1.4; color: var(--ink); margin-bottom: 0; }

/* ---- Charts ---- */
.chart-grid { display: flex; flex-wrap: wrap; gap: 16px; }
.chart-card { flex: 1 1 calc(50% - 8px); margin-bottom: 0; padding: 12px; }
.chart-grid .chart-card:last-child:nth-child(odd) { flex-basis: 100%; }
.chart-title { font-size: 12pt; font-weight: 600; color: var(--ink); margin-bottom: 12px; }

.hbar-row { margin-bottom: 8px; }
.hbar-row:last-child { margin-bottom: 0; }
.hbar-row-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 4px; font-size: 9pt; }
.hbar-label { color: var(--ink); font-weight: 500; }
.hbar-value { color: var(--ink-muted); font-variant-numeric: tabular-nums; font-size: 8.5pt; }
.hbar-meta { margin-top: 4px; font-size: 7.5pt; }
.hbar-flag-label { color: var(--red); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }

.donut-wrap { display: flex; align-items: center; gap: 20px; }
.donut { position: relative; width: 100px; height: 100px; border-radius: 50%; flex: 0 0 auto; }
.donut::after { content: ""; position: absolute; inset: 14px; border-radius: 50%; background: var(--card-bg); }
.donut-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 1; }
.donut-value { font-size: 15pt; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; }
.donut-caption { font-size: 7.5pt; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-muted); }
.donut-legend { display: flex; flex-direction: column; gap: 8px; font-size: 9pt; color: var(--ink-muted); }
.legend-item { display: flex; align-items: center; gap: 8px; }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; }
.legend-dot-primary { background: var(--red); }
.legend-dot-secondary { background: var(--line-soft); border: 1px solid var(--line); }

/* ---- Badges ---- */
.badge {
  display: inline-block; font-size: 7.5pt; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 3px 7px; border-radius: var(--radius); background: var(--line-soft); color: var(--ink-muted);
}
.badge.status-healthy { background: var(--green-soft); color: var(--green); }
.badge.status-at-risk { background: var(--amber-soft); color: var(--amber); }
.badge.status-critical { background: var(--red-soft); color: var(--red); }
.badge.severity-high { background: var(--red-soft); color: var(--red); }
.badge.severity-medium { background: var(--amber-soft); color: var(--amber); }
.badge.severity-low { background: var(--line-soft); color: var(--ink-muted); }
.badge-outline { background: transparent; border: 1px solid var(--line); color: var(--ink-muted); }
.badge-priority { background: var(--gold-soft); color: var(--gold); }

/* ---- Trend indicators ---- */
.trend { font-size: 8pt; font-weight: 600; }
.trend-up { color: var(--green); }
.trend-down { color: var(--red); }
.trend-flat { color: var(--ink-muted); }
.trend-unavailable { color: var(--ink-faint); font-weight: 400; font-style: italic; }

/* ---- Label/value rows (meta chips + info rows) ---- */
.meta-row { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 8px 0; }
.meta-item { display: flex; flex-direction: column; gap: 2px; }
.meta-k { font-size: 7.5pt; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-faint); }
.meta-v { font-size: 9pt; color: var(--ink); }

.info-stack { margin-top: 8px; }
.info-row { display: flex; gap: 12px; padding: 8px 0; border-top: 1px solid var(--line-soft); font-size: 9.5pt; }
.info-stack .info-row:first-child { border-top: none; padding-top: 0; }
.info-label { flex: 0 0 auto; width: 96px; font-size: 7.5pt; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-faint); }
.info-value { flex: 1 1 auto; color: var(--ink-muted); }

/* ---- Executive summary ---- */
.lead { font-size: 10.5pt; color: var(--ink); }
.summary-grid { display: table; width: 100%; table-layout: fixed; border-collapse: collapse; margin-top: 4px; }
.summary-row { display: table-row; }
.summary-domain, .summary-text {
  display: table-cell; padding: 10px 0; border-bottom: 1px solid var(--line-soft);
  font-size: 9.5pt; color: var(--ink-muted); vertical-align: top;
}
.summary-row:last-child .summary-domain, .summary-row:last-child .summary-text { border-bottom: none; }
.summary-domain { width: 150px; padding-right: 16px; color: var(--gold); font-weight: 600; font-size: 8pt; text-transform: uppercase; letter-spacing: 0.05em; }

/* ---- Risks ---- */
.risk-header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
.risk-rank {
  flex: 0 0 auto; width: 22px; height: 22px; border-radius: var(--radius); background: var(--ink);
  color: var(--card-bg); font-size: 9pt; font-weight: 700; display: inline-flex; align-items: center; justify-content: center;
}
.risk-title { font-size: 11.5pt; font-weight: 600; letter-spacing: -0.005em; flex: 1 1 auto; }
.risk-badges { display: flex; gap: 6px; flex: 0 0 auto; }

/* ---- Root causes ---- */
.root-cause-header { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 4px; }
.root-cause-heading { display: flex; align-items: center; gap: 10px; }
.root-cause-note { font-size: 9.5pt; color: var(--ink-muted); }
.evidence-list { margin: 0 0 8px; padding-left: 16px; font-size: 9.5pt; color: var(--ink-muted); }
.evidence-list li { margin-bottom: 4px; }

/* ---- If No Action ---- */
.no-action-card { border-left: 2px solid var(--red); }

/* ---- Action plan ---- */
.phase-stack { display: flex; flex-direction: column; gap: 20px; }
.phase-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.phase-dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; background: var(--ink-muted); }
.phase-immediate .phase-dot { background: var(--red); }
.phase-near-term .phase-dot { background: var(--amber); }
.phase-strategic .phase-dot { background: var(--gold); }
.phase-title { font-size: 12pt; font-weight: 600; color: var(--ink); margin: 0; }
.phase-cards { display: flex; flex-direction: column; gap: 12px; }
.action-card { margin-bottom: 0; }
.action-card-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 4px; }
.action-title { font-size: 11pt; font-weight: 600; margin: 0; }
.action-reason { font-size: 9.5pt; }
.action-rationale { font-size: 9.5pt; color: var(--ink-muted); }

/* ---- Expected impact ---- */
.impact-list { list-style: none; margin: 0; padding: 0; }
.impact-list li { display: flex; gap: 12px; padding: 8px 0; border-top: 1px solid var(--line-soft); font-size: 9.5pt; }
.impact-list li:first-child { border-top: none; padding-top: 0; }
.impact-area { flex: 0 0 auto; width: 150px; font-weight: 600; color: var(--ink); }
.impact-text { flex: 1 1 auto; color: var(--ink-muted); }

/* ---- Department breakdown ---- */
.dept-grid { display: flex; flex-wrap: wrap; gap: 16px; }
.dept-card { flex: 1 1 calc(50% - 8px); margin-bottom: 0; position: relative; }
.dept-grid .dept-card:last-child:nth-child(odd) { flex-basis: 100%; }
.dept-index { position: absolute; top: 16px; right: 16px; font-size: 8pt; font-weight: 700; color: var(--line); font-variant-numeric: tabular-nums; }
.dept-title { font-size: 11.5pt; font-weight: 600; margin-bottom: 8px; padding-right: 28px; }
.dept-text { font-size: 9.5pt; color: var(--ink-muted); }

/* ---- Appendix ---- */
.appendix-heading { font-size: 12pt; font-weight: 600; margin: 16px 0 6px; }
.appendix-heading:first-child { margin-top: 0; }
.appendix-card p { font-size: 9.5pt; color: var(--ink-muted); }
.limitations-list { margin: 0; padding-left: 16px; font-size: 9pt; color: var(--ink-muted); }
.limitations-list li { margin-bottom: 4px; }
.appendix-footnote { font-size: 8pt; color: var(--ink-faint); margin-top: 16px; }

.empty-state { color: var(--ink-faint); font-style: italic; font-size: 9.5pt; }

/* ---- Executive Conclusion ---- */
.conclusion-card { border-left: 2px solid var(--gold); }
.conclusion-headline { font-size: 12pt; font-weight: 600; color: var(--ink); margin-bottom: 16px; }
.conclusion-block { margin-bottom: 12px; }
.conclusion-block:last-of-type { margin-bottom: 0; }
.conclusion-label { font-size: 8pt; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--gold); margin-bottom: 6px; }
.conclusion-list { margin: 0; padding-left: 16px; font-size: 9.5pt; color: var(--ink-muted); }
.conclusion-list li { margin-bottom: 4px; }
.conclusion-list li:last-child { margin-bottom: 0; }
.conclusion-outlook { font-size: 9.5pt; color: var(--ink); padding-top: 12px; margin-top: 4px; border-top: 1px solid var(--line-soft); }

/* ---- Closing footer ---- */
.report-footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 8pt; color: var(--ink-faint); }
.report-footer p { margin: 0 0 4px; }
.confidential { letter-spacing: 0.04em; text-transform: uppercase; font-size: 7.5pt; }

/* ==========================================================
   Responsive -- this design is single-column and print-first
   already, so only mobile screen widths need adjustment.
   ========================================================== */
@media screen and (max-width: 560px) {
  .report { padding: 20px 16px 32px; }
  .cover-title { font-size: 24pt; }
  .kpi-card, .dept-card, .chart-card { flex-basis: 100%; }
}

/* ---- Print ----
   Page size, margins, and the repeating footer with page numbers are
   owned by ai/pdf_report_renderer.py's page.pdf() call, not here -- a
   PDF-level margin (unlike the old zero-margin/`.page`-padding
   approach this file used to require) now paints the page background
   correctly on its own, so this block only ever adds pagination
   *behavior*. */
@page {
  size: A4 portrait;
  margin: 14mm 14mm 16mm 14mm;
}

@media print {
  .report { max-width: none; padding: 0; margin: 0; width: 100%; }

  h1, h2, h3, h4 { break-after: avoid; page-break-after: avoid; }
  p, li { orphans: 3; widows: 3; }

  .cover { break-after: page; page-break-after: always; margin: 0; }
  .section-appendix { break-before: page; page-break-before: always; }

  .section { break-inside: auto; }
  /* Only small, structured cards get a hard no-split rule -- letting a
     KPI/risk/action card break mid-card looks broken. Long prose-only
     cards (.card-prose: the executive-summary intro, no-action,
     appendix) are deliberately left out: forcing a multi-paragraph
     block to stay whole is what strands the next section alone on a
     mostly-blank page whenever that block doesn't fit in the
     remaining space. */
  .kpi-card, .fact-card, .chart-card, .risk-card, .root-cause-card,
  .action-card, .dept-card, .conclusion-card, table, figure {
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .summary-row { break-inside: avoid; page-break-inside: avoid; }
  .phase-head { break-after: avoid; page-break-after: avoid; }
}
"""
