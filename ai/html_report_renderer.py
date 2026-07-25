"""Client-facing HTML rendering for the Real Estate Executive Intelligence Report.

This module owns exactly one job: turn the final report dict --
already fully computed and validated by ai/report_generator.py -- into
a single self-contained HTML document. It performs no analytics, no
scoring, and no narrative generation; every value on the page is read
directly from the report dict, the same source of truth the JSON and
Markdown outputs are built from. If a value isn't already in `report`,
it doesn't appear on the page.

The document is document-first (a single portrait content column,
print typography, an A4-shaped content width), a dark title-panel
cover followed by a light editorial body -- the same design system
built for this platform's original operations report, carried forward
unchanged (colors, spacing scale, typography scale, print pagination
rules) and extended with two new primitives real estate's richer,
more tabular data needs: `_data_table` (compact ranking/comparison
tables) and `_fmt_currency` (EGP 8.4M / EGP 1.26B style compact
currency formatting).

All dynamic text is passed through `html.escape` before being placed
in the document.

Layout mirrors the report's own 19-section structure: cover, executive
dashboard, business health overview, nine domain sections (commercial
performance, lead funnel, inventory & pricing, marketing, sales team &
broker, collections, cancellations, construction & handover, customer
experience), cross-functional root causes, financial exposure, 90-day
action plan, department accountability, forecast & outlook, executive
conclusion, appendix, closing footer. Only the report-wide narrative
layer (headline, summary, per-domain one-liners, root-cause/action
rationale, department briefings, forecast commentary) is AI-written;
every domain section's tables, rankings, and charts are rendered
directly from the report dict's own Python-computed data.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class HTMLRenderError(Exception):
    """The report dict was missing a field this renderer requires."""


# --------------------------------------------------------------------------
# Presentation vocabulary
# --------------------------------------------------------------------------

STATUS_CLASS = {"Healthy": "status-healthy", "Watch": "status-watch", "At Risk": "status-at-risk", "Critical": "status-critical"}
SEVERITY_CLASS = {"HIGH": "severity-high", "MEDIUM": "severity-medium", "LOW": "severity-low"}
DEFAULT_STATUS_CLASS = "status-unknown"
DEFAULT_SEVERITY_CLASS = "severity-low"

DOMAIN_ORDER = [
    "commercial_health", "sales_funnel_health", "inventory_health", "marketing_efficiency_health",
    "collections_health", "cancellations_health", "construction_delivery_health",
    "handover_readiness_health", "customer_experience_health",
]

ACTION_HORIZON_ORDER = ("Immediate (0-30 Days)", "Near-Term (30-60 Days)", "Strategic (60-120 Days)", "Longer-Term (120-365 Days)")
ACTION_HORIZON_CLASS = {
    "Immediate (0-30 Days)": "phase-immediate", "Near-Term (30-60 Days)": "phase-near-term",
    "Strategic (60-120 Days)": "phase-strategic", "Longer-Term (120-365 Days)": "phase-longer-term",
}


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------


def _esc(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value), quote=True)


def _label(key: str) -> str:
    return key.replace("_", " ").title()


def _fmt_score(score: float | None) -> str:
    return "N/A" if score is None else str(score)


def _percent(value: float | None, of: float = 100.0) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(100.0, float(value) / of * 100))


def _fmt_confidence(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0%}"


def _fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}%"


def _fmt_currency(value: float | None, currency: str = "EGP") -> str:
    """8_400_000 -> "EGP 8.4M"; 1_260_000_000 -> "EGP 1.26B". Never exposes excessive decimals."""
    if value is None:
        return "N/A"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}{currency} {value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{sign}{currency} {value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}{currency} {value / 1_000:.0f}K"
    return f"{sign}{currency} {value:,.0f}"


def _fmt_datetime(value: str | None) -> str:
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


def _exposure_line_item(financial_exposure: dict[str, Any], category: str) -> dict[str, Any] | None:
    for item in financial_exposure.get("line_items", []) or []:
        if item.get("category") == category:
            return item
    return None


def _project_cancellation_rate_pct(report: dict[str, Any], project_id: str | None) -> float | None:
    if not project_id:
        return None
    by_project = (
        report.get("domains", {}).get("cancellations", {}).get("kpis", {}).get("cuts", {}).get("by_project", [])
    )
    for row in by_project:
        if row.get("project_id") == project_id:
            return row.get("cancellation_rate_pct")
    return None


def _brand_lockup(platform: str | None) -> tuple[str, str]:
    words = (platform or "").split()
    if not words:
        return "", ""
    if len(words) == 1:
        return words[0].upper(), ""
    tagline_words = words[1:-1] if len(words) > 2 else words[1:]
    return words[0].upper(), " ".join(tagline_words)


def _paragraphs_html(text: str, css_class: str = "") -> str:
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
    items = "".join(
        f'<div class="meta-item"><span class="meta-k">{_esc(k)}</span><span class="meta-v">{_esc(v)}</span></div>'
        for k, v in pairs if v not in (None, "")
    )
    return f'<div class="meta-row">{items}</div>' if items else ""


def _info_row(label: str, value: Any) -> str:
    if value in (None, ""):
        return ""
    return f'<div class="info-row"><span class="info-label">{_esc(label)}</span><span class="info-value">{_esc(value)}</span></div>'


# --------------------------------------------------------------------------
# Chart / table primitives
# --------------------------------------------------------------------------


def _hbar_row(label: str, percent: float, display_value: str, bar_class: str = "", meta_html: str = "") -> str:
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


def _data_table(columns: list[tuple[str, str]], rows: list[dict[str, Any]], max_rows: int = 8) -> str:
    """A compact print-safe table. `columns` is [(field_key, header_label), ...]."""
    if not rows:
        return '<p class="empty-state">No data available for this period.</p>'
    head = "".join(f"<th>{_esc(label)}</th>" for _key, label in columns)
    body_rows = []
    for row in rows[:max_rows]:
        cells = "".join(f"<td>{_esc(row.get(key, ''))}</td>" for key, _label in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"""
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>""".strip()


# --------------------------------------------------------------------------
# Cover
# --------------------------------------------------------------------------


def _render_cover(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    currency = metadata.get("currency", "EGP")
    brand, tagline = _brand_lockup(metadata.get("platform"))
    monogram = _esc(brand[:1]) if brand else ""
    tagline_html = f'<p class="cover-tagline">{_esc(tagline)}</p>' if tagline else ""

    financial_exposure = report.get("financial_exposure", {}) or {}
    top_action = (report.get("recommended_actions") or [{}])[0] or {}
    top_project = report.get("highest_risk_project")

    cancellation_item = _exposure_line_item(financial_exposure, "Cancellation Exposure")
    receivables_item = _exposure_line_item(financial_exposure, "Receivables at Risk")
    project_cancellation_rate = _project_cancellation_rate_pct(report, top_project)

    hero_amount = _fmt_currency(cancellation_item.get("amount") if cancellation_item else None, currency)
    cancellation_label = _esc(cancellation_item.get("category")) if cancellation_item else "Cancellation Exposure"
    receivables_label = _esc(receivables_item.get("category")) if receivables_item else "Receivables at Risk"
    rate_label = f"{_esc(top_project)} Cancellation Rate" if top_project else "Project Cancellation Rate"

    metric_cards = [
        (hero_amount, cancellation_label),
        (_fmt_currency(receivables_item.get("amount") if receivables_item else None, currency), receivables_label),
        (_fmt_pct(project_cancellation_rate), rate_label),
        (_fmt_confidence(top_action.get("confidence")), "Diagnostic Confidence"),
    ]
    metrics_html = "\n".join(
        f'    <div class="cover-metric"><p class="cover-metric-label">{label}</p><p class="cover-metric-value">{value}</p></div>'
        for value, label in metric_cards
    )

    action_html = ""
    if top_action.get("recommended_action"):
        action_html = f"""
  <div class="cover-action">
    <p class="cover-action-label">Immediate Executive Action</p>
    <p class="cover-action-text">{_esc(top_action.get('recommended_action'))}</p>
    <div class="cover-action-meta">
      <div class="cover-action-item"><span class="cover-action-k">Owner</span><span class="cover-action-v">{_esc(top_action.get('owner'))}</span></div>
      <div class="cover-action-item"><span class="cover-action-k">Timeline</span><span class="cover-action-v">{_esc(top_action.get('horizon'))}</span></div>
    </div>
  </div>"""

    alert_label = "Highest-Priority Risk" + (f" &middot; {_esc(top_project)}" if top_project else "")

    return f"""
<section class="cover">
  <div class="cover-topline">
    <div class="cover-brand">
      <span class="cover-mark">{monogram}</span>
      <div>
        <p class="cover-word">{_esc(brand)}</p>
        {tagline_html}
      </div>
    </div>
    <div class="cover-meta">
      <p class="cover-report-type">{_esc(metadata.get('report_type'))}</p>
      <p class="cover-date">{_esc(_fmt_datetime(metadata.get('generated_at')))}</p>
    </div>
  </div>

  <div class="cover-alert">
    <p class="cover-alert-label">{alert_label}</p>
    <p class="cover-alert-title">{_esc(top_action.get('title'))}</p>
    <p class="cover-figure">{hero_amount}</p>
    <p class="cover-figure-label">At risk from cancellations</p>
    <p class="cover-diagnosis">{_esc(report.get('why_it_matters'))}</p>
  </div>

  <div class="cover-metrics">
{metrics_html}
  </div>
{action_html}
  <div class="cover-consequence">
    <span class="cover-consequence-label">No action</span>
    <span class="cover-consequence-text">{_esc(report.get('consequence_of_inaction'))}</span>
  </div>

  <p class="cover-footnote">Report {_esc(metadata.get('report_id'))}</p>
</section>
""".strip()


# --------------------------------------------------------------------------
# Executive Dashboard
# --------------------------------------------------------------------------


def _render_kpi_card(card: dict[str, Any], overall_business_health: dict[str, Any], currency: str) -> str:
    unit = card.get("unit")
    value = card.get("value")
    if unit == "currency":
        display_value = _fmt_currency(value, currency)
    elif unit == "percent":
        display_value = _fmt_pct(value)
    else:
        display_value = _fmt_score(value)

    status_ref = card.get("status_ref")
    block = overall_business_health.get(status_ref, {}) if status_ref else {}
    status = block.get("status") or "Unknown"
    status_class = STATUS_CLASS.get(status, DEFAULT_STATUS_CLASS)
    target = card.get("target")
    target_html = ""
    if target is not None:
        target_display = _fmt_currency(target, currency) if unit == "currency" else f"{target}%"
        target_html = f'<p class="kpi-target">Target: {_esc(target_display)}</p>'

    return f"""
    <div class="card kpi-card">
      <p class="kpi-label">{_esc(card.get('label'))}</p>
      <p class="kpi-score">{_esc(display_value)}</p>
      {target_html}
      <div class="kpi-meta-row">
        {_badge_html(status, status_class)}
        {_trend_html(block.get('trend', {}), compact=True)}
      </div>
    </div>""".strip()


def _render_dashboard(report: dict[str, Any]) -> str:
    currency = report["metadata"].get("currency", "EGP")
    overall_business_health = report.get("overall_business_health", {})
    kpi_cards = "\n".join(_render_kpi_card(card, overall_business_health, currency) for card in report.get("kpi_dashboard", []))

    return f"""
<section class="section">
  <h2 class="section-title">Executive Dashboard</h2>
  <div class="kpi-grid">
{kpi_cards}
  </div>
  <div class="fact-grid">
    <div class="card fact-card">
      <p class="fact-label">Highest Risk Project</p>
      <p class="fact-value">{_esc(report.get('highest_risk_project') or 'Not identified this period.')}</p>
    </div>
    <div class="card fact-card">
      <p class="fact-label">Strongest Performing Project</p>
      <p class="fact-value">{_esc(report.get('strongest_project') or 'Not identified this period.')}</p>
    </div>
    <div class="card fact-card">
      <p class="fact-label">Highest Priority Initiative</p>
      <p class="fact-value">{_esc(report.get('highest_priority_initiative') or 'None identified this period.')}</p>
    </div>
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Business Health Overview
# --------------------------------------------------------------------------


def _render_business_health_overview(report: dict[str, Any]) -> str:
    health = report.get("overall_business_health", {})
    rows = "\n".join(
        _hbar_row(
            _label(domain), _percent(health[domain].get("score")), _fmt_score(health[domain].get("score")),
            bar_class=STATUS_CLASS.get(health[domain].get("status"), DEFAULT_STATUS_CLASS),
            meta_html=f'<div class="hbar-meta">{_trend_html(health[domain].get("trend", {}), compact=True)}</div>',
        )
        for domain in DOMAIN_ORDER if domain in health
    )
    narrative_rows = "\n".join(
        f'    <div class="summary-row"><span class="summary-domain">{_esc(_label(domain))}</span>'
        f'<span class="summary-text">{_esc(health[domain].get("narrative"))}</span></div>'
        for domain in DOMAIN_ORDER if domain in health and health[domain].get("narrative")
    )
    overall = health.get("overall", {})
    return f"""
<section class="section">
  <h2 class="section-title">Executive Summary</h2>
  <div class="card card-prose">
    {_paragraphs_html(report.get('executive_summary', ''), css_class="prose lead")}
  </div>
  <div class="card chart-card chart-card-full">
    <p class="chart-title">Business Health by Domain &middot; Overall {_esc(_fmt_score(overall.get('score')))}/100 ({_esc(overall.get('status'))})</p>
    {rows}
  </div>
  <div class="summary-grid">
{narrative_rows}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Domain section builder (shared shape for the nine domain sections)
# --------------------------------------------------------------------------


def _domain_section(title: str, intro_stat: str, body_html: str) -> str:
    intro_html = f'<p class="section-intro">{_esc(intro_stat)}</p>' if intro_stat else ""
    return f"""
<section class="section">
  <h2 class="section-title">{_esc(title)}</h2>
  {intro_html}
  {body_html}
</section>
""".strip()


def _render_commercial_performance(report: dict[str, Any]) -> str:
    domain = report["domains"]["commercial_performance"]
    currency = report["metadata"].get("currency", "EGP")
    summary = domain.get("summary", {})
    by_project = domain.get("rankings", {}).get("by_project", [])

    intro = (
        f"Net contracted sales of {_fmt_currency(summary.get('net_contracted_sales_value'), currency)} across "
        f"{summary.get('units_sold_net')} units, at a {_fmt_pct(summary.get('gross_to_net_realization_pct'))} "
        "gross-to-net realization rate."
    )
    rows = [
        {"project_id": r["project_id"], "units_sold": r["units_sold"],
         "net_sales_value": _fmt_currency(r["net_sales_value"], currency),
         "avg_discount_pct": _fmt_pct(r["avg_discount_pct"]),
         "avg_price_per_sqm": _fmt_currency(r["avg_price_per_sqm"], currency)}
        for r in by_project
    ]
    table = _data_table(
        [("project_id", "Project"), ("units_sold", "Units"), ("net_sales_value", "Net Sales Value"),
         ("avg_discount_pct", "Avg Discount"), ("avg_price_per_sqm", "Avg EGP/sqm")],
        rows,
    )
    meta = _meta_row([
        ("Reservation to Contract", _fmt_pct(summary.get("reservation_to_contract_conversion_pct"))),
        ("Cancellation Rate", _fmt_pct(summary.get("contract_cancellation_rate_pct"))),
        ("Avg Discount", _fmt_pct(summary.get("average_discount_pct"))),
    ])
    body = f'<div class="card">{table}{meta}</div>'
    return _domain_section("Commercial Performance", intro, body)


def _render_lead_funnel(report: dict[str, Any]) -> str:
    domain = report["domains"]["lead_funnel"]
    summary = domain.get("summary", {})
    waterfall = domain.get("kpis", {}).get("funnel_waterfall", [])

    intro = (
        f"{summary.get('total_leads', 0):,} leads generated {summary.get('leads_converted_to_sale')} sales "
        f"({_fmt_pct(summary.get('lead_to_reservation_conversion_pct'))} conversion), with "
        f"{_fmt_pct(summary.get('response_sla_attainment_pct'))} of leads first-responded-to within SLA."
    )
    rows = "\n".join(
        _hbar_row(stage["stage"], _percent(stage["pct_of_total"]), f"{stage['count']:,} ({_fmt_pct(stage['step_conversion_pct'])} of prior stage)")
        for stage in waterfall
    )
    by_project = domain.get("rankings", {}).get("by_project", [])
    table_rows = [
        {"project_id": r["project_id"], "leads": r["leads"], "avg_response_minutes": f"{r['avg_response_minutes']:.0f} min" if r.get("avg_response_minutes") is not None else "N/A",
         "lead_to_contract_conversion_pct": _fmt_pct(r["lead_to_contract_conversion_pct"])}
        for r in by_project
    ]
    table = _data_table(
        [("project_id", "Project"), ("leads", "Leads"), ("avg_response_minutes", "Avg Response"), ("lead_to_contract_conversion_pct", "Lead to Contract")],
        table_rows,
    )
    body = f'<div class="card chart-card chart-card-full">{rows}</div><div class="card">{table}</div>'
    return _domain_section("Lead Funnel & Sales Capacity", intro, body)


def _render_inventory_pricing(report: dict[str, Any]) -> str:
    domain = report["domains"]["inventory_pricing"]
    currency = report["metadata"].get("currency", "EGP")
    summary = domain.get("summary", {})
    aging = domain.get("kpis", {}).get("aging", {}).get("aging_distribution", [])
    absorption = domain.get("kpis", {}).get("absorption", {}).get("by_project", [])

    intro = (
        f"{summary.get('available_units')} available units worth {_fmt_currency(summary.get('available_inventory_value'), currency)}; "
        f"{_fmt_pct(summary.get('stale_units_pct_of_available'))} of available inventory has been on the market 180+ days."
    )
    rows = "\n".join(_hbar_row(b["bucket"], _percent(b["percentage"]), f"{b['count']} units ({_fmt_pct(b['percentage'])})") for b in aging)
    table_rows = [
        {"project_id": r["project_id"], "absorption_pct": _fmt_pct(r["absorption_pct"]),
         "available_units": r["available_units"],
         "months_of_supply": f"{r['months_of_supply']:.1f} months" if r.get("months_of_supply") is not None else "N/A"}
        for r in absorption
    ]
    table = _data_table(
        [("project_id", "Project"), ("absorption_pct", "Absorption"), ("available_units", "Available Units"), ("months_of_supply", "Months of Supply")],
        table_rows,
    )
    body = f'<div class="card chart-card chart-card-full"><p class="chart-title">Available Inventory Aging</p>{rows}</div><div class="card">{table}</div>'
    return _domain_section("Inventory & Pricing", intro, body)


def _render_marketing_performance(report: dict[str, Any]) -> str:
    domain = report["domains"]["marketing_performance"]
    currency = report["metadata"].get("currency", "EGP")
    summary = domain.get("summary", {})
    by_channel = domain.get("rankings", {}).get("by_channel", [])

    intro = (
        f"{_fmt_currency(summary.get('total_spend'), currency)} in spend generated "
        f"{summary.get('total_leads', 0):,} leads and {summary.get('total_contracts')} contracts, a "
        f"{summary.get('overall_marketing_efficiency_ratio')}x attributed-revenue-to-spend ratio."
    )
    table_rows = [
        {"channel": r["channel"], "spend": _fmt_currency(r["spend"], currency),
         "cost_per_lead": _fmt_currency(r["cost_per_lead"], currency) if r.get("cost_per_lead") else "N/A",
         "lead_to_contract_pct": _fmt_pct(r["lead_to_contract_pct"]),
         "marketing_efficiency_ratio": f"{r['marketing_efficiency_ratio']}x" if r.get("marketing_efficiency_ratio") else "N/A"}
        for r in by_channel
    ]
    table = _data_table(
        [("channel", "Channel"), ("spend", "Spend"), ("cost_per_lead", "Cost / Lead"),
         ("lead_to_contract_pct", "Lead to Contract"), ("marketing_efficiency_ratio", "Efficiency Ratio")],
        table_rows, max_rows=10,
    )
    body = f'<div class="card">{table}</div>'
    return _domain_section("Marketing Performance", intro, body)


def _render_sales_team_broker(report: dict[str, Any]) -> str:
    domain = report["domains"]["sales_team_broker"]
    currency = report["metadata"].get("currency", "EGP")
    summary = domain.get("summary", {})
    teams = domain.get("rankings", {}).get("teams", [])
    brokers = domain.get("rankings", {}).get("brokers", [])

    intro = (
        f"Brokers source {_fmt_pct(summary.get('broker_share_of_reservations_pct'))} of reservations at "
        f"{_fmt_pct(summary.get('commission_as_pct_of_net_sales'))} of net sales value in commission."
    )
    team_rows = [
        {"team_id": r["team_id"], "project_id": r["project_id"], "reservations": r["reservations"],
         "reservation_to_contract_pct": _fmt_pct(r["reservation_to_contract_pct"]),
         "cancellation_rate_pct": _fmt_pct(r["cancellation_rate_pct"])}
        for r in teams
    ]
    team_table = _data_table(
        [("team_id", "Sales Team"), ("project_id", "Project"), ("reservations", "Reservations"),
         ("reservation_to_contract_pct", "Reservation to Contract"), ("cancellation_rate_pct", "Cancellation Rate")],
        team_rows,
    )
    broker_rows = [
        {"broker_name": r["broker_name"], "reservation_count": r["reservation_count"],
         "cancellation_rate_pct": _fmt_pct(r["cancellation_rate_pct"]),
         "average_discount_pct": _fmt_pct(r["average_discount_pct"]),
         "estimated_net_contribution": _fmt_currency(r["estimated_net_contribution"], currency)}
        for r in brokers
    ]
    broker_table = _data_table(
        [("broker_name", "Broker"), ("reservation_count", "Reservations"), ("cancellation_rate_pct", "Cancellation Rate"),
         ("average_discount_pct", "Avg Discount"), ("estimated_net_contribution", "Est. Net Contribution")],
        broker_rows,
    )
    body = (
        f'<div class="card"><p class="chart-title">Sales Teams</p>{team_table}</div>'
        f'<div class="card"><p class="chart-title">Brokers</p>{broker_table}</div>'
    )
    return _domain_section("Sales Team & Broker Performance", intro, body)


def _render_collections_risk(report: dict[str, Any]) -> str:
    domain = report["domains"]["collections_risk"]
    currency = report["metadata"].get("currency", "EGP")
    summary = domain.get("summary", {})
    aging = domain.get("kpis", {}).get("aging", {}).get("aging_buckets", [])
    forward = domain.get("kpis", {}).get("forward_obligations", {})

    intro = (
        f"{_fmt_currency(summary.get('total_overdue_amount'), currency)} overdue "
        f"({_fmt_pct(summary.get('overdue_amount_pct_of_due'))} of amount due), at a "
        f"{_fmt_pct(summary.get('collection_rate_pct'))} overall collection rate."
    )
    overdue_buckets = [b for b in aging if b["aging_bucket"] != "Current" and b["amount"]]
    max_amount = max((b["amount"] for b in overdue_buckets), default=1) or 1
    rows = "\n".join(
        _hbar_row(b["aging_bucket"], _percent(b["amount"], of=max_amount), _fmt_currency(b["amount"], currency))
        for b in overdue_buckets
    )
    meta = _meta_row([
        ("Next 30 Days Due", _fmt_currency(forward.get("next_30_days"), currency)),
        ("Next 60 Days Due", _fmt_currency(forward.get("next_60_days"), currency)),
        ("Next 90 Days Due", _fmt_currency(forward.get("next_90_days"), currency)),
    ])
    body = f'<div class="card chart-card chart-card-full"><p class="chart-title">Overdue Receivables Aging</p>{rows}</div><div class="card">{meta}</div>'
    return _domain_section("Collections & Receivables Risk", intro, body)


def _render_cancellations(report: dict[str, Any]) -> str:
    domain = report["domains"]["cancellations"]
    currency = report["metadata"].get("currency", "EGP")
    summary = domain.get("summary", {})
    reasons = domain.get("kpis", {}).get("reasons", [])[:5]
    timing = domain.get("kpis", {}).get("timing", {})

    intro = (
        f"{summary.get('total_cancellations')} cancellations ({_fmt_pct(summary.get('contract_cancellation_rate_pct'))} "
        f"of contracts) worth {_fmt_currency(summary.get('cancelled_gross_value'), currency)}, peaking "
        f"{timing.get('peak_window', 'N/A')} after reservation."
    )
    reason_rows = [{"reason": r["reason"], "count": r["count"], "value": _fmt_currency(r["value"], currency)} for r in reasons]
    table = _data_table([("reason", "Cancellation Reason"), ("count", "Count"), ("value", "Value")], reason_rows)
    timing_rows = "\n".join(
        _hbar_row(w["window"], _percent(w["percentage"]), f"{w['count']} ({_fmt_pct(w['percentage'])})")
        for w in timing.get("distribution", [])
    )
    body = f'<div class="card">{table}</div><div class="card chart-card chart-card-full"><p class="chart-title">Cancellation Timing (Days Since Reservation)</p>{timing_rows}</div>'
    return _domain_section("Cancellations & Revenue Leakage", intro, body)


def _render_construction_handover(report: dict[str, Any]) -> str:
    domain = report["domains"]["construction_handover"]
    currency = report["metadata"].get("currency", "EGP")
    summary = domain.get("summary", {})
    by_building = domain.get("kpis", {}).get("delay_concentration", {}).get("by_building", [])[:8]
    exposure = domain.get("kpis", {}).get("delay_exposure", {})

    intro = (
        f"{summary.get('handovers_delayed')} of {summary.get('total_handovers_scheduled')} scheduled handovers "
        f"are delayed ({_fmt_pct(summary.get('handover_delay_rate_pct'))}), exposing "
        f"{_fmt_currency(exposure.get('contracted_value_exposed'), currency)} in contracted value."
    )
    rows = [
        {"project_id": r["project_id"], "building": r["building"],
         "avg_variance_days": f"{r['avg_variance_days']:.0f} days" if r.get("avg_variance_days") is not None else "N/A",
         "high_severity_issues": r["high_severity_issues"]}
        for r in by_building
    ]
    table = _data_table(
        [("project_id", "Project"), ("building", "Building"), ("avg_variance_days", "Avg Schedule Variance"), ("high_severity_issues", "High-Severity Issues")],
        rows,
    )
    meta = _meta_row([
        ("Units Exposed", exposure.get("units_exposed_to_delay")),
        ("Customers Exposed", exposure.get("customers_exposed")),
        ("Already Collected on Delayed Units", _fmt_currency(exposure.get("amount_already_collected_on_delayed_units"), currency)),
    ])
    body = f'<div class="card">{table}{meta}</div>'
    return _domain_section("Construction & Handover Risk", intro, body)


def _render_customer_experience(report: dict[str, Any]) -> str:
    domain = report["domains"]["customer_experience"]
    summary = domain.get("summary", {})
    by_category = domain.get("kpis", {}).get("by_category", [])[:6]

    intro = (
        f"{summary.get('total_cases')} cases this period, {_fmt_pct(summary.get('negative_sentiment_pct'))} negative "
        f"sentiment, {_fmt_pct(summary.get('resolution_sla_attainment_pct'))} SLA attainment."
    )
    rows = [
        {"category": c["category"], "cases": c["cases"], "share_of_total_pct": _fmt_pct(c["share_of_total_pct"]),
         "negative_sentiment_pct": _fmt_pct(c["negative_sentiment_pct"])}
        for c in by_category
    ]
    table = _data_table(
        [("category", "Case Category"), ("cases", "Cases"), ("share_of_total_pct", "Share"), ("negative_sentiment_pct", "Negative Sentiment")],
        rows,
    )
    body = f'<div class="card">{table}</div>'
    return _domain_section("Customer Experience", intro, body)


# --------------------------------------------------------------------------
# Cross-Functional Root Causes
# --------------------------------------------------------------------------


def _render_root_causes(root_causes: list[dict[str, Any]]) -> str:
    if not root_causes:
        body = '<p class="empty-state">No cross-domain root cause met the platform&rsquo;s evidence threshold this period.</p>'
    else:
        cards = []
        for index, rc in enumerate(root_causes, start=1):
            evidence_items = (rc.get("statistical_evidence") or []) + (rc.get("operational_evidence") or [])
            evidence_html = (
                "<ul class=\"evidence-list\">" + "".join(f"<li>{_esc(item)}</li>" for item in evidence_items) + "</ul>"
                if evidence_items else ""
            )
            note_html = f'<p class="root-cause-note">{_esc(rc.get("executive_note"))}</p>' if rc.get("executive_note") else ""
            meta = _meta_row([("Owner", rc.get("owner")), ("Horizon", rc.get("horizon")), ("Confidence", _fmt_confidence(rc.get("confidence")))])
            cards.append(f"""
    <div class="card root-cause-card">
      <div class="root-cause-header">
        <div class="root-cause-heading">
          {_badge_html(f"P{index}", "badge-priority")}
          <span class="risk-title">{_esc(rc.get('title'))}</span>
        </div>
      </div>
      <div class="hbar-track confidence-track"><div class="hbar-fill hbar-confidence" style="width:{_percent((rc.get('confidence') or 0) * 100)}%"></div></div>
      {note_html}
      {evidence_html}
      {meta}
    </div>""".strip())
        body = "\n".join(cards)
    return f"""
<section class="section">
  <h2 class="section-title">Cross-Functional Root Causes</h2>
  {body}
</section>
""".strip()


# --------------------------------------------------------------------------
# Financial Exposure
# --------------------------------------------------------------------------


def _render_financial_exposure(financial_exposure: dict[str, Any], currency: str) -> str:
    line_items = financial_exposure.get("line_items", [])
    rows = "".join(
        f'<li><span class="impact-area">{_esc(item["category"])}</span>'
        f'<span class="impact-text">{_esc(_fmt_currency(item["amount"], currency))} &mdash; {_esc(item["description"])}</span></li>'
        for item in line_items
    )
    total = financial_exposure.get("total_material_risk_exposure")
    total_html = (
        f'<p class="conclusion-outlook"><strong>Total Material Risk Exposure:</strong> {_esc(_fmt_currency(total, currency))}</p>'
        if total is not None else ""
    )
    return f"""
<section class="section">
  <h2 class="section-title">Financial Exposure</h2>
  <div class="card">
    <ul class="impact-list">{rows}</ul>
    {total_html}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# 90-Day Action Plan
# --------------------------------------------------------------------------


def _render_action_card(action: dict[str, Any], currency: str) -> str:
    rationale_html = f'<p class="action-rationale">{_esc(action.get("executive_rationale"))}</p>' if action.get("executive_rationale") else ""
    exposure_html = (
        _info_row("Financial Exposure", _fmt_currency(action.get("financial_exposure"), currency))
        if action.get("financial_exposure") is not None else ""
    )
    meta_html = _meta_row([
        ("Owner", action.get("owner")), ("Difficulty", action.get("difficulty")),
        ("Timeline", action.get("estimated_timeframe")), ("Confidence", _fmt_confidence(action.get("confidence"))),
    ])
    return f"""
      <div class="card action-card">
        <div class="action-card-head">
          {_badge_html(f"P{action.get('priority')}", "badge-priority")}
          <p class="action-title">{_esc(action.get('title'))}</p>
        </div>
        {meta_html}
        <p class="action-reason">{_esc(action.get('business_issue'))}</p>
        {rationale_html}
        <div class="info-stack">
          {exposure_html}
          {_info_row("Recommended Action", action.get('recommended_action'))}
        </div>
      </div>""".strip()


def _render_action_plan(actions: list[dict[str, Any]], currency: str) -> str:
    if not actions:
        return """
<section class="section">
  <h2 class="section-title">90-Day Action Plan</h2>
  <p class="empty-state">No priority actions were identified this period.</p>
</section>
""".strip()

    phase_blocks = []
    for horizon_label in ACTION_HORIZON_ORDER:
        horizon_actions = [a for a in actions if a.get("horizon") == horizon_label]
        if not horizon_actions:
            continue
        cards = "\n".join(_render_action_card(a, currency) for a in horizon_actions)
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
# Department Accountability
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
  <h2 class="section-title">Department Accountability</h2>
  <div class="dept-grid">
{cards}
  </div>
</section>
""".strip()


# --------------------------------------------------------------------------
# Forecast & Outlook
# --------------------------------------------------------------------------


def _render_forecast(forecast: dict[str, Any], currency: str) -> str:
    if not forecast.get("available"):
        body = f'<p class="empty-state">{_esc(forecast.get("note", "Insufficient history for a forecast this period."))}</p>'
    else:
        rows = _meta_row([
            ("Expected Month-End Net Sales", _fmt_currency(forecast.get("expected_month_end_net_sales"), currency)),
            ("Projected Full-Year Net Sales", _fmt_currency(forecast.get("projected_full_year_net_sales"), currency)),
            ("Projected Target Attainment", _fmt_pct(forecast.get("projected_full_year_target_attainment_pct"))),
            ("Expected Next-Month Collections", _fmt_currency(forecast.get("expected_next_month_collections"), currency)),
            ("Expected Next-Month Cancellations", forecast.get("expected_next_month_cancellations")),
        ])
        narrative = _paragraphs_html(forecast.get("narrative", ""), css_class="prose")
        body = f'<div class="card">{narrative}{rows}</div>'
    return f"""
<section class="section">
  <h2 class="section-title">Forecast &amp; Outlook</h2>
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
# Executive Conclusion
# --------------------------------------------------------------------------


def _render_conclusion(report: dict[str, Any]) -> str:
    health = report.get("overall_business_health", {}) or {}
    statuses = [health[d].get("status") for d in DOMAIN_ORDER if d in health and health[d].get("status")]
    healthy = statuses.count("Healthy")
    watch = statuses.count("Watch")
    at_risk = statuses.count("At Risk")
    critical = statuses.count("Critical")
    total = len(statuses)

    if total:
        health_line = (
            f"Across {total} monitored business domains this period, {healthy} are Healthy, {watch} on Watch, "
            f"{at_risk} At Risk, and {critical} Critical."
        )
    else:
        health_line = "No business health domains were scored this period."

    findings_items = [rc.get("title") for rc in (report.get("root_causes") or [])[:3] if rc.get("title")]
    findings_html = "".join(f"<li>{_esc(item)}</li>" for item in findings_items) or '<li class="empty-state">No material findings were identified this period.</li>'

    risks = (report.get("top_business_risks") or [])[:3]
    risks_html = "".join(
        f'<li>{_esc(r.get("title"))} {_badge_html(r.get("severity") or "LOW", SEVERITY_CLASS.get(r.get("severity"), DEFAULT_SEVERITY_CLASS))}</li>'
        for r in risks
    ) or '<li class="empty-state">No significant business risks were identified this period.</li>'

    actions = sorted(report.get("recommended_actions") or [], key=lambda a: a.get("priority") or 99)[:3]
    actions_html = "".join(
        f'<li><strong>{_esc(a.get("title"))}</strong> &mdash; Owner: {_esc(a.get("owner"))}</li>' for a in actions
    ) or '<li class="empty-state">No priority actions were identified this period.</li>'

    if critical:
        outlook = (
            f"Immediate executive attention is warranted: {critical} domain{'s' if critical != 1 else ''} rated "
            "Critical this period. Executing the highest-impact recommendations above is the primary lever to "
            "stabilize the portfolio."
        )
    elif at_risk:
        outlook = (
            f"The portfolio is broadly functional, but {at_risk} domain{'s' if at_risk != 1 else ''} require close "
            "monitoring. Sustained execution of the recommended actions above should keep the business on a "
            "Healthy trajectory."
        )
    elif total:
        outlook = "All monitored business domains are Healthy or on Watch this period. Continued execution of the recommended actions will help sustain this position."
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


# --------------------------------------------------------------------------
# Appendix
# --------------------------------------------------------------------------


def _render_appendix(report: dict[str, Any]) -> str:
    appendix = report.get("appendix", {})
    limitations = appendix.get("data_limitations") or []
    limitations_html = "".join(f"<li>{_esc(item)}</li>" for item in limitations)
    dq = appendix.get("data_quality_summary", {})
    dq_checks = dq.get("checks_with_findings", [])
    dq_rows = [
        {"check": c["check"], "severity": c["severity"], "affected_pct": f"{c['affected_pct']}%"}
        for c in dq_checks
    ]
    dq_table = _data_table([("check", "Check"), ("severity", "Severity"), ("affected_pct", "Affected")], dq_rows) if dq_rows else '<p class="prose">No data quality findings this period.</p>'

    return f"""
<section class="section section-appendix">
  <h2 class="section-title">Appendix</h2>
  <div class="card card-prose appendix-card">
    <h3 class="appendix-heading">Methodology</h3>
    <p class="prose">{_esc(appendix.get('methodology_summary'))}</p>
    <h3 class="appendix-heading">Confidence Methodology</h3>
    <p class="prose">{_esc(appendix.get('confidence_explanation'))}</p>
    <h3 class="appendix-heading">Data Quality: {_esc(dq.get('score'))}/100 ({_esc(dq.get('status'))})</h3>
    {dq_table}
    <h3 class="appendix-heading">Data Limitations</h3>
    <ul class="limitations-list">{limitations_html}</ul>
    <p class="appendix-footnote">Generated by {_esc(appendix.get('generated_by'))}</p>
  </div>
</section>
""".strip()


def _render_footer(metadata: dict[str, Any]) -> str:
    return f"""
<footer class="report-footer">
  <p>{_esc(metadata.get('platform'))} for {_esc(metadata.get('company_name'))} &middot; AI model: {_esc(metadata.get('ai_model'))}</p>
  <p class="confidential">Confidential &mdash; prepared for internal executive distribution. All data is synthetic demonstration data.</p>
</footer>
""".strip()


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def render_html(report: dict[str, Any]) -> str:
    """Render the final report dict into a single self-contained HTML document.

    Raises:
        HTMLRenderError: If `report` is missing a field this renderer requires.
    """
    try:
        metadata = report["metadata"]
        currency = metadata.get("currency", "EGP")
        body = "\n".join([
            _render_cover(report),
            _render_dashboard(report),
            _render_business_health_overview(report),
            _render_commercial_performance(report),
            _render_lead_funnel(report),
            _render_inventory_pricing(report),
            _render_marketing_performance(report),
            _render_sales_team_broker(report),
            _render_collections_risk(report),
            _render_cancellations(report),
            _render_construction_handover(report),
            _render_customer_experience(report),
            _render_root_causes(report.get("root_causes", [])),
            _render_financial_exposure(report.get("financial_exposure", {}), currency),
            _render_action_plan(report.get("recommended_actions", []), currency),
            _render_departments(report.get("department_breakdown", {})),
            _render_forecast(report.get("forecast_outlook", {}), currency),
            _render_no_action(report),
            _render_conclusion(report),
            _render_appendix(report),
            _render_footer(metadata),
        ])
    except (KeyError, TypeError, AttributeError) as exc:
        raise HTMLRenderError(f"Report dict is missing a field the HTML renderer requires: {exc}") from exc

    title = escape(f"{metadata.get('report_type', 'Executive Report')} - {metadata.get('company_name', '')}")
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
# Same design system as this platform's original operations report: a
# dark title panel opens the report; every page after it is a light
# editorial document (dark ink on warm off-white). Typography in `pt`
# against the hierarchy: cover ~30pt, section titles ~16pt,
# subsection/item titles ~11.5-12pt, body ~10pt, labels ~7.5-9pt.
# Spacing on a 4/8/12/16/24/32 px scale. Page geometry (A4 portrait,
# margins, footer page numbers) is owned by ai/pdf_report_renderer.py.
# New in this version: `.data-table` for the many project/channel/
# agent/broker comparison tables real estate's richer domain model
# needs, and `.section-intro` for the one Python-templated headline
# line each domain section opens with.

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
  --blue: #2F5C8A;
  --blue-soft: rgba(47, 92, 138, 0.10);
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
.prose { max-width: 68ch; }

h1, h2, h3, h4 { margin: 0; }

/* ---- Cover ----
   A dedicated full A4-page executive alert, not a card floating on the
   page: the dark panel is a flex column whose direct children (topline,
   alert, metrics, action, consequence, footnote) are spaced with `gap`
   on screen and stretched to fill the page with `justify-content:
   space-between` in print (see the print override below), so the
   panel's own content -- not an arbitrary spacer -- fills the height. */
.cover {
  background: var(--cover-bg);
  color: var(--cover-ink);
  border-radius: 10px;
  padding: 44px 44px 40px;
  margin: 24px 0 32px;
  display: flex;
  flex-direction: column;
  gap: 28px;
}
.cover-topline { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
.cover-brand { display: flex; align-items: center; gap: 10px; }
.cover-mark {
  width: 26px; height: 26px; flex: 0 0 auto; border-radius: 4px;
  background: var(--cover-gold); color: #14120C; font-weight: 700; font-size: 8.5pt;
  display: inline-flex; align-items: center; justify-content: center;
}
.cover-word { margin: 0; font-size: 8.5pt; font-weight: 700; letter-spacing: 0.16em; color: var(--cover-gold); }
.cover-tagline { margin: 2px 0 0; font-size: 7.5pt; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--cover-ink-muted); }
.cover-meta { text-align: right; }
.cover-report-type { margin: 0; font-size: 9pt; font-weight: 600; color: var(--cover-ink); }
.cover-date { margin: 3px 0 0; font-size: 8pt; color: var(--cover-ink-muted); }

.cover-alert { border-top: 1px solid var(--cover-line); border-bottom: 1px solid var(--cover-line); padding: 26px 0; }
.cover-alert-label { margin: 0 0 10px; font-size: 8.5pt; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--cover-gold); }
.cover-alert-title { margin: 0 0 16px; font-size: 13pt; font-weight: 600; color: var(--cover-ink); max-width: 50ch; }
.cover-figure { margin: 0; font-size: 62pt; font-weight: 700; letter-spacing: -0.02em; line-height: 1; color: var(--cover-ink); font-variant-numeric: tabular-nums; }
.cover-figure-label { margin: 10px 0 18px; font-size: 11pt; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--cover-gold); }
.cover-diagnosis {
  margin: 0; font-size: 11pt; line-height: 1.5; color: var(--cover-ink-muted); max-width: 62ch;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}

.cover-metrics { display: flex; }
.cover-metric { flex: 1 1 0; padding: 0 22px; border-left: 1px solid var(--cover-line); }
.cover-metric:first-child { padding-left: 0; border-left: none; }
.cover-metric-label { margin: 0 0 8px; font-size: 7.5pt; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--cover-ink-muted); min-height: 2.4em; }
.cover-metric-value { margin: 0; font-size: 19pt; font-weight: 700; letter-spacing: -0.01em; color: var(--cover-ink); font-variant-numeric: tabular-nums; line-height: 1.05; }

.cover-action { border: 1px solid var(--cover-gold); border-radius: 6px; padding: 20px 24px; background: rgba(201, 168, 76, 0.07); }
.cover-action-label { margin: 0 0 10px; font-size: 8.5pt; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--cover-gold); }
.cover-action-text { margin: 0 0 14px; font-size: 10.5pt; line-height: 1.5; color: var(--cover-ink); max-width: 74ch; }
.cover-action-meta { display: flex; gap: 32px; flex-wrap: wrap; }
.cover-action-item { display: flex; flex-direction: column; gap: 3px; }
.cover-action-k { font-size: 7.5pt; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--cover-ink-muted); }
.cover-action-v { font-size: 9.5pt; color: var(--cover-ink); }

.cover-consequence { display: flex; gap: 12px; align-items: baseline; font-size: 9pt; }
.cover-consequence-label { flex: 0 0 auto; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--cover-ink-muted); }
.cover-consequence-text {
  flex: 1 1 auto; color: var(--cover-ink-muted);
  display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical; overflow: hidden;
}

.cover-footnote { margin: 0; font-size: 7pt; color: var(--cover-ink-muted); opacity: 0.7; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; text-align: right; }

/* ---- Section rhythm ---- */
.section { margin-bottom: 0; padding: 24px 16px 32px; }
.section-title {
  font-size: 16pt; font-weight: 600; letter-spacing: -0.005em; color: var(--ink);
  margin: 0 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--line);
}
.section-intro { font-size: 9.5pt; color: var(--ink-muted); margin: 0 0 12px; max-width: 72ch; }

/* ---- Cards ---- */
.card {
  background: var(--card-bg); color: var(--ink); border-radius: var(--radius);
  padding: 16px; margin-bottom: 8px; border: 1px solid var(--line);
}
.card:last-child { margin-bottom: 0; }
.card p { margin: 0 0 8px; }
.card p:last-child { margin-bottom: 0; }
.card-prose { padding: 24px; }

/* ---- KPI dashboard ---- */
.kpi-grid { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; }
.kpi-card { flex: 1 1 calc(50% - 8px); }
.kpi-label { font-size: 8pt; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 8px; }
.kpi-score { font-size: 22pt; font-weight: 600; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; color: var(--ink); margin-bottom: 4px; line-height: 1.1; }
.kpi-target { font-size: 8.5pt; color: var(--ink-faint); margin-bottom: 8px; }
.kpi-meta-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }

.hbar-track { height: 5px; background: var(--line-soft); border-radius: 2px; overflow: hidden; }
.hbar-fill { height: 100%; border-radius: 2px; background: var(--ink-muted); }
.hbar-fill.status-healthy { background: var(--green); }
.hbar-fill.status-watch { background: var(--blue); }
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
.chart-card { margin-bottom: 0; padding: 12px; }
.chart-card-full { margin-bottom: 8px; }
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

/* ---- Data tables ---- */
.table-wrap { overflow-x: auto; }
.data-table { width: 100%; border-collapse: collapse; font-size: 8.5pt; }
.data-table th {
  text-align: left; font-size: 7.5pt; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--ink-faint); padding: 6px 8px; border-bottom: 1px solid var(--line);
}
.data-table td { padding: 6px 8px; border-bottom: 1px solid var(--line-soft); color: var(--ink); white-space: nowrap; }
.data-table tr:last-child td { border-bottom: none; }
.data-table td:first-child, .data-table th:first-child { padding-left: 0; }

/* ---- Badges ---- */
.badge {
  display: inline-block; font-size: 7.5pt; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 3px 7px; border-radius: var(--radius); background: var(--line-soft); color: var(--ink-muted);
}
.badge.status-healthy { background: var(--green-soft); color: var(--green); }
.badge.status-watch { background: var(--blue-soft); color: var(--blue); }
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
.summary-domain { width: 170px; padding-right: 16px; color: var(--gold); font-weight: 600; font-size: 8pt; text-transform: uppercase; letter-spacing: 0.05em; }

/* ---- Risks / root causes ---- */
.risk-header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
.risk-rank {
  flex: 0 0 auto; width: 22px; height: 22px; border-radius: var(--radius); background: var(--ink);
  color: var(--card-bg); font-size: 9pt; font-weight: 700; display: inline-flex; align-items: center; justify-content: center;
}
.risk-title { font-size: 11.5pt; font-weight: 600; letter-spacing: -0.005em; flex: 1 1 auto; }
.risk-badges { display: flex; gap: 6px; flex: 0 0 auto; }

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
.phase-strategic .phase-dot { background: var(--blue); }
.phase-longer-term .phase-dot { background: var(--gold); }
.phase-title { font-size: 12pt; font-weight: 600; color: var(--ink); margin: 0; }
.phase-cards { display: flex; flex-direction: column; gap: 12px; }
.action-card { margin-bottom: 0; }
.action-card-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 4px; }
.action-title { font-size: 11pt; font-weight: 600; margin: 0; }
.action-reason { font-size: 9.5pt; }
.action-rationale { font-size: 9.5pt; color: var(--ink-muted); }

/* ---- Financial exposure / expected impact ---- */
.impact-list { list-style: none; margin: 0; padding: 0; }
.impact-list li { display: flex; gap: 12px; padding: 8px 0; border-top: 1px solid var(--line-soft); font-size: 9.5pt; }
.impact-list li:first-child { border-top: none; padding-top: 0; }
.impact-area { flex: 0 0 auto; width: 170px; font-weight: 600; color: var(--ink); }
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
   Responsive -- single-column, print-first already; only
   mobile screen widths need adjustment.
   ========================================================== */
@media screen and (max-width: 560px) {
  .report { padding: 20px 16px 32px; }
  .cover-title { font-size: 22pt; }
  .kpi-card, .dept-card { flex-basis: 100%; }
}

/* ---- Print ---- */
@page {
  size: A4 portrait;
  margin: 14mm 14mm 16mm 14mm;
}

@media print {
  .report { max-width: none; padding: 0; margin: 0; width: 100%; }

  h1, h2, h3, h4 { break-after: avoid; page-break-after: avoid; }
  p, li { orphans: 3; widows: 3; }

  /* @page content box is 297mm - 14mm top - 16mm bottom = 267mm tall;
     stop a hair short so rounding never tips the panel onto page two. */
  .cover {
    break-after: page; page-break-after: always; margin: 0;
    min-height: 262mm;
    justify-content: space-between;
  }

  .section { break-inside: auto; }

  /* A heading (section title, card/chart title, intro line) must never be
     the last thing on a page -- keep it glued to whatever comes right
     after it, so the browser carries the heading forward with its
     content instead of stranding it alone at the bottom. */
  .section-title, .section-intro, .chart-title, .dept-title, .phase-title, .appendix-heading {
    break-after: avoid;
    page-break-after: avoid;
  }

  .card, .kpi-card, .fact-card, .chart-card, .risk-card, .root-cause-card,
  .action-card, .dept-card, .conclusion-card, table, figure {
    break-inside: avoid;
    page-break-inside: avoid;
  }
  /* Long free-flowing narrative cards are the exception: they're allowed
     to break across pages (orphans/widows above keep the wrap clean)
     rather than being dragged whole onto a fresh page. */
  .card-prose, .appendix-card {
    break-inside: auto;
    page-break-inside: auto;
  }
  .summary-row { break-inside: avoid; page-break-inside: avoid; }
  .phase-head { break-after: avoid; page-break-after: avoid; }
  .data-table tr { break-inside: avoid; page-break-inside: avoid; }
}
"""
