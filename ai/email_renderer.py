"""Renders the automated executive delivery email.

This module owns exactly one job, the same way ai/html_report_renderer.py
and ai/pdf_report_renderer.py own theirs: turn the already-computed
report dict into a client-facing artifact -- here, the HTML body and
subject line of the email that carries the PDF. It performs no
analytics of its own; every dynamic value it shows (overall health,
primary risk, exposure, confidence) is read directly from the same
report dict the PDF is built from, via the same formatting helpers
ai/html_report_renderer.py uses, so the two documents never disagree
about a number.

What this module deliberately does NOT do: send the email. Per this
platform's existing architecture (see README -- "n8n owns orchestration
only: scheduling, triggering POST /run-analysis, downloading the
generated PDF artifact, emailing it"), actual delivery (SMTP/provider
credentials, the send call itself) stays outside this repository. This
module's output -- outputs/executive_email.html plus the small
outputs/email_meta.json sidecar (subject, sender name/address, PDF
attachment label) -- is what n8n's email-send node should read from
instead of hardcoding a subject line or a person's inbox as the sender.

Email HTML is written table-based with inline styles (not the class-
based stylesheet ai/html_report_renderer.py uses for the PDF): Outlook's
Word rendering engine and many mobile mail clients don't reliably apply
external/`<style>`-block CSS or modern layout properties, so this is
the standard, defensive way to keep one document readable across Gmail,
Apple Mail, and Outlook alike. It also does not embed the custom
webfonts ai/_assets.py provides for the PDF -- web fonts are unreliable
in email -- and instead uses system font stacks that already resemble
the brand's chosen faces closely enough (a platform UI monospace stack
standing in for IBM Plex Mono, a system sans stack for IBM Plex Sans).
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

from ai._assets import LOGO_ASPECT_RATIO, LOGO_ON_LIGHT
from ai.html_report_renderer import (
    _exposure_line_item,
    _fmt_confidence,
    _fmt_currency,
    _fmt_pct,
    _strip_project_prefix,
)
from analysis._config import CONFIG

EMAIL_SENDER_NAME = CONFIG.get("email", {}).get("sender_name", "Solven Intelligence")
EMAIL_SENDER_ADDRESS = CONFIG.get("email", {}).get("sender_email", "hello@solvenhq.com")

_INK = "#0B0B0A"
_INK_MUTED = "#5B584E"
_INK_FAINT = "#8A8676"
_CREAM = "#F4F1EA"
_WHITE = "#FFFFFF"
_LINE = "#E3DFD1"
_SIGNAL = "#DF5316"

_FONT_SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
_FONT_MONO = "'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', monospace"


def _esc(value: Any) -> str:
    return escape(str(value), quote=True) if value is not None else ""


def _fmt_subject_date(generated_at: str | None) -> str:
    """'22 Aug 2026' -- a compact, human date for the subject line."""
    if not generated_at:
        return ""
    try:
        parsed = datetime.fromisoformat(str(generated_at))
    except ValueError:
        return ""
    return f"{parsed.day} {parsed.strftime('%b %Y')}"


def _fmt_meta_date(generated_at: str | None) -> str:
    """'22 AUG 2026' -- the same date, styled for the header's mono metadata."""
    return _fmt_subject_date(generated_at).upper()


def build_subject(report: dict[str, Any]) -> str:
    """'Executive Intelligence Report — 22 Aug 2026'. No emoji, no marketing
    language, no urgency framing -- a professional, dynamically-dated
    subject line a system sends, not a newsletter."""
    date = _fmt_subject_date(report.get("metadata", {}).get("generated_at"))
    suffix = f" — {date}" if date else ""
    return f"Executive Intelligence Report{suffix}"


def _row(label: str, value: str, *, value_size: str = "20px", top_border: bool = True) -> str:
    border = f"border-top:1px solid {_LINE};" if top_border else ""
    return f"""
<tr><td style="padding:18px 0 0;{border}"></td></tr>
<tr><td style="padding:14px 0 18px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
    <tr><td style="font-family:{_FONT_MONO};font-size:11px;font-weight:600;letter-spacing:0.08em;
        text-transform:uppercase;color:{_INK_MUTED};padding-bottom:6px;">{label}</td></tr>
    <tr><td style="font-family:{_FONT_SANS};font-size:{value_size};font-weight:700;color:{_INK};
        line-height:1.3;">{value}</td></tr>
  </table>
</td></tr>""".strip()


def render_email(report: dict[str, Any]) -> dict[str, str]:
    """Build the executive delivery email.

    Returns:
        {"subject", "html", "sender_name", "sender_address",
         "attachment_label"} -- everything n8n's email-send node needs,
        so nothing about delivery identity is hardcoded there either.
    """
    metadata = report.get("metadata", {})
    currency = metadata.get("currency", "EGP")

    overall = report.get("overall_business_health", {}).get("overall", {})
    top_action = (report.get("recommended_actions") or [{}])[0] or {}
    top_project = report.get("highest_risk_project")
    financial_exposure = report.get("financial_exposure", {}) or {}
    cancellation_item = _exposure_line_item(financial_exposure, "Cancellation Exposure")

    overall_score = overall.get("score")
    overall_status = overall.get("status") or "N/A"
    primary_risk_title = _strip_project_prefix(top_action.get("title"), top_project)
    primary_risk_line = f"{_esc(top_project)} {_esc(primary_risk_title)}" if top_project else _esc(top_action.get("title") or "Not identified this period.")
    exposure = _fmt_currency(cancellation_item.get("amount") if cancellation_item else None, currency)
    confidence = _fmt_confidence(top_action.get("confidence"))

    subject = build_subject(report)
    meta_date = _fmt_meta_date(metadata.get("generated_at"))
    logo_h = 22
    logo_w = round(logo_h * LOGO_ASPECT_RATIO)

    snapshot_rows = "".join([
        _row("Overall Health", f"{_esc(f'{overall_score:.1f}' if overall_score is not None else 'N/A')} / 100 &nbsp;&middot;&nbsp; {_esc(overall_status).upper()}", top_border=False),
        _row("Primary Risk", primary_risk_line),
        _row("Exposure", _esc(exposure)),
        _row("Confidence", _esc(confidence)),
    ])

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="color-scheme" content="light">
<title>{_esc(subject)}</title>
<!--[if mso]>
<style type="text/css">
  table {{ border-collapse: collapse; }}
  body, table, td {{ font-family: Arial, sans-serif !important; }}
</style>
<![endif]-->
<style>
  @media screen and (max-width: 480px) {{
    .solven-container {{ width: 100% !important; }}
    .solven-pad {{ padding-left: 20px !important; padding-right: 20px !important; }}
  }}
</style>
</head>
<body style="margin:0; padding:0; background-color:#EAE7DC;">
  <div style="display:none; max-height:0; overflow:hidden; opacity:0; mso-hide:all;">
    Your latest operational analysis is complete. Overall health {_esc(f'{overall_score:.1f}' if overall_score is not None else 'N/A')}/100 &middot; {_esc(overall_status)}.
  </div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#EAE7DC;">
    <tr>
      <td align="center" style="padding:32px 16px;">
        <table role="presentation" class="solven-container" width="600" cellpadding="0" cellspacing="0" border="0" style="width:600px; max-width:600px; background-color:{_CREAM};">

          <!-- Header -->
          <tr>
            <td class="solven-pad" style="padding:28px 32px 20px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td valign="middle" style="width:60%;">
                    <img src="{LOGO_ON_LIGHT}" width="{logo_w}" height="{logo_h}" alt="Solven" style="display:block; height:{logo_h}px; width:{logo_w}px; border:0;">
                  </td>
                  <td valign="middle" align="right" style="width:40%; font-family:{_FONT_MONO}; font-size:9.5px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:{_INK_MUTED}; line-height:1.6;">
                    OPERATIONS INTELLIGENCE<br>REPORT / {_esc(meta_date)}
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          <tr><td style="padding:0 32px;"><div style="border-top:1px solid {_INK};"></div></td></tr>

          <!-- Body -->
          <tr>
            <td class="solven-pad" style="padding:30px 32px 6px;">
              <p style="margin:0 0 22px; font-family:{_FONT_SANS}; font-size:16px; line-height:1.5; color:{_INK};">
                Your latest operational analysis is complete.
              </p>
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
                {snapshot_rows}
              </table>
            </td>
          </tr>

          <tr>
            <td class="solven-pad" style="padding:6px 32px 30px;">
              <p style="margin:0; font-family:{_FONT_SANS}; font-size:13.5px; line-height:1.6; color:{_INK_MUTED};">
                The full executive report, including root-cause analysis, financial exposure, and recommended actions, is attached.
              </p>
            </td>
          </tr>

          <!-- Attachment indicator (not a link -- the PDF only exists as
               this email's attachment, so a clickable "view" CTA that
               points nowhere real would be dishonest; see
               ai/email_renderer.py's module docstring). -->
          <tr>
            <td class="solven-pad" style="padding:0 32px 32px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border:1px solid {_LINE};">
                <tr>
                  <td style="padding:14px 16px; font-family:{_FONT_MONO}; font-size:11px; color:{_INK_MUTED};">
                    <span style="color:{_SIGNAL}; font-weight:700;">&#128206;</span>&nbsp;
                    Executive Intelligence Report &middot; PDF &middot; attached
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <tr><td style="padding:0 32px;"><div style="border-top:1px solid {_LINE};"></div></td></tr>

          <!-- Footer -->
          <tr>
            <td class="solven-pad" style="padding:22px 32px 30px;">
              <p style="margin:0 0 2px; font-family:{_FONT_SANS}; font-size:12.5px; font-weight:700; color:{_INK};">SOLVEN</p>
              <p style="margin:0 0 12px; font-family:{_FONT_MONO}; font-size:9.5px; letter-spacing:0.1em; text-transform:uppercase; color:{_INK_FAINT};">Operations Intelligence &middot; solvenhq.com</p>
              <p style="margin:0; font-family:{_FONT_SANS}; font-size:10.5px; color:{_INK_FAINT};">Generated automatically by Solven Operations Intelligence.</p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

    return {
        "subject": subject,
        "html": html,
        "sender_name": EMAIL_SENDER_NAME,
        "sender_address": EMAIL_SENDER_ADDRESS,
        "attachment_label": "Executive Intelligence Report / PDF",
    }
