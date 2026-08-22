"""Loads the Solven brand assets (logo variants, embedded webfonts) from
`assets/` and exposes them as ready-to-inline `data:` URIs.

Kept separate from ai/html_report_renderer.py and ai/email_renderer.py
(which both need these) so neither renderer duplicates file I/O or path
logic, and so the base64 encoding happens once per process (module-level
constants, computed at import time) rather than once per render call.

Every renderer that uses these stays self-contained: no external CDN,
no network dependency, no font/logo file that has to ship alongside the
HTML output -- the asset bytes are embedded directly in the document,
consistent with this platform's existing "no external dependency"
design constraint for its generated reports.
"""

from __future__ import annotations

import base64
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_BRAND_DIR = _ASSETS_DIR / "brand"
_FONTS_DIR = _ASSETS_DIR / "fonts"


def _data_uri(path: Path, mime: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# --------------------------------------------------------------------------
# Logo variants -- pick the one with correct contrast for the background:
#   LOGO_ON_DARK  -- cream wordmark + orange mark, for the near-black cover
#   LOGO_ON_LIGHT -- near-black wordmark + orange mark, for cream/white
#                    backgrounds (email header, internal pages if ever used)
#   MARK_ONLY     -- the orange mark alone, no wordmark, for compact use
# All three are derived once from the supplied source logo by
# scripts/_prep_brand_assets.py and committed under assets/brand/ --
# never regenerated at request time.
# --------------------------------------------------------------------------

LOGO_ON_DARK = _data_uri(_BRAND_DIR / "solven-logo-on-dark.png", "image/png")
LOGO_ON_LIGHT = _data_uri(_BRAND_DIR / "solven-logo-on-light.png", "image/png")
MARK_ONLY = _data_uri(_BRAND_DIR / "solven-mark.png", "image/png")

# Intrinsic aspect ratio (both wordmark variants share it) -- used to size
# the <img> without ever stretching or distorting the logo.
LOGO_ASPECT_RATIO = 1566 / 264


# --------------------------------------------------------------------------
# Fonts -- embedded for the print/PDF report only (ai/html_report_renderer.py).
# The email (ai/email_renderer.py) deliberately does NOT embed these: web
# fonts are unreliable across email clients (Outlook in particular), so
# the email uses system font stacks instead. Embedding here instead of
# linking Google Fonts keeps the print HTML fully self-contained and
# rendering identically regardless of what fonts happen to be installed
# on the machine running Chromium (a local Mac vs. a CI/Linux runner) --
# see ai/html_report_renderer.py's design-system note and the platform's
# future-video-capture requirement for deterministic, reproducible output.
#
# IBM Plex Mono ships genuinely distinct static files per weight. IBM
# Plex Sans and Inter are variable fonts: Google serves one file for all
# requested weights, and multiple @font-face rules (one per declared
# font-weight, all pointing at the same file) is the standard technique
# for letting the browser resolve the correct weight from it -- so one
# data URI is reused across several @font-face declarations below.
# --------------------------------------------------------------------------

PLEX_MONO_400 = _data_uri(_FONTS_DIR / "ibm-plex-mono-400.woff2", "font/woff2")
PLEX_MONO_500 = _data_uri(_FONTS_DIR / "ibm-plex-mono-500.woff2", "font/woff2")
PLEX_MONO_600 = _data_uri(_FONTS_DIR / "ibm-plex-mono-600.woff2", "font/woff2")
PLEX_SANS_VARIABLE = _data_uri(_FONTS_DIR / "ibm-plex-sans-variable.woff2", "font/woff2")
INTER_VARIABLE = _data_uri(_FONTS_DIR / "inter-variable.woff2", "font/woff2")


FONT_FACES_CSS = f"""
@font-face {{ font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 400; src: url({PLEX_MONO_400}) format('woff2'); font-display: block; }}
@font-face {{ font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 500; src: url({PLEX_MONO_500}) format('woff2'); font-display: block; }}
@font-face {{ font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 600; src: url({PLEX_MONO_600}) format('woff2'); font-display: block; }}
@font-face {{ font-family: 'IBM Plex Sans'; font-style: normal; font-weight: 400; src: url({PLEX_SANS_VARIABLE}) format('woff2'); font-display: block; }}
@font-face {{ font-family: 'IBM Plex Sans'; font-style: normal; font-weight: 500; src: url({PLEX_SANS_VARIABLE}) format('woff2'); font-display: block; }}
@font-face {{ font-family: 'IBM Plex Sans'; font-style: normal; font-weight: 600; src: url({PLEX_SANS_VARIABLE}) format('woff2'); font-display: block; }}
@font-face {{ font-family: 'IBM Plex Sans'; font-style: normal; font-weight: 700; src: url({PLEX_SANS_VARIABLE}) format('woff2'); font-display: block; }}
@font-face {{ font-family: 'Inter'; font-style: normal; font-weight: 700; src: url({INTER_VARIABLE}) format('woff2'); font-display: block; }}
@font-face {{ font-family: 'Inter'; font-style: normal; font-weight: 800; src: url({INTER_VARIABLE}) format('woff2'); font-display: block; }}
@font-face {{ font-family: 'Inter'; font-style: normal; font-weight: 900; src: url({INTER_VARIABLE}) format('woff2'); font-display: block; }}
""".strip()
