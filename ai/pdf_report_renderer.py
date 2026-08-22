"""Local HTML-to-PDF rendering for the executive report.

This module owns exactly one job: take the already-rendered
`executive_report.html` file on disk and print it to
`executive_report.pdf` using headless Chromium via Playwright. It
performs no analytics and no re-styling of its own -- Chromium simply
loads the same self-contained HTML file that a human would open in a
browser, so the PDF is a faithful print of what
`ai/html_report_renderer.py` already produced rather than a second,
independently-maintained document.

Unlike the report's previous fixed-width desktop design, the HTML
document itself is now document-first (a single portrait column, print
typography, an A4-shaped content width) -- see the design-system note
at the bottom of ai/html_report_renderer.py. That means this module no
longer needs to fight the page geometry to make a desktop layout fit;
it just prints a standard A4 portrait page with real margins, the way
any consulting/enterprise PDF is produced.

Page geometry -- paper size, margin, and the repeating "Confidential /
Page X of Y" footer -- is owned entirely by the document's own CSS
`@page` rules (see the `@media print` block in
ai/html_report_renderer.py), not by Playwright/Chromium's separate
header-footer-template mechanism:
  - `format="A4"`, `landscape=False` set the paper size as a fallback;
    `prefer_css_page_size=True` gives the document's own `@page { size:
    A4 portrait }` rule priority if the two ever disagree.
  - `display_header_footer=False` -- the footer text is printed by a
    plain CSS `@page { @bottom-left {...} @bottom-right {...} }` margin
    box instead. This repo used Chromium's JS-templated header/footer
    for that footer in an earlier version, but that mechanism reserves
    its own fixed margin band on *every* page (including the first),
    layered independently of the page's own `@page` CSS -- which is
    exactly why the cover page's near-black background couldn't be made
    to reach the true page edge no matter how its own margin/height was
    adjusted: Chromium was still carving out a footer band underneath
    it. Moving the footer into the document's own `@page` rule puts one
    thing in charge of page geometry, and lets the cover suppress that
    band entirely via `@page :first { margin: 0; ... }` for a genuine
    full-bleed first page.

Loading the file via a `file://` URL (rather than serving it over
HTTP) means no server process is needed, which keeps this working
identically on a local macOS machine and in GitHub Actions.
"""

from __future__ import annotations

from pathlib import Path

PDF_FORMAT = "A4"

# Desktop-width viewport so any live (pre-print) page state -- font
# loading, general layout -- resolves the same way a normal browser
# window would. Print layout itself is governed by the document's own
# @page CSS (see ai/html_report_renderer.py), not by this viewport.
PDF_VIEWPORT = {"width": 900, "height": 1400}


class PDFRenderError(Exception):
    """The HTML report could not be printed to PDF.

    Raised for a missing input file, a missing/not-yet-installed
    Playwright Chromium browser, or any failure Chromium itself
    reports while loading or printing the page -- so a caller gets a
    message that names the problem instead of a raw Playwright
    traceback.
    """


def render_pdf(html_path: str | Path, pdf_path: str | Path) -> Path:
    """Render a local `executive_report.html` file to a PDF at `pdf_path`.

    Launches headless Chromium, waits for layout and web fonts to
    settle, then prints an A4 portrait PDF with real print margins and
    a repeating "Confidential · Page X of Y" footer.

    Args:
        html_path: Path to the already-generated executive_report.html.
        pdf_path: Path to write executive_report.pdf to. Parent
            directories are created if needed.

    Returns:
        The resolved `pdf_path`.

    Raises:
        PDFRenderError: If `html_path` doesn't exist, Playwright's
            Chromium browser isn't installed, or Chromium fails to
            load or print the page for any other reason.
    """
    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path).resolve()

    if not html_path.is_file():
        raise PDFRenderError(f"HTML report not found at {html_path}; cannot render PDF.")

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PDFRenderError(
            "The 'playwright' package is not installed. Run "
            "`pip install -r requirements.txt` then `playwright install --with-deps chromium`."
        ) from exc

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport=PDF_VIEWPORT)
                page.goto(html_path.as_uri(), wait_until="load")
                page.evaluate("document.fonts.ready")

                page.emulate_media(media="print")
                page.pdf(
                    path=str(pdf_path),
                    format=PDF_FORMAT,
                    landscape=False,
                    print_background=True,
                    prefer_css_page_size=True,
                    display_header_footer=False,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        raise PDFRenderError(
            "Chromium failed to render the PDF. If this is a fresh environment, run "
            f"`playwright install --with-deps chromium` first. Underlying error: {exc}"
        ) from exc

    return pdf_path
