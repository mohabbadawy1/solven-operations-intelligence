"""Local HTML-to-PDF rendering for the executive report.

This module owns exactly one job: take the already-rendered
`executive_report.html` file on disk and print it to
`executive_report.pdf` using headless Chromium via Playwright. It
performs no analytics and no re-styling of its own -- Chromium simply
loads the same self-contained HTML file (including its embedded
`@media print` rules) that a human would open in a browser, so the
PDF is a faithful print of what `ai/html_report_renderer.py` already
produced rather than a second, independently-maintained document.

Loading the file via a `file://` URL (rather than serving it over
HTTP) means no server process is needed, which keeps this working
identically on a local macOS machine and in GitHub Actions.
"""

from __future__ import annotations

from pathlib import Path


class PDFRenderError(Exception):
    """The HTML report could not be printed to PDF.

    Raised for a missing input file, a missing/not-yet-installed
    Playwright Chromium browser, or any failure Chromium itself
    reports while loading or printing the page -- so a caller gets a
    message that names the problem instead of a raw Playwright
    traceback.
    """


# A4 in portrait, with margins generous enough to match the 1.6cm
# `@page` margin the HTML report already declares for print in its
# embedded stylesheet (see the `@media print` block in
# ai/html_report_renderer.py) -- kept in sync there rather than
# imported, since that module has no dependency on this one.
PDF_MARGINS = {"top": "1.6cm", "bottom": "1.6cm", "left": "1.6cm", "right": "1.6cm"}


def render_pdf(html_path: str | Path, pdf_path: str | Path) -> Path:
    """Render a local `executive_report.html` file to a PDF at `pdf_path`.

    Launches headless Chromium via Playwright, opens `html_path`
    directly from disk, emulates print media (so the report's
    `@media print` rules apply), and prints an A4 PDF with backgrounds
    enabled -- preserving the report's black-and-gold styling instead
    of falling back to a plain white print.

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
                page = browser.new_page()
                page.goto(html_path.as_uri(), wait_until="load")
                page.emulate_media(media="print")
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin=PDF_MARGINS,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        raise PDFRenderError(
            "Chromium failed to render the PDF. If this is a fresh environment, run "
            f"`playwright install --with-deps chromium` first. Underlying error: {exc}"
        ) from exc

    return pdf_path
