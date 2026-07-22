"""Local HTML-to-PDF rendering for the executive report.

This module owns exactly one job: take the already-rendered
`executive_report.html` file on disk and print it to
`executive_report.pdf` using headless Chromium via Playwright. It
performs no analytics and no re-styling of its own -- Chromium simply
loads the same self-contained HTML file that a human would open in a
browser, so the PDF is a faithful print of what
`ai/html_report_renderer.py` already produced rather than a second,
independently-maintained document.

The report's CSS is a fixed-width desktop design (`.report` caps at
1120px) with a four/two/three/two-column grid system. An A4 *portrait*
page is only ~640px wide once margins are subtracted, which is
narrower than any of that design's grid tracks were ever meant to
render at -- forcing the grids to reflow into that width crushes every
card. This module instead prints to A4 **landscape** (the widest
standard page size this design can use without a custom paper size),
whose ~26.5cm-wide printable area lets `.report`'s own `max-width:
1120px` reflow only mildly (to ~1002px, ~89% of its full design width)
-- a normal, gentle responsive narrowing rather than a crush, and
close enough to the desktop design that grid columns, card
proportions, and text wrapping all stay visually close to the browser
original.

Page *size* is intentionally owned here in Python, not in the HTML's
CSS: an explicit `@page { size }` rule would take priority over the
`width`/`height` passed to `page.pdf()` below, so the report's
stylesheet deliberately leaves page size unset (see the `@media print`
block in ai/html_report_renderer.py) and only owns pagination rules
(break-inside, etc.) instead.

Page *margin* is deliberately passed to `page.pdf()` as zero. Chromium
never paints a page's background into its own PDF margin -- that
gutter is physically outside the printable content box, so a nonzero
`page.pdf(margin=...)` shows up as unpainted white paper framing the
dark report regardless of `print_background` or any CSS background.
The report's 1.6cm of visual breathing room is instead spent as
`.page { padding: 1.6cm }` in the print stylesheet -- still inside
body's own painted box, so the dark background reaches every physical
edge of the page, and `.report`'s own `max-width` naturally reflows to
fit next to that padding with no extra work here (an explicit
`page.pdf(scale=...)` on top of that padding was tried and rejected --
see git history -- because it double-counted the margin, shrinking
`.report` too far and leaving the padding visibly larger than 1.6cm).

Loading the file via a `file://` URL (rather than serving it over
HTTP) means no server process is needed, which keeps this working
identically on a local macOS machine and in GitHub Actions.
"""

from __future__ import annotations

from pathlib import Path

# A4 landscape, not portrait: portrait's ~640px-wide printable area is
# narrower than this report's grid system was designed for, forcing
# every card grid to crush down. Landscape's printable width comes
# much closer to the design's native 1120px.
PAGE_WIDTH_CM = 29.7
PAGE_HEIGHT_CM = 21.0

# Zero PDF-level margin: the report's 1.6cm of visual breathing room
# is spent as `.page { padding: 1.6cm }` in the print stylesheet
# instead (see module docstring for why). Keep these two values in
# sync by hand -- the modules don't import from each other.
ZERO_PDF_MARGINS = {"top": "0cm", "bottom": "0cm", "left": "0cm", "right": "0cm"}

# Desktop-width viewport so any live (pre-print) page state -- font
# loading, general layout -- resolves the same way a normal browser
# window would. Print layout itself is governed by PAGE_WIDTH_CM/
# PAGE_HEIGHT_CM above, not by this viewport.
PDF_VIEWPORT = {"width": 1440, "height": 1200}


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
    settle, then prints an A4-landscape PDF with a zero PDF-level
    margin (see module docstring) so the dark report background
    reaches every edge of the page -- no white paper border, and no
    reflow beyond the mild, natural narrowing `.report`'s own
    `max-width` already does next to its in-document padding.

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
                    width=f"{PAGE_WIDTH_CM}cm",
                    height=f"{PAGE_HEIGHT_CM}cm",
                    margin=ZERO_PDF_MARGINS,
                    print_background=True,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        raise PDFRenderError(
            "Chromium failed to render the PDF. If this is a fresh environment, run "
            f"`playwright install --with-deps chromium` first. Underlying error: {exc}"
        ) from exc

    return pdf_path
