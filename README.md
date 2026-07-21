# Solven Operations Intelligence Platform

An enterprise-style analytics platform that turns raw operations data
(shipments, customer complaints, and warehouse inventory) into
AI-generated executive reports. Built as a portfolio project for
**Solven**, an AI automation agency, to demonstrate a production-shaped
approach to operational analytics and LLM-powered reporting.

## Project Overview

Businesses running logistics and fulfillment operations generate large
volumes of operational data — shipments, customer complaints,
inventory levels — but rarely have the time to synthesize that data
into a clear, executive-ready narrative. The Solven Operations
Intelligence Platform addresses this by combining:

1. **Data ingestion** of operational CSV exports (shipments,
   complaints, inventory).
2. **Analytical processing** that surfaces performance trends, risk
   signals, and anomalies (e.g. a warehouse becoming overloaded).
3. **AI-powered reporting** that turns those structured findings into
   a natural-language executive summary.

This repository currently implements the data layer and project
scaffolding. Analytical logic and AI reasoning are being built out
incrementally (see [Future Roadmap](#future-roadmap)).

## Features

- **Synthetic data generator** (`generate_data.py`) that produces
  realistic, internally-consistent operational data, including a
  deliberately embedded operational problem (a warehouse overload
  event) so downstream analytics have something real to detect.
- **Modular analysis layer** (`analysis/`) with one module per
  business domain — deliveries, complaints, inventory — each
  documented with the exact metrics it will compute.
- **AI reporting layer** (`ai/`) structured around a clear interface
  (`OperationsReportGenerator`) for turning analysis output into an
  executive-facing narrative via an LLM.
- **Single entry point** (`app.py`) that wires data loading, analysis,
  and report generation into one pipeline via `run_pipeline()`.
- **HTTP trigger** (`api.py`) exposing that same pipeline as a FastAPI
  service (`POST /run-analysis`, `GET /health`) for automation tools
  like n8n to call — no analytics logic is duplicated there.

## Architecture

```
                 ┌──────────────┐
                 │  data/*.csv  │   Raw operational exports
                 └──────┬───────┘
                        │  pandas.read_csv
                        ▼
                 ┌──────────────┐
                 │  analysis/   │   Domain-specific metrics & anomaly
                 │  delivery.py │   detection (deliveries, complaints,
                 │  complaints. │   inventory)
                 │  inventory.py│
                 └──────┬───────┘
                        │  structured insights (dict)
                        ▼
                 ┌──────────────┐
                 │     ai/      │   Prompt construction + LLM call to
                 │  report_gen. │   produce an executive-ready report
                 └──────┬───────┘
                        │
                        ▼
                 Executive Report
```

Each layer depends only on the layer below it, so the analysis and AI
layers can be developed, tested, and swapped independently of one
another. `app.py` is the only module aware of the full pipeline.

## Folder Structure

```
solven-operations-intelligence/
│
├── data/
│   ├── shipments.csv        # ~5,000 synthetic shipment records
│   ├── complaints.csv       # ~500 synthetic customer complaints
│   └── inventory.csv        # ~300 synthetic warehouse inventory records
│
├── analysis/
│   ├── __init__.py
│   ├── delivery.py          # Delivery performance analysis (placeholder)
│   ├── complaints.py        # Complaint trend analysis (placeholder)
│   └── inventory.py         # Inventory health analysis (placeholder)
│
├── ai/
│   ├── __init__.py
│   ├── report_generator.py      # Orchestration: analytics -> AI narrative -> JSON/Markdown report
│   └── html_report_renderer.py  # Renders the report dict to a self-contained HTML file
│
├── outputs/                     # Generated on each run (see "Generated Reports" below)
│   ├── executive_report.json
│   ├── executive_report.md
│   └── executive_report.html
│
├── generate_data.py         # Synthetic data generator
├── app.py                   # Application entry point + run_pipeline() orchestration function
├── api.py                   # FastAPI layer exposing the pipeline over HTTP (for n8n)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

**Requirements:** Python 3.12+

```bash
# 1. Clone the repository and move into it
cd solven-operations-intelligence

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables (needed once AI reporting is implemented)
cp .env.example .env
# then edit .env and add your Groq API key

# 5. Generate the synthetic datasets
python generate_data.py

# 6. Run the application
python app.py
```

## Generated Reports

Each run of `app.py` writes the same executive report in three formats
to `outputs/`, all built from the same underlying report dict (the
Markdown and HTML renderers are pure, deterministic functions over
that dict — no analytics or AI reasoning happens during rendering):

- **`outputs/executive_report.json`** — the structured report: every
  score, ranking, root cause, confidence value, and narrative field.
  The source of truth for the other two formats and the natural format
  for programmatic/API consumption.
- **`outputs/executive_report.md`** — a plain-text Markdown rendering,
  suited to being pasted into Slack, GitHub, or a wiki page.
- **`outputs/executive_report.html`** — a premium, self-contained
  client-facing report (Solven black-and-gold branding, KPI dashboard,
  risk/root-cause cards, a 90-day action plan). No external fonts,
  scripts, or CDN dependencies, so it opens correctly from disk with no
  internet connection, and includes print styling for a clean PDF/paper
  copy straight from the browser's print dialog.

Open the HTML report locally on macOS with:

```bash
open outputs/executive_report.html
```

## API (n8n integration)

`api.py` is a thin FastAPI wrapper around `app.run_pipeline()` — the
same orchestration function `python app.py` calls. It adds no
analytics of its own; it only runs the existing pipeline over HTTP and
returns its result, so n8n (or any other scheduler) can trigger a run
without needing a Python environment of its own. n8n is expected to
own scheduling, triggering this endpoint, and whatever happens after a
report exists (storage, email, notifications) — not analysis.

Start the API locally:

```bash
uvicorn api:app --reload --port 8000
```

Endpoints:

- **`GET /health`** — liveness check.

  ```bash
  curl http://localhost:8000/health
  ```

- **`POST /run-analysis`** — runs the full pipeline synchronously (this
  call blocks until the run finishes, including the AI report-generation
  step, and returns only once `outputs/executive_report.{json,md,html}`
  have been written) and returns the same report identifiers and output
  paths available from `outputs/executive_report.json`.

  ```bash
  curl -X POST http://localhost:8000/run-analysis
  ```

  A successful call returns `200` with a `report_id`, `generated_at`,
  the three `outputs` paths, and a `summary` (operations health score,
  highest-risk location, highest-priority initiative). A failed run
  (e.g. a missing `GROQ_API_KEY`, or the AI provider rejecting the
  request) returns `500` with a `message` describing what failed — the
  API never returns a raw stack trace or environment details, though a
  successful response does include the absolute server-local paths to
  the generated report files (the same paths already stored in
  `executive_report.json`'s `metadata`). Only one analysis run is
  allowed at a time; a call made while another is already running gets
  `409` immediately rather than being queued.

  This endpoint has no authentication yet and is not intended to be
  deployed publicly as-is.

## Future Roadmap

- [ ] Implement `analyze_deliveries()` — delivery time trends, delay
      rates, and warehouse overload detection.
- [ ] Implement `analyze_complaints()` — complaint volume, category
      breakdown, sentiment trends, and correlation with delivery
      performance.
- [ ] Implement `analyze_inventory()` — stockout risk, stock-to-sales
      ratios, and warehouse-level inventory health scoring.
- [ ] Implement `OperationsReportGenerator` — prompt construction and
      Groq integration to produce narrative executive reports.
- [ ] Add automated tests for the analysis layer.
- [ ] Add a simple dashboard/UI for exploring generated reports.

## License

This is a portfolio project and is not licensed for production use.
