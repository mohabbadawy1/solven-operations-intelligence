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
  and (eventually) report generation into one pipeline.

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
│   └── report_generator.py  # OperationsReportGenerator (placeholder)
│
├── generate_data.py         # Synthetic data generator
├── app.py                   # Application entry point
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
# then edit .env and add your OpenAI API key

# 5. Generate the synthetic datasets
python generate_data.py

# 6. Run the application
python app.py
```

## Future Roadmap

- [ ] Implement `analyze_deliveries()` — delivery time trends, delay
      rates, and warehouse overload detection.
- [ ] Implement `analyze_complaints()` — complaint volume, category
      breakdown, sentiment trends, and correlation with delivery
      performance.
- [ ] Implement `analyze_inventory()` — stockout risk, stock-to-sales
      ratios, and warehouse-level inventory health scoring.
- [ ] Implement `OperationsReportGenerator` — prompt construction and
      OpenAI integration to produce narrative executive reports.
- [ ] Add automated tests for the analysis layer.
- [ ] Add a lightweight report export (Markdown/PDF).
- [ ] Add a simple dashboard/UI for exploring generated reports.

## License

This is a portfolio project and is not licensed for production use.
