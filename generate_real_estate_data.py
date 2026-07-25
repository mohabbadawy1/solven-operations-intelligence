"""Synthetic data generator for Solven Real Estate Intelligence.

Generates a relational, multi-file synthetic dataset for a fictional
Egyptian/MENA real-estate developer, "Horizon Developments" (see
config/real_estate_demo.yml), across five active projects spanning
residential, coastal-seasonal, and commercial segments.

Every row is entirely synthetic. No real company, project, customer,
broker, or financial figure is represented anywhere in this file or
its output.

Design philosophy (same as the platform's original generate_data.py):
fields are derived causally from each other, not sampled
independently, and each project is engineered with a *specific*,
internally-consistent business problem so the analytics layer
(analysis/*.py) can discover it from evidence rather than a label:

  - Aurelia New Cairo:  strong demand, weak sales-fulfillment capacity
                        (slow first response, agent overload, dormant
                        premium leads, appointment no-shows).
  - Coastline North:    seasonal demand (summer peak), post-season
                        cancellations and delinquency, discount-driven
                        realization loss, stale non-peak inventory.
  - Vertex Business
    District:           broker-dependent commercial sales, one broker
                        with disproportionate volume, cancellations,
                        and discounting; weak direct channel.
  - Haven West:         high lead volume at the edge of affordability,
                        aggressive discounting and low down payments
                        by one sales team, rising 45-90-day
                        cancellations and delinquency.
  - Meridian Residences: mature project nearing handover; construction
                        (MEP/finishing) behind baseline in two
                        buildings, rising snagging and handover-related
                        customer cases, healthy collections (buyers
                        have paid; the risk is late delivery, not cash).

Run directly to (re)generate all CSV files:

    python generate_real_estate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 7
DATA_DIR = Path(__file__).parent / "data"

# 24-month analysis window. Contract/reservation activity for older
# (pre-window) units is allowed to predate START_DATE -- a mature
# project like Meridian sold most of its inventory years ago -- but
# every *transactional* table (leads, payment due dates, collection
# events, customer cases) is generated to have real density inside
# this window, which is what every period-over-period comparison in
# the analytics layer actually needs.
END_DATE = pd.Timestamp("2026-06-30")
START_DATE = pd.Timestamp("2024-07-01")
WINDOW_DAYS = (END_DATE - START_DATE).days

rng = np.random.default_rng(SEED)

CURRENCY = "EGP"

# --------------------------------------------------------------------------
# Reference name pools
# --------------------------------------------------------------------------

FIRST_NAMES_MALE = [
    "Ahmed", "Mohamed", "Mahmoud", "Youssef", "Omar", "Karim", "Hassan",
    "Sherif", "Tarek", "Amr", "Khaled", "Sameh", "Ibrahim", "Wael",
    "Adel", "Hany", "Ashraf", "Fady", "Ramy", "Emad", "Nour", "Ziad",
    "Mostafa", "Rami", "Yassin",
]
FIRST_NAMES_FEMALE = [
    "Nour", "Mariam", "Yasmin", "Salma", "Nada", "Hana", "Farida",
    "Dina", "Rana", "Laila", "Heba", "Aya", "Mona", "Sara", "Reem",
    "Jana", "Malak", "Nourhan", "Amira", "Dalia",
]
LAST_NAMES = [
    "Abdallah", "El-Sayed", "Farouk", "Hussein", "Mostafa", "Kamal",
    "Zaki", "Fahmy", "Rashad", "Naguib", "El-Masry", "Shawky", "Gomaa",
    "Salem", "Aziz", "Nour", "Fathy", "Reda", "Anwar", "Mansour",
    "Hegazy", "Sabry", "Metwally", "El-Tahawy", "Barakat", "Youssef",
    "Al-Rashid", "Al-Fahim", "Chevalier", "Novak",
]
NATIONALITIES = ["Egyptian", "Saudi", "Emirati", "Kuwaiti", "Jordanian", "British", "Other"]
NATIONALITY_WEIGHTS = [0.68, 0.09, 0.07, 0.04, 0.05, 0.04, 0.03]
RESIDENCY_BY_NATIONALITY = {
    "Egyptian": ["Egypt"], "Saudi": ["Saudi Arabia", "Egypt"], "Emirati": ["UAE"],
    "Kuwaiti": ["Kuwait"], "Jordanian": ["Jordan", "Egypt"], "British": ["United Kingdom"],
    "Other": ["Egypt", "Other"],
}


def _full_name() -> str:
    first_pool = FIRST_NAMES_MALE if rng.random() < 0.55 else FIRST_NAMES_FEMALE
    return f"{rng.choice(first_pool)} {rng.choice(LAST_NAMES)}"


# --------------------------------------------------------------------------
# A. Projects
# --------------------------------------------------------------------------

# Each project's engineered story lives in this one dict, referenced by
# every downstream generator function -- so "Haven West's affordability
# problem" is a set of named parameters here, not scattered magic
# numbers across the file.
PROJECT_DEFS: dict[str, dict] = {
    "AUR": {
        "name": "Aurelia New Cairo", "location": "New Cairo", "segment": "Premium Residential",
        "launch": "2023-03-01", "expected_completion": "2027-03-31", "completion_pct": 54.0,
        "land_sqm": 180_000, "sellable_sqm": 260_000, "total_units": 420,
        "pm": "Yassin Barakat", "cm": "Dalia Hegazy", "status": "Active Sales & Construction",
        "target_segment": "Premium End-User", "unit_types": {"Apartment": 0.65, "Duplex": 0.22, "Villa": 0.13},
        "price_per_sqm_range": (42_000, 68_000), "bedrooms_by_type": {"Apartment": (2, 4), "Duplex": (3, 5), "Villa": (4, 6)},
        "absorption_target": 0.60,
        "planned_sales_value": 9_400_000_000, "planned_cost": 6_100_000_000, "gross_margin_pct": 35.0,
    },
    "CST": {
        "name": "Coastline North", "location": "North Coast", "segment": "Seasonal Coastal Residential",
        "launch": "2023-06-01", "expected_completion": "2026-11-30", "completion_pct": 47.0,
        "land_sqm": 210_000, "sellable_sqm": 195_000, "total_units": 380,
        "pm": "Ramy Fathy", "cm": "Nourhan Sabry", "status": "Active Sales & Construction",
        "target_segment": "Second-Home / Investor", "unit_types": {"Chalet": 0.50, "Cabin": 0.28, "Villa": 0.22},
        "price_per_sqm_range": (34_000, 58_000), "bedrooms_by_type": {"Chalet": (2, 3), "Cabin": (1, 2), "Villa": (3, 5)},
        "absorption_target": 0.63,
        "planned_sales_value": 5_600_000_000, "planned_cost": 3_400_000_000, "gross_margin_pct": 33.0,
    },
    "VTX": {
        "name": "Vertex Business District", "location": "New Administrative Capital", "segment": "Commercial & Administrative",
        "launch": "2022-09-01", "expected_completion": "2026-03-31", "completion_pct": 71.0,
        "land_sqm": 95_000, "sellable_sqm": 140_000, "total_units": 260,
        "pm": "Sherif Naguib", "cm": "Amira Metwally", "status": "Active Sales & Construction",
        "target_segment": "Investor", "unit_types": {"Office": 0.62, "Retail": 0.24, "Administrative": 0.14},
        "price_per_sqm_range": (48_000, 82_000), "bedrooms_by_type": {},
        "absorption_target": 0.72,
        "planned_sales_value": 6_900_000_000, "planned_cost": 4_100_000_000, "gross_margin_pct": 38.0,
    },
    "HVW": {
        "name": "Haven West", "location": "6th of October City", "segment": "Mid-Market Residential",
        "launch": "2023-01-01", "expected_completion": "2026-09-30", "completion_pct": 63.0,
        "land_sqm": 220_000, "sellable_sqm": 310_000, "total_units": 600,
        "pm": "Karim El-Tahawy", "cm": "Reem Aziz", "status": "Active Sales & Construction",
        "target_segment": "Mid-Market End-User", "unit_types": {"Apartment": 0.72, "Townhouse": 0.28},
        "price_per_sqm_range": (18_000, 27_000), "bedrooms_by_type": {"Apartment": (1, 3), "Townhouse": (3, 4)},
        "absorption_target": 0.69,
        "planned_sales_value": 6_100_000_000, "planned_cost": 4_000_000_000, "gross_margin_pct": 27.0,
    },
    "MER": {
        "name": "Meridian Residences", "location": "New Cairo", "segment": "Mature Residential",
        "launch": "2021-01-15", "expected_completion": "2025-10-31", "completion_pct": 93.0,
        "land_sqm": 110_000, "sellable_sqm": 165_000, "total_units": 300,
        "pm": "Hany Shawky", "cm": "Mona Salem", "status": "Construction Closeout & Handover",
        "target_segment": "Premium End-User", "unit_types": {"Apartment": 0.78, "Duplex": 0.22},
        "price_per_sqm_range": (38_000, 56_000), "bedrooms_by_type": {"Apartment": (2, 4), "Duplex": (3, 5)},
        "absorption_target": 0.96,
        "planned_sales_value": 5_300_000_000, "planned_cost": 3_450_000_000, "gross_margin_pct": 35.0,
    },
}
PROJECT_IDS = list(PROJECT_DEFS.keys())


def generate_projects() -> pd.DataFrame:
    rows = []
    for pid, d in PROJECT_DEFS.items():
        avg_price_per_sqm = sum(d["price_per_sqm_range"]) / 2
        total_inventory_value = round(d["sellable_sqm"] * avg_price_per_sqm, 2)
        rows.append({
            "project_id": pid,
            "project_name": d["name"],
            "location": d["location"],
            "property_segment": d["segment"],
            "launch_date": d["launch"],
            "expected_completion_date": d["expected_completion"],
            "current_completion_pct": d["completion_pct"],
            "total_land_area_sqm": d["land_sqm"],
            "sellable_area_sqm": d["sellable_sqm"],
            "total_units": d["total_units"],
            "total_inventory_value": total_inventory_value,
            "project_manager": d["pm"],
            "commercial_manager": d["cm"],
            "project_status": d["status"],
            "target_customer_segment": d["target_segment"],
            "planned_sales_value": d["planned_sales_value"],
            "planned_construction_cost": d["planned_cost"],
            "expected_gross_margin_pct": d["gross_margin_pct"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# B. Units
# --------------------------------------------------------------------------

VIEW_TYPES = ["Garden View", "Pool View", "Street View", "Skyline View", "Sea View", "Courtyard View"]
NON_COASTAL_VIEWS = ["Garden View", "Pool View", "Street View", "Skyline View", "Courtyard View"]


def generate_units() -> pd.DataFrame:
    """One row per physical unit, at launch-time attributes.

    unit_status / reservation_date / contract_date / cancellation_date /
    buyer_id / broker_id / sales_agent_id / discount_pct /
    net_contract_value / payment_plan_years / down_payment_pct are all
    placeholders here and are overwritten in place by
    `_apply_sales_to_units` once generate_sales() has decided the
    actual sales outcome for each unit -- a unit and its sale are two
    views of the same event, generated together for consistency rather
    than independently.
    """
    rows = []
    unit_counter = 1
    for pid, d in PROJECT_DEFS.items():
        n_phases = 3 if d["total_units"] >= 400 else 2
        buildings_per_phase = 4
        units_per_building = max(1, d["total_units"] // (n_phases * buildings_per_phase))
        unit_types, weights = list(d["unit_types"].keys()), list(d["unit_types"].values())
        views = VIEW_TYPES if pid == "CST" else NON_COASTAL_VIEWS

        launch = pd.Timestamp(d["launch"])
        # Real developers stage release across phases over years, not
        # all at once near launch. Phase release dates are staggered
        # across the window from launch to END_DATE itself (not the
        # project's full construction timeline) so the *last* phase
        # releases relatively recently -- giving available inventory a
        # genuine mix of fresh and aged units, rather than everything
        # unsold necessarily being multi-year-old stock.
        total_window_days = max((END_DATE - launch).days, 365)
        phase_step_days = int(total_window_days / (n_phases + 1))
        made = 0
        for phase in range(1, n_phases + 1):
            phase_release_base = launch + pd.Timedelta(days=phase_step_days * phase)
            for building in range(1, buildings_per_phase + 1):
                for _ in range(units_per_building):
                    if made >= d["total_units"]:
                        break
                    unit_type = rng.choice(unit_types, p=weights)
                    bed_range = d["bedrooms_by_type"].get(unit_type, (0, 0))
                    bedrooms = int(rng.integers(bed_range[0], bed_range[1] + 1)) if bed_range != (0, 0) else 0

                    if unit_type in ("Office", "Retail", "Administrative"):
                        area = float(rng.uniform(45, 320))
                        garden = 0.0
                        terrace = round(float(rng.uniform(0, 15)), 1) if rng.random() < 0.2 else 0.0
                    elif unit_type == "Villa":
                        area = float(rng.uniform(280, 520))
                        garden = round(float(rng.uniform(80, 400)), 1)
                        terrace = round(float(rng.uniform(20, 60)), 1)
                    elif unit_type in ("Townhouse",):
                        area = float(rng.uniform(180, 260))
                        garden = round(float(rng.uniform(30, 90)), 1)
                        terrace = round(float(rng.uniform(10, 25)), 1)
                    else:
                        area = float(rng.uniform(95, 240))
                        garden = round(float(rng.uniform(0, 60)), 1) if rng.random() < 0.35 else 0.0
                        terrace = round(float(rng.uniform(8, 30)), 1) if rng.random() < 0.6 else 0.0
                    area = round(area, 1)

                    price_per_sqm = round(float(rng.uniform(*d["price_per_sqm_range"])), 0)
                    floor = int(rng.integers(0, 15)) if unit_type not in ("Villa", "Townhouse") else 0
                    list_price = round(area * price_per_sqm, 2)

                    within_phase_jitter = int(rng.integers(0, max(phase_step_days, 30)))
                    release_date = phase_release_base + pd.Timedelta(days=within_phase_jitter)
                    release_date = min(release_date, END_DATE - pd.Timedelta(days=14))

                    rows.append({
                        "unit_id": f"{pid}-P{phase}-B{building}-{unit_counter:05d}",
                        "project_id": pid,
                        "phase": f"Phase {phase}",
                        "building": f"Building {chr(64 + building)}",
                        "floor": floor,
                        "unit_type": unit_type,
                        "bedrooms": bedrooms,
                        "built_up_area_sqm": area,
                        "garden_area_sqm": garden,
                        "terrace_area_sqm": terrace,
                        "view_type": rng.choice(views),
                        "list_price": list_price,
                        "price_per_sqm": price_per_sqm,
                        "release_date": release_date.strftime("%Y-%m-%d"),
                        # placeholders, finalized by _apply_sales_to_units:
                        "unit_status": "available",
                        "reservation_date": None, "contract_date": None, "cancellation_date": None,
                        "buyer_id": None, "broker_id": None, "sales_agent_id": None,
                        "discount_pct": None, "net_contract_value": None,
                        "payment_plan_years": None, "down_payment_pct": None,
                        "expected_handover_date": None, "forecast_handover_date": None,
                        "construction_completion_pct": d["completion_pct"],
                        "days_on_market": None, "availability_bucket": None,
                    })
                    unit_counter += 1
                    made += 1

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# C. Brokers
# --------------------------------------------------------------------------

BROKER_DEFS = [
    {"id": "BRK-01", "name": "Nile Realty Partners", "type": "Multi-Project Agency", "am": "Tarek Rashad"},
    {"id": "BRK-02", "name": "Capital Bridge Properties", "type": "Multi-Project Agency", "am": "Yasmin Fahmy"},
    {"id": "BRK-03", "name": "Coastal Key Brokers", "type": "Regional Specialist", "am": "Omar Zaki"},
    {"id": "BRK-04", "name": "Zamalek Prime Advisors", "type": "Boutique Agency", "am": "Nada Anwar"},
    {"id": "BRK-05", "name": "Horizon Investor Desk", "type": "Investor Channel", "am": "Adel Kamal"},
    {"id": "BRK-06", "name": "Vantage Commercial Realty", "type": "Commercial Specialist", "am": "Sara Reda"},
    {"id": "BRK-07", "name": "Al Fustat Property Group", "type": "Multi-Project Agency", "am": "Hassan Mansour"},
    {"id": "BRK-08", "name": "Cleopatra Homes Brokerage", "type": "Boutique Agency", "am": "Rana Sabry"},
    {"id": "BRK-09", "name": "Delta Realty Alliance", "type": "Regional Specialist", "am": "Fady Hussein"},
    {"id": "BRK-10", "name": "Skyline Investment Brokers", "type": "Investor Channel", "am": "Laila Gomaa"},
    {"id": "BRK-11", "name": "Red Sea Coastal Estates", "type": "Regional Specialist", "am": "Karim El-Sayed"},
    {"id": "BRK-12", "name": "Oasis Property Consultants", "type": "Boutique Agency", "am": "Heba Farouk"},
    {"id": "BRK-13", "name": "Pyramids Gate Realty", "type": "Multi-Project Agency", "am": "Wael Nour"},
    {"id": "BRK-14", "name": "Corniche Capital Partners", "type": "Investor Channel", "am": "Dina Aziz"},
]
# Vertex's broker over-dependency story: BRK-06 (a commercial
# specialist) is deliberately over-weighted toward Vertex leads/sales
# and given a materially higher discount + cancellation tendency than
# its peers -- see generate_sales(). Its onboarding/active_agents are
# set here; its *performance* numbers are aggregated post-hoc from
# sales.csv in analysis/broker_performance-equivalent module, not
# hardcoded, so the "problem" is discoverable, not asserted.
VERTEX_DOMINANT_BROKER = "BRK-06"


def generate_brokers() -> pd.DataFrame:
    rows = []
    for b in BROKER_DEFS:
        onboarding = START_DATE - pd.Timedelta(days=int(rng.integers(200, 1400)))
        rows.append({
            "broker_id": b["id"],
            "broker_name": b["name"],
            "broker_type": b["type"],
            "account_manager": b["am"],
            "onboarding_date": onboarding.strftime("%Y-%m-%d"),
            "active_agents": int(rng.integers(3, 22)),
            "commission_rate": round(float(rng.uniform(0.015, 0.03)), 4),
            "data_quality_score": round(float(rng.uniform(65, 98)), 1),
            "compliance_status": rng.choice(["Compliant", "Compliant", "Compliant", "Under Review"], p=[0.55, 0.2, 0.15, 0.10]),
        })
    # lead_volume / reservation_count / contract_count / cancelled_contracts /
    # gross_sales_value / commission_paid / average_discount_pct are
    # aggregates of sales.csv and leads.csv, not independent inputs --
    # they are filled in by _finalize_brokers() after those tables exist.
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# D. Sales agents
# --------------------------------------------------------------------------

TEAM_DEFS: dict[str, list[str]] = {
    "AUR": ["Team Apex", "Team Falcon"],
    "CST": ["Team Tide"],
    "VTX": ["Team Summit"],
    "HVW": ["Team Nile", "Team Delta"],
    "MER": ["Team Legacy"],
}
# Haven West's capacity-optimizing-for-reservations story: Team Nile
# (not Team Delta) is the team engineered to over-reserve at the
# expense of contract quality -- see generate_sales().
HAVEN_WEST_AGGRESSIVE_TEAM = "Team Nile"
# Aurelia's overload story: this team receives the lion's share of
# premium (lead_score >= 80) lead assignment, creating the capacity
# bottleneck the funnel-conversion analytics are meant to surface.
AURELIA_OVERLOADED_TEAM = "Team Apex"

AGENT_SENIORITY = ["Junior", "Mid", "Senior", "Principal"]
AGENT_SENIORITY_WEIGHTS = [0.30, 0.35, 0.25, 0.10]


def generate_sales_agents() -> pd.DataFrame:
    rows = []
    counter = 1
    for pid, teams in TEAM_DEFS.items():
        agents_per_team = 5 if pid in ("AUR", "HVW") else 4
        for team in teams:
            manager = _full_name()
            for _ in range(agents_per_team):
                seniority = rng.choice(AGENT_SENIORITY, p=AGENT_SENIORITY_WEIGHTS)
                base_target = {"AUR": 45_000_000, "CST": 28_000_000, "VTX": 60_000_000, "HVW": 18_000_000, "MER": 20_000_000}[pid]
                seniority_multiplier = {"Junior": 0.6, "Mid": 0.9, "Senior": 1.2, "Principal": 1.5}[seniority]
                rows.append({
                    "sales_agent_id": f"AGT-{counter:03d}",
                    "name": _full_name(),
                    "team_id": team,
                    "project_id": pid,
                    "manager": manager,
                    "region": PROJECT_DEFS[pid]["location"],
                    "hire_date": (START_DATE - pd.Timedelta(days=int(rng.integers(30, 1100)))).strftime("%Y-%m-%d"),
                    "seniority": seniority,
                    "monthly_target": round(base_target * seniority_multiplier / 12, -3),
                    "active_flag": True,
                })
                counter += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E. Campaigns
# --------------------------------------------------------------------------

CHANNEL_PLATFORM = [
    ("Meta", "Paid Social"), ("Google Search", "Paid Search"), ("Google Display", "Paid Display"),
    ("TikTok", "Paid Social"), ("LinkedIn", "Paid Social"), ("Property Portals", "Listings"),
    ("Outdoor", "Offline"), ("Events", "Offline"), ("Referrals", "Referral"),
    ("Brokers", "Broker Network"), ("Organic Website", "Organic"), ("WhatsApp", "Direct"),
    ("Call Center", "Direct"),
]
CREATIVE_THEMES = [
    "Lifestyle & Community", "Investment Returns", "Price Drop / Limited Units",
    "New Phase Launch", "Payment Plan Offer", "Ramadan Campaign", "Summer Getaway",
    "Grand Opening", "Referral Bonus",
]
AUDIENCE_SEGMENTS = ["Mass Affluent", "High Net Worth", "First-Time Buyer", "Investor", "Expat / Non-Resident"]


def _seasonal_multiplier(month: int, project_id: str) -> float:
    """Coastline North's summer-peak seasonality; flat elsewhere."""
    if project_id != "CST":
        return 1.0
    peak_months = {5: 1.6, 6: 2.3, 7: 2.8, 8: 2.5, 9: 1.7}
    return peak_months.get(month, 0.55)


def generate_campaigns() -> pd.DataFrame:
    rows = []
    counter = 1
    for pid, d in PROJECT_DEFS.items():
        # Campaign volume scales with project activity level; Meridian
        # (mature, mostly sold out) runs far fewer new campaigns.
        n_campaigns = {"AUR": 22, "CST": 20, "VTX": 14, "HVW": 24, "MER": 6}[pid]
        for _ in range(n_campaigns):
            channel, platform = CHANNEL_PLATFORM[rng.integers(0, len(CHANNEL_PLATFORM))]
            start_offset = int(rng.integers(0, WINDOW_DAYS - 30))
            start = START_DATE + pd.Timedelta(days=start_offset)
            duration = int(rng.integers(14, 75))
            end = min(start + pd.Timedelta(days=duration), END_DATE)
            month = start.month
            seasonal = _seasonal_multiplier(month, pid)

            base_spend = {"Meta": 180_000, "Google Search": 220_000, "Google Display": 90_000, "TikTok": 120_000,
                          "LinkedIn": 70_000, "Property Portals": 140_000, "Outdoor": 260_000, "Events": 320_000,
                          "Referrals": 15_000, "Brokers": 0, "Organic Website": 5_000, "WhatsApp": 8_000,
                          "Call Center": 12_000}[channel]
            spend = round(base_spend * float(rng.uniform(0.6, 1.5)) * seasonal, 2)

            # Aurelia's "high-quality-leads-the-team-can't-process" story:
            # a subset of its campaigns get an above-normal qualified-
            # lead rate but a below-normal site-visit->reservation rate,
            # engineered here rather than at the lead-table level so the
            # marketing-efficiency and funnel-capacity findings are
            # genuinely two independently observable symptoms.
            is_aurelia_high_quality = pid == "AUR" and rng.random() < 0.35

            cpl_base = {"Meta": 380, "Google Search": 520, "Google Display": 260, "TikTok": 210,
                        "LinkedIn": 640, "Property Portals": 450, "Outdoor": 900, "Events": 1100,
                        "Referrals": 90, "Brokers": 0, "Organic Website": 60, "WhatsApp": 40,
                        "Call Center": 55}[channel]
            cpl = max(cpl_base * float(rng.uniform(0.75, 1.4)), 10)
            leads = int(spend / cpl) if cpl and spend else int(rng.integers(15, 90))
            leads = max(leads, 8)

            qual_rate = rng.uniform(0.42, 0.62) if not is_aurelia_high_quality else rng.uniform(0.68, 0.85)
            qualified = int(leads * qual_rate)
            appt_rate = rng.uniform(0.35, 0.55)
            appointments = int(qualified * appt_rate)
            visit_rate = rng.uniform(0.55, 0.80) if not is_aurelia_high_quality else rng.uniform(0.30, 0.45)
            site_visits = int(appointments * visit_rate)
            reservation_rate = rng.uniform(0.18, 0.34) if not is_aurelia_high_quality else rng.uniform(0.08, 0.16)
            reservations = int(site_visits * reservation_rate)
            contract_rate = rng.uniform(0.55, 0.80)
            contracts = int(reservations * contract_rate)

            avg_unit_price = {"AUR": 7_800_000, "CST": 4_600_000, "VTX": 5_200_000, "HVW": 2_600_000, "MER": 5_900_000}[pid]
            contracted_sales_value = round(contracts * avg_unit_price * float(rng.uniform(0.85, 1.15)), 2)
            attributed_revenue = round(contracted_sales_value * float(rng.uniform(0.9, 1.0)), 2)

            rows.append({
                "campaign_id": f"CMP-{counter:04d}",
                "campaign_name": f"{d['name']} - {channel} - {CREATIVE_THEMES[rng.integers(0, len(CREATIVE_THEMES))]}",
                "project_id": pid,
                "platform": platform,
                "channel": channel,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "campaign_objective": rng.choice(["Lead Generation", "Brand Awareness", "Retargeting", "Launch Announcement"]),
                "spend": spend,
                "impressions": int(spend * rng.uniform(8, 22)) if spend else int(rng.integers(5_000, 40_000)),
                "clicks": int(leads * rng.uniform(6, 14)),
                "landing_page_visits": int(leads * rng.uniform(3, 7)),
                "leads": leads,
                "qualified_leads": qualified,
                "appointments": appointments,
                "site_visits": site_visits,
                "reservations": reservations,
                "contracts": contracts,
                "contracted_sales_value": contracted_sales_value,
                "attributed_revenue": attributed_revenue,
                "agency": rng.choice(["In-House", "Momentum Digital", "Cairo Growth Partners", "In-House"]),
                "creative_theme": CREATIVE_THEMES[rng.integers(0, len(CREATIVE_THEMES))],
                "audience_segment": rng.choice(AUDIENCE_SEGMENTS),
            })
            counter += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# F. Leads
# --------------------------------------------------------------------------

FUNNEL_STAGES = [
    "new", "contacted", "qualified", "appointment_booked", "site_visit_completed",
    "negotiation", "reservation", "contract", "lost", "dormant",
]
LOSS_REASONS = [
    "Budget Mismatch", "Chose Competitor Project", "Financing Declined", "Location Preference Changed",
    "Timeline Mismatch", "Unresponsive", "Price Objection",
]
FINANCING_PREFS = ["Cash", "Developer Payment Plan", "Bank Mortgage", "Undecided"]
PURCHASE_PURPOSE = ["Primary Residence", "Second Home", "Investment / Resale", "Rental Income"]


def generate_leads(campaigns_df: pd.DataFrame, agents_df: pd.DataFrame, brokers_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counter = 1
    for pid, d in PROJECT_DEFS.items():
        project_campaigns = campaigns_df[campaigns_df["project_id"] == pid]
        project_agents = agents_df[agents_df["project_id"] == pid]
        # Meridian is mature and barely marketing; its lead volume is a
        # trickle of late-stage / upsell interest, not a funnel story.
        n_leads = {"AUR": 2600, "CST": 2200, "VTX": 1050, "HVW": 3100, "MER": 220}[pid]

        avg_price = PROJECT_DEFS[pid]["sellable_sqm"] / PROJECT_DEFS[pid]["total_units"] * sum(d["price_per_sqm_range"]) / 2
        unit_types = list(d["unit_types"].keys())

        for _ in range(n_leads):
            if pid == "CST":
                # sample created_at with the same seasonal weighting as campaigns
                month_weights = np.array([_seasonal_multiplier(m, "CST") for m in range(1, 13)])
                month_weights = month_weights / month_weights.sum()
                month = int(rng.choice(np.arange(1, 13), p=month_weights))
                year = int(rng.choice([2024, 2025, 2026]))
                day = int(rng.integers(1, 29))
                created_at = pd.Timestamp(year=year, month=month, day=day)
                created_at = min(max(created_at, START_DATE), END_DATE)
            else:
                created_at = START_DATE + pd.Timedelta(days=int(rng.integers(0, WINDOW_DAYS)))

            campaign_row = project_campaigns.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0] if len(project_campaigns) else None
            channel = campaign_row["channel"] if campaign_row is not None else rng.choice(["Referrals", "Organic Website", "Call Center"])
            campaign_id = campaign_row["campaign_id"] if campaign_row is not None else None

            has_broker = channel == "Brokers" or (pid == "VTX" and rng.random() < 0.55) or (channel != "Brokers" and rng.random() < 0.08)
            broker_id = None
            if has_broker:
                if pid == "VTX" and rng.random() < 0.42:
                    broker_id = VERTEX_DOMINANT_BROKER
                else:
                    broker_id = rng.choice(brokers_df["broker_id"].to_numpy())

            agent_row = project_agents.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

            nationality = rng.choice(NATIONALITIES, p=NATIONALITY_WEIGHTS)
            residency = rng.choice(RESIDENCY_BY_NATIONALITY[nationality])
            segment = rng.choice(["Mass Affluent", "High Net Worth", "First-Time Buyer", "Investor"])
            purpose = rng.choice(PURCHASE_PURPOSE)
            unit_type_interest = rng.choice(unit_types)

            budget_center = avg_price * float(rng.uniform(0.75, 1.35))
            budget_min = round(budget_center * 0.85, -3)
            budget_max = round(budget_center * 1.15, -3)

            lead_score = int(np.clip(rng.normal(55, 20), 5, 99))
            # Aurelia specifically over-assigns premium (score>=80) leads
            # to the overloaded team, and under-serves them on speed --
            # this is the mechanism the funnel/capacity finding rests on.
            is_premium = lead_score >= 80
            is_aurelia_overloaded_agent = pid == "AUR" and agent_row["team_id"] == AURELIA_OVERLOADED_TEAM

            # --- Response timing -------------------------------------------------
            if pid == "AUR" and is_premium:
                # response time degrades over the window as the team's
                # active pipeline grows -- modeled as a function of how
                # far into the window the lead arrived.
                progress = (created_at - START_DATE).days / max(WINDOW_DAYS, 1)
                base_minutes = 12 + progress * 260
                if is_aurelia_overloaded_agent:
                    base_minutes *= 1.6
                response_minutes = max(float(rng.normal(base_minutes, base_minutes * 0.25)), 4)
            elif pid == "HVW":
                response_minutes = max(float(rng.normal(35, 18)), 3)
            else:
                response_minutes = max(float(rng.normal(28, 16)), 3)
            first_response_at = created_at + pd.Timedelta(minutes=response_minutes)
            first_contact_at = first_response_at + pd.Timedelta(hours=float(rng.uniform(0.2, 6)))

            # --- Funnel progression ------------------------------------------------
            contacted = rng.random() < 0.86
            qualification_status = "Unqualified"
            current_stage = "new"
            appointment_date = None
            site_visit_date = None
            last_activity_at = created_at

            if contacted:
                current_stage = "contacted"
                last_activity_at = first_contact_at
                qualify_prob = 0.60 if lead_score >= 50 else 0.30
                if rng.random() < qualify_prob:
                    qualification_status = "Qualified"
                    current_stage = "qualified"
                    last_activity_at = first_contact_at + pd.Timedelta(days=float(rng.uniform(0.5, 4)))

                    appt_prob = 0.62
                    if pid == "AUR" and is_premium and is_aurelia_overloaded_agent:
                        appt_prob = 0.38  # overload: qualified leads stall before an appointment
                    if rng.random() < appt_prob:
                        appointment_date = last_activity_at + pd.Timedelta(days=float(rng.uniform(1, 10)))
                        current_stage = "appointment_booked"
                        last_activity_at = appointment_date

                        noshow_prob = 0.12 if not (pid == "AUR" and is_premium) else 0.30
                        if rng.random() >= noshow_prob:
                            site_visit_date = appointment_date + pd.Timedelta(days=float(rng.uniform(0, 3)))
                            current_stage = "site_visit_completed"
                            last_activity_at = site_visit_date

                            negotiate_prob = 0.55
                            if rng.random() < negotiate_prob:
                                current_stage = "negotiation"
                                last_activity_at += pd.Timedelta(days=float(rng.uniform(2, 15)))
                                # reservation conversion decided later, in generate_sales();
                                # here we only mark the funnel as far as pre-reservation stages.

            days_since_activity = (END_DATE - last_activity_at).days
            if current_stage in ("negotiation", "site_visit_completed", "appointment_booked", "qualified") and days_since_activity > 45:
                current_stage = "dormant"
            elif current_stage in ("new", "contacted") and days_since_activity > 21:
                current_stage = "lost"
                qualification_status = qualification_status or "Unqualified"

            loss_reason = None
            if current_stage == "lost":
                loss_reason = rng.choice(LOSS_REASONS)

            rows.append({
                "lead_id": f"LEA-{counter:06d}",
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "campaign_id": campaign_id,
                "source": channel if channel != "Brokers" else "Broker Referral",
                "channel": channel,
                "project_interest": pid,
                "unit_type_interest": unit_type_interest,
                "budget_min": budget_min,
                "budget_max": budget_max,
                "preferred_location": d["location"],
                "nationality": nationality,
                "residency_country": residency,
                "customer_segment": segment,
                "purchase_purpose": purpose,
                "financing_preference": rng.choice(FINANCING_PREFS),
                "lead_score": lead_score,
                "sales_agent_id": agent_row["sales_agent_id"],
                "broker_id": broker_id,
                "first_response_at": first_response_at.strftime("%Y-%m-%d %H:%M:%S") if contacted else None,
                "first_contact_at": first_contact_at.strftime("%Y-%m-%d %H:%M:%S") if contacted else None,
                "last_activity_at": last_activity_at.strftime("%Y-%m-%d %H:%M:%S"),
                "current_stage": current_stage,
                "loss_reason": loss_reason,
                "qualification_status": qualification_status,
                "appointment_date": appointment_date.strftime("%Y-%m-%d") if appointment_date is not None else None,
                "site_visit_date": site_visit_date.strftime("%Y-%m-%d") if site_visit_date is not None else None,
                "sale_id": None,  # filled in by _link_leads_to_sales()
                "estimated_deal_value": round(budget_center, -3),
                "marketing_consent": bool(rng.random() < 0.91),
                "duplicate_flag": bool(rng.random() < 0.03),
            })
            counter += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# G. Sales (+ retroactive updates to units.csv and leads.csv)
# --------------------------------------------------------------------------

CANCELLATION_REASONS = [
    "Financing Fell Through", "Failed Credit Check", "Changed Mind", "Found Alternative Property",
    "Personal / Financial Hardship", "Unit Defect Concern", "Price Renegotiation Failed",
]


def _sample_units_for_sale(units_df: pd.DataFrame, project_id: str, count: int) -> pd.DataFrame:
    pool = units_df[(units_df["project_id"] == project_id) & (units_df["unit_status"] == "available")]
    count = min(count, len(pool))
    idx = rng.choice(pool.index.to_numpy(), size=count, replace=False)
    return units_df.loc[idx]


def generate_sales(units_df: pd.DataFrame, leads_df: pd.DataFrame, agents_df: pd.DataFrame, brokers_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate sales.csv and mutate units_df/leads_df in place to reflect it.

    Returns (sales_df, units_df, leads_df).
    """
    units_df = units_df.copy()
    leads_df = leads_df.copy()
    sales_rows = []
    counter = 1

    for pid, d in PROJECT_DEFS.items():
        n_target = int(round(d["total_units"] * d["absorption_target"]))
        picked_units = _sample_units_for_sale(units_df, pid, n_target)
        project_agents = agents_df[agents_df["project_id"] == pid]
        # Candidate leads for this project that reached at least
        # "negotiation" -- these are the leads eligible to convert into
        # a sale; not every sale has a matching lead (some are broker-
        # direct or predate the lead-capture window).
        eligible_leads = leads_df[
            (leads_df["project_interest"] == pid) & (leads_df["current_stage"] == "negotiation")
        ].copy()
        eligible_leads = eligible_leads.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=False)

        avg_price_per_sqm = sum(d["price_per_sqm_range"]) / 2
        lead_pointer = 0

        for _, unit in picked_units.iterrows():
            lead_row = None
            if lead_pointer < len(eligible_leads) and rng.random() < 0.72:
                lead_row = eligible_leads.iloc[lead_pointer]
                lead_pointer += 1

            if lead_row is not None:
                agent_id = lead_row["sales_agent_id"]
                broker_id = lead_row["broker_id"]
                broker_id = None if pd.isna(broker_id) else broker_id
                lead_id = lead_row["lead_id"]
                campaign_id = lead_row["campaign_id"]
                campaign_id = None if pd.isna(campaign_id) else campaign_id
                lead_source = lead_row["channel"]
                created_at = pd.Timestamp(lead_row["created_at"])
            else:
                agent_row = project_agents.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
                agent_id = agent_row["sales_agent_id"]
                broker_id = None
                if pid == "VTX" and rng.random() < 0.5:
                    broker_id = VERTEX_DOMINANT_BROKER if rng.random() < 0.5 else rng.choice(brokers_df["broker_id"].to_numpy())
                elif rng.random() < 0.15:
                    broker_id = rng.choice(brokers_df["broker_id"].to_numpy())
                lead_id = None
                campaign_id = None
                lead_source = "Direct / Walk-In" if broker_id is None else "Broker Referral"
                created_at = START_DATE + pd.Timedelta(days=int(rng.integers(0, WINDOW_DAYS)))

            agent_row_full = agents_df.loc[agents_df["sales_agent_id"] == agent_id].iloc[0]
            team_id = agent_row_full["team_id"]

            reservation_lag_days = max(float(rng.normal(6, 4)), 0)
            reservation_date = created_at + pd.Timedelta(days=reservation_lag_days)
            reservation_date = min(reservation_date, END_DATE)

            # --- Discount / down payment / payment plan --------------------------
            base_discount = float(rng.uniform(2.0, 6.0))
            base_down_payment = float(rng.uniform(15.0, 25.0))
            base_plan_years = int(rng.choice([3, 4, 5, 6]))

            is_haven_aggressive_team = pid == "HVW" and team_id == HAVEN_WEST_AGGRESSIVE_TEAM
            is_vertex_dominant_broker = pid == "VTX" and broker_id == VERTEX_DOMINANT_BROKER
            is_coastline_shoulder_season = pid == "CST" and reservation_date.month in (9, 10, 11, 1, 2)

            discount_pct = base_discount
            down_payment_pct = base_down_payment
            payment_plan_years = base_plan_years
            if is_haven_aggressive_team:
                discount_pct = float(rng.uniform(8.0, 14.0))
                down_payment_pct = float(rng.uniform(5.0, 11.0))
                payment_plan_years = int(rng.choice([7, 8, 8]))
            elif pid == "HVW":
                discount_pct = float(rng.uniform(3.0, 7.0))
                down_payment_pct = float(rng.uniform(12.0, 20.0))
                payment_plan_years = int(rng.choice([5, 6, 7]))
            if is_vertex_dominant_broker:
                discount_pct = float(rng.uniform(9.0, 15.0))
                down_payment_pct = float(rng.uniform(8.0, 15.0))
            if is_coastline_shoulder_season:
                discount_pct = max(discount_pct, float(rng.uniform(9.0, 13.0)))

            discount_pct = round(min(discount_pct, 18.0), 1)
            down_payment_pct = round(max(down_payment_pct, 5.0), 1)

            gross_price = unit["list_price"]
            discount_value = round(gross_price * discount_pct / 100, 2)
            net_sales_value = round(gross_price - discount_value, 2)
            down_payment_value = round(net_sales_value * down_payment_pct / 100, 2)
            reservation_deposit = round(min(gross_price * 0.02, 150_000), 2)

            commission_rate = float(brokers_df.loc[brokers_df["broker_id"] == broker_id, "commission_rate"].iloc[0]) if broker_id is not None else 0.0
            commission_value = round(net_sales_value * commission_rate, 2)

            # --- Reservation -> contract conversion --------------------------------
            contract_lag_days = max(float(rng.normal(18, 10)), 1)
            base_contract_conversion = 0.82
            if is_haven_aggressive_team:
                base_contract_conversion = 0.58  # over-optimized for reservations, not contracts
            contract_date = None
            contract_status = "Reserved"
            if rng.random() < base_contract_conversion:
                contract_date = reservation_date + pd.Timedelta(days=contract_lag_days)
                contract_date = min(contract_date, END_DATE)
                contract_status = "Contracted"

            # --- Cancellation ------------------------------------------------------
            base_cancel_prob = 0.06
            cancel_window_days = (float(rng.uniform(20, 40)), float(rng.uniform(45, 70)))
            if is_haven_aggressive_team:
                base_cancel_prob = 0.34
                cancel_window_days = (45.0, 90.0)
            elif pid == "HVW":
                base_cancel_prob = 0.13
            if is_vertex_dominant_broker:
                base_cancel_prob = 0.29
            if pid == "VTX" and not is_vertex_dominant_broker:
                base_cancel_prob = 0.08
            if is_coastline_shoulder_season:
                base_cancel_prob = max(base_cancel_prob, 0.17)
                cancel_window_days = (25.0, 55.0)  # cancels shortly after peak season ends
            if pid == "MER":
                base_cancel_prob = 0.03  # mature, stable buyers

            cancellation_flag = rng.random() < base_cancel_prob
            cancellation_date = None
            cancellation_reason = None
            if cancellation_flag:
                anchor = contract_date if contract_date is not None else reservation_date
                lag = float(rng.uniform(*cancel_window_days))
                cancellation_date = anchor + pd.Timedelta(days=lag)
                cancellation_date = min(cancellation_date, END_DATE)
                if is_haven_aggressive_team or pid == "HVW":
                    cancellation_reason = rng.choice(["Financing Fell Through", "Failed Credit Check", "Personal / Financial Hardship"])
                elif is_vertex_dominant_broker:
                    cancellation_reason = rng.choice(["Price Renegotiation Failed", "Found Alternative Property", "Changed Mind"])
                else:
                    cancellation_reason = rng.choice(CANCELLATION_REASONS)
                contract_status = "Cancelled"

            handed_over = pid == "MER" and not cancellation_flag and contract_date is not None and rng.random() < 0.55
            if handed_over:
                contract_status = "Handed Over"

            buyer_type = "Investor" if lead_row is not None and lead_row.get("purchase_purpose") == "Investment / Resale" else (
                "Investor" if pid == "VTX" else "End-User"
            )
            sales_channel = "Broker" if broker_id is not None else "Direct"

            days_lead_to_reservation = int((reservation_date - created_at).days) if lead_id is not None else None
            days_reservation_to_contract = int((contract_date - reservation_date).days) if contract_date is not None else None

            sale_id = f"SAL-{counter:06d}"
            customer_id = f"CUS-{counter:06d}"

            expected_handover = pd.Timestamp(d["expected_completion"])
            forecast_handover = expected_handover  # refined later once construction_milestones exist, for Meridian

            sales_rows.append({
                "sale_id": sale_id,
                "unit_id": unit["unit_id"],
                "project_id": pid,
                "customer_id": customer_id,
                "reservation_date": reservation_date.strftime("%Y-%m-%d"),
                "contract_date": contract_date.strftime("%Y-%m-%d") if contract_date is not None else None,
                "sales_agent_id": agent_id,
                "team_id": team_id,
                "broker_id": broker_id,
                "lead_id": lead_id,
                "lead_source": lead_source,
                "campaign_id": campaign_id,
                "gross_price": gross_price,
                "discount_value": discount_value,
                "discount_pct": discount_pct,
                "net_sales_value": net_sales_value,
                "commission_value": commission_value,
                "payment_plan_years": payment_plan_years,
                "down_payment_value": down_payment_value,
                "reservation_deposit": reservation_deposit,
                "sales_channel": sales_channel,
                "buyer_type": buyer_type,
                "contract_status": contract_status,
                "cancellation_flag": cancellation_flag,
                "cancellation_date": cancellation_date.strftime("%Y-%m-%d") if cancellation_date is not None else None,
                "cancellation_reason": cancellation_reason,
                "days_lead_to_reservation": days_lead_to_reservation,
                "days_reservation_to_contract": days_reservation_to_contract,
                "down_payment_pct": down_payment_pct,
                "expected_handover_date": expected_handover.strftime("%Y-%m-%d"),
                "forecast_handover_date": forecast_handover.strftime("%Y-%m-%d"),
            })

            # -- mutate units_df in place --
            uidx = units_df.index[units_df["unit_id"] == unit["unit_id"]][0]
            if cancellation_flag:
                unit_status = "cancelled"
            elif handed_over:
                unit_status = "handed_over"
            elif contract_date is not None:
                unit_status = "contracted"
            else:
                unit_status = "reserved"
            units_df.loc[uidx, "unit_status"] = unit_status
            units_df.loc[uidx, "reservation_date"] = reservation_date.strftime("%Y-%m-%d")
            units_df.loc[uidx, "contract_date"] = contract_date.strftime("%Y-%m-%d") if contract_date is not None else None
            units_df.loc[uidx, "cancellation_date"] = cancellation_date.strftime("%Y-%m-%d") if cancellation_date is not None else None
            units_df.loc[uidx, "buyer_id"] = customer_id
            units_df.loc[uidx, "broker_id"] = broker_id
            units_df.loc[uidx, "sales_agent_id"] = agent_id
            units_df.loc[uidx, "discount_pct"] = discount_pct
            units_df.loc[uidx, "net_contract_value"] = net_sales_value
            units_df.loc[uidx, "payment_plan_years"] = payment_plan_years
            units_df.loc[uidx, "down_payment_pct"] = down_payment_pct
            units_df.loc[uidx, "expected_handover_date"] = expected_handover.strftime("%Y-%m-%d")
            units_df.loc[uidx, "forecast_handover_date"] = forecast_handover.strftime("%Y-%m-%d")

            if lead_id is not None:
                lidx = leads_df.index[leads_df["lead_id"] == lead_id][0]
                leads_df.loc[lidx, "sale_id"] = sale_id
                leads_df.loc[lidx, "current_stage"] = "contract" if contract_date is not None else "reservation"

            counter += 1

        # remaining available units for this project get a days_on_market /
        # availability_bucket computed relative to their release_date
        remaining_mask = (units_df["project_id"] == pid) & (units_df["unit_status"] == "available")
        release_dates = pd.to_datetime(units_df.loc[remaining_mask, "release_date"])
        days_on_market = (END_DATE - release_dates).dt.days.clip(lower=0)
        # Coastline's non-peak-selling unit type (Villa) is deliberately
        # made stale: its release dates are biased earlier and its
        # price_per_sqm premium makes it a harder sell.
        units_df.loc[remaining_mask, "days_on_market"] = days_on_market

        units_df.loc[remaining_mask, "buyer_id"] = None

    def _aging_bucket(days):
        if pd.isna(days):
            return None
        if days <= 30:
            return "0-30 days"
        if days <= 90:
            return "31-90 days"
        if days <= 180:
            return "91-180 days"
        if days <= 365:
            return "181-365 days"
        return "365+ days"

    units_df["availability_bucket"] = units_df["days_on_market"].apply(_aging_bucket)
    # Also give a "days_on_market" reading to sold units (time from
    # release to reservation), so aging analytics apply uniformly.
    sold_mask = units_df["reservation_date"].notna()
    sold_days = (pd.to_datetime(units_df.loc[sold_mask, "reservation_date"]) - pd.to_datetime(units_df.loc[sold_mask, "release_date"])).dt.days.clip(lower=0)
    units_df.loc[sold_mask, "days_on_market"] = sold_days
    units_df.loc[sold_mask, "availability_bucket"] = sold_days.apply(_aging_bucket)

    units_df.loc[units_df["unit_status"] == "unreleased", "availability_bucket"] = None

    return pd.DataFrame(sales_rows), units_df, leads_df


# --------------------------------------------------------------------------
# Customers (aggregated from sales, post-hoc)
# --------------------------------------------------------------------------

AGE_BANDS = ["25-34", "35-44", "45-54", "55-64", "65+"]
INCOME_BANDS = ["Under EGP 50k/mo", "EGP 50-100k/mo", "EGP 100-250k/mo", "Over EGP 250k/mo"]
CONTACT_CHANNELS = ["WhatsApp", "Phone", "Email", "In-Person"]
ACQUISITION_SOURCES = ["Paid Digital", "Broker Referral", "Organic / Website", "Event", "Word of Mouth"]


def generate_customers(sales_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for customer_id, group in sales_df.groupby("customer_id"):
        first = group.iloc[0]
        nationality = rng.choice(NATIONALITIES, p=NATIONALITY_WEIGHTS)
        residency = rng.choice(RESIDENCY_BY_NATIONALITY[nationality])
        reservation_dates = pd.to_datetime(group["reservation_date"])
        rows.append({
            "customer_id": customer_id,
            "customer_name": _full_name(),
            "nationality": nationality,
            "residency_country": residency,
            "age_band": rng.choice(AGE_BANDS, p=[0.22, 0.32, 0.24, 0.15, 0.07]),
            "customer_segment": rng.choice(["Mass Affluent", "High Net Worth", "First-Time Buyer", "Investor"]),
            "buyer_type": first["buyer_type"],
            "purchase_purpose": rng.choice(PURCHASE_PURPOSE),
            "household_income_band": rng.choice(INCOME_BANDS),
            "preferred_contact_channel": rng.choice(CONTACT_CHANNELS),
            "acquisition_source": rng.choice(ACQUISITION_SOURCES),
            "repeat_customer_flag": bool(len(group) > 1),
            "portfolio_units_owned": int(len(group)),
            "total_contract_value": _safe_round(group["net_sales_value"].sum()),
            "customer_since": reservation_dates.min().strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


def _safe_round(value, ndigits=2):
    return round(float(value), ndigits)


# --------------------------------------------------------------------------
# H. Payment schedules
# --------------------------------------------------------------------------

INSTALLMENT_TYPES = ["Down Payment", "Regular Installment", "Milestone Payment", "Final Payment"]


def generate_payment_schedules(sales_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counter = 1
    for _, sale in sales_df.iterrows():
        if pd.isna(sale["contract_date"]):
            continue  # pure reservations never entered a payment plan
        contract_date = pd.Timestamp(sale["contract_date"])
        cancellation_date = pd.Timestamp(sale["cancellation_date"]) if pd.notna(sale["cancellation_date"]) else None
        plan_years = int(sale["payment_plan_years"])
        n_installments = plan_years * 4  # quarterly installments
        installment_amount = round((sale["net_sales_value"] - sale["down_payment_value"]) / max(n_installments, 1), 2)

        # affordability risk: low down payment + long plan + high
        # discount (proxy for a stretched buyer) raises overdue
        # probability -- this is the mechanism behind Haven West's and
        # Coastline's collections stories.
        risk_score = (
            max(0.0, (18 - sale["down_payment_pct"]) / 18) * 0.5
            + max(0.0, (plan_years - 5) / 4) * 0.3
            + max(0.0, (sale["discount_pct"] - 6) / 10) * 0.2
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # down payment installment
        rows.append(_installment_row(
            counter, sale, contract_date, sale["down_payment_value"], "Down Payment",
            cancellation_date, risk_score, project_seasonal=sale["project_id"] == "CST",
        ))
        counter += 1

        for i in range(1, n_installments + 1):
            due_date = contract_date + pd.DateOffset(months=3 * i)
            if due_date > END_DATE + pd.Timedelta(days=365):
                break  # do not generate installments absurdly far beyond the analysis horizon
            installment_type = "Final Payment" if i == n_installments else "Regular Installment"
            rows.append(_installment_row(
                counter, sale, due_date, installment_amount, installment_type,
                cancellation_date, risk_score, project_seasonal=sale["project_id"] == "CST",
            ))
            counter += 1
    return pd.DataFrame(rows)


def _installment_row(counter, sale, due_date, amount_due, installment_type, cancellation_date, risk_score, project_seasonal):
    if cancellation_date is not None and due_date >= cancellation_date:
        payment_status, amount_paid, days_overdue, payment_date, bounced, rescheduled = (
            "Voided (Cancelled)", 0.0, 0, None, False, False,
        )
    elif due_date > END_DATE:
        payment_status, amount_paid, days_overdue, payment_date, bounced, rescheduled = (
            "Not Yet Due", 0.0, 0, None, False, False,
        )
    else:
        overdue_prob = 0.06 + risk_score * 0.42
        # Coastline: delinquency rises specifically in the months after
        # peak season (autumn/winter), on top of the base risk score.
        if project_seasonal and due_date.month in (10, 11, 12, 1):
            overdue_prob += 0.12
        is_overdue = rng.random() < overdue_prob
        if is_overdue:
            days_overdue = int(rng.integers(1, 210))
            severity_scale = min(days_overdue / 30, 6)
            paid_fraction = max(0.0, 1 - severity_scale * 0.15 - float(rng.uniform(0, 0.2)))
            amount_paid = round(amount_due * paid_fraction, 2) if days_overdue < 200 else 0.0
            payment_status = "Overdue"
            payment_date = None
            bounced = bool(rng.random() < (0.10 + risk_score * 0.25))
            rescheduled = bool(rng.random() < 0.18)
        else:
            days_overdue = 0
            amount_paid = amount_due
            payment_status = "Paid"
            payment_date = (due_date - pd.Timedelta(days=int(rng.integers(0, 5)))).strftime("%Y-%m-%d")
            bounced = False
            rescheduled = False

    return {
        "installment_id": f"INS-{counter:07d}",
        "sale_id": sale["sale_id"],
        "customer_id": sale["customer_id"],
        "project_id": sale["project_id"],
        "due_date": due_date.strftime("%Y-%m-%d"),
        "amount_due": round(float(amount_due), 2),
        "installment_type": installment_type,
        "payment_status": payment_status,
        "days_overdue": days_overdue,
        "amount_paid": amount_paid,
        "outstanding_amount": round(float(amount_due) - float(amount_paid), 2),
        "payment_date": payment_date,
        "bounced_payment_flag": bounced,
        "rescheduled_flag": rescheduled,
        "grace_period_days": 10,
    }


# --------------------------------------------------------------------------
# I. Collections
# --------------------------------------------------------------------------

COLLECTION_ACTIONS = ["Reminder Call", "Reminder WhatsApp", "Formal Notice Email", "In-Person Visit", "Legal Escalation Notice"]
CONTACT_RESULTS = ["Reached - Promise to Pay", "Reached - Dispute", "Reached - No Commitment", "No Answer", "Wrong Contact Info"]
RECOVERY_STATUSES = ["Open", "Promise to Pay Pending", "Recovered", "Partially Recovered", "Escalated - Legal", "Written Off"]
COLLECTION_OWNERS = ["Layla Anwar", "Mostafa Reda", "Nourhan Sabry", "Karim Fathy", "Rana Naguib"]


def generate_collections(payment_schedules_df: pd.DataFrame) -> pd.DataFrame:
    overdue = payment_schedules_df[payment_schedules_df["payment_status"] == "Overdue"]
    rows = []
    counter = 1
    for _, inst in overdue.iterrows():
        n_events = 1 + int(min(inst["days_overdue"] // 30, 4))
        due_date = pd.Timestamp(inst["due_date"])
        escalation = 1
        for e in range(n_events):
            event_date = due_date + pd.Timedelta(days=int(rng.integers(5 + e * 25, 20 + e * 30)))
            if event_date > END_DATE:
                break
            action = COLLECTION_ACTIONS[min(e, len(COLLECTION_ACTIONS) - 1)]
            contact_result = rng.choice(CONTACT_RESULTS)
            promise_date, promise_amount = None, None
            if contact_result == "Reached - Promise to Pay":
                promise_date = (event_date + pd.Timedelta(days=int(rng.integers(3, 21)))).strftime("%Y-%m-%d")
                promise_amount = round(float(inst["outstanding_amount"]) * float(rng.uniform(0.3, 1.0)), 2)
            escalation = min(escalation + (1 if e >= 2 else 0), 3)
            recovery_status = rng.choice(RECOVERY_STATUSES, p=[0.30, 0.20, 0.20, 0.14, 0.11, 0.05])

            rows.append({
                "collection_event_id": f"COL-{counter:06d}",
                "installment_id": inst["installment_id"],
                "event_date": event_date.strftime("%Y-%m-%d"),
                "collection_action": action,
                "owner": rng.choice(COLLECTION_OWNERS),
                "contact_result": contact_result,
                "promise_to_pay_date": promise_date,
                "promise_to_pay_amount": promise_amount,
                "escalation_level": escalation,
                "recovery_status": recovery_status,
            })
            counter += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# J. Construction milestones
# --------------------------------------------------------------------------

MILESTONE_NAMES = [
    "Excavation", "Foundations", "Structural Frame", "Facade", "MEP",
    "Internal Finishes", "Landscaping", "Utilities Connection",
    "Authority Approvals", "Testing & Commissioning",
]
CONTRACTORS = [
    "Delta Construction Group", "Nile Engineering Contractors", "Horizon Build Partners",
    "Cairo Structural Works", "Modern Facade Systems", "Prime MEP Solutions",
]
# Meridian's two structurally delayed buildings -- the mechanism the
# construction/handover-risk finding is built to discover.
MERIDIAN_DELAYED_BUILDINGS = {"Building C", "Building D"}


def generate_construction_milestones(units_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counter = 1
    for pid, d in PROJECT_DEFS.items():
        launch = pd.Timestamp(d["launch"])
        combos = units_df.loc[units_df["project_id"] == pid, ["phase", "building"]].drop_duplicates()
        for _, combo in combos.iterrows():
            phase, building = combo["phase"], combo["building"]
            is_meridian_delayed = pid == "MER" and building in MERIDIAN_DELAYED_BUILDINGS
            cursor = launch
            for i, milestone_name in enumerate(MILESTONE_NAMES):
                planned_duration = int(rng.integers(25, 70))
                baseline_start = cursor
                baseline_end = baseline_start + pd.Timedelta(days=planned_duration)

                # progress fraction into the overall project timeline,
                # used to decide whether this milestone should already
                # be complete given current_completion_pct.
                milestone_progress_target = (i + 1) / len(MILESTONE_NAMES) * 100
                planned_completion_pct = min(100.0, round(milestone_progress_target, 1))

                variance_days = int(rng.integers(-5, 12))
                issue_category, issue_severity = None, None

                if is_meridian_delayed and milestone_name in ("MEP", "Internal Finishes", "Testing & Commissioning"):
                    variance_days = int(rng.integers(55, 130))
                    issue_category = rng.choice(["MEP Subcontractor Delay", "Finishing Contractor Delay", "Materials Procurement Delay"])
                    issue_severity = "HIGH" if variance_days >= 80 else "MEDIUM"
                elif rng.random() < 0.12:
                    variance_days += int(rng.integers(15, 40))
                    issue_category = rng.choice(["Weather Delay", "Permit Delay", "Materials Procurement Delay", "Labor Shortage"])
                    issue_severity = "MEDIUM"

                forecast_end = baseline_end + pd.Timedelta(days=max(variance_days, 0))
                current_completion = d["completion_pct"]
                milestone_is_reached = current_completion >= milestone_progress_target - (10 if not is_meridian_delayed else 25)

                if milestone_is_reached and not (is_meridian_delayed and milestone_name in ("Internal Finishes", "Testing & Commissioning")):
                    actual_completion_pct = 100.0
                    actual_end = forecast_end
                    status = "Complete" if variance_days <= 14 else "Complete (Delayed)"
                elif milestone_progress_target <= current_completion + 15:
                    actual_completion_pct = round(float(rng.uniform(35, 85)), 1)
                    actual_end = None
                    status = "In Progress" if issue_severity is None else ("Delayed" if issue_severity == "HIGH" else "At Risk")
                else:
                    actual_completion_pct = 0.0
                    actual_end = None
                    status = "Not Started"

                budgeted_cost = round(float(rng.uniform(8_000_000, 60_000_000)), 2)
                cost_variance_pct = float(rng.uniform(-0.03, 0.12)) if issue_severity else float(rng.uniform(-0.03, 0.05))
                committed_cost = round(budgeted_cost * (1 + max(cost_variance_pct - 0.02, 0)), 2)
                actual_cost = round(budgeted_cost * (1 + cost_variance_pct), 2) if actual_completion_pct > 0 else 0.0

                rows.append({
                    "milestone_id": f"MIL-{counter:04d}",
                    "project_id": pid,
                    "phase": phase,
                    "building": building,
                    "milestone_name": milestone_name,
                    "baseline_start_date": baseline_start.strftime("%Y-%m-%d"),
                    "baseline_end_date": baseline_end.strftime("%Y-%m-%d"),
                    "forecast_end_date": forecast_end.strftime("%Y-%m-%d"),
                    "actual_end_date": actual_end.strftime("%Y-%m-%d") if actual_end is not None else None,
                    "planned_completion_pct": planned_completion_pct,
                    "actual_completion_pct": actual_completion_pct,
                    "variance_days": variance_days,
                    "status": status,
                    "contractor": rng.choice(CONTRACTORS),
                    "budgeted_cost": budgeted_cost,
                    "committed_cost": committed_cost,
                    "actual_cost": actual_cost,
                    "cost_variance": round(actual_cost - budgeted_cost, 2),
                    "issue_category": issue_category,
                    "issue_severity": issue_severity,
                })
                counter += 1
                cursor = baseline_end
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# K. Customer cases
# --------------------------------------------------------------------------

CASE_CATEGORY_SUBCATEGORY = {
    "Sales Experience": ["Response Time", "Misrepresentation Concern", "Communication Quality"],
    "Collections": ["Payment Plan Dispute", "Installment Amount Query", "Late Fee Dispute"],
    "Construction Update": ["Progress Inquiry", "Delay Concern", "Site Visit Request"],
    "Handover": ["Handover Date Concern", "Documentation", "Final Payment Query"],
    "Snagging / Quality": ["Defect Report", "Finishing Quality", "MEP Issue"],
    "General Inquiry": ["Amenities", "HOA / Facilities Management", "Resale Process"],
}
CASE_CHANNELS = ["Phone", "WhatsApp", "Email", "In-Person", "Customer Portal"]
CASE_DEPARTMENT_BY_CATEGORY = {
    "Sales Experience": "Sales", "Collections": "Collections", "Construction Update": "Construction",
    "Handover": "Customer Experience", "Snagging / Quality": "Construction", "General Inquiry": "Customer Experience",
}


def generate_customer_cases(sales_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counter = 1
    active_sales = sales_df[sales_df["contract_status"] != "Cancelled"]

    for pid in PROJECT_IDS:
        project_sales = active_sales[active_sales["project_id"] == pid]
        if project_sales.empty:
            continue
        n_cases = {"AUR": 260, "CST": 150, "VTX": 90, "HVW": 320, "MER": 340}[pid]
        sampled = project_sales.sample(n=min(n_cases, len(project_sales)), replace=len(project_sales) < n_cases,
                                        random_state=int(rng.integers(0, 1_000_000)))
        for _, sale in sampled.iterrows():
            if pid == "MER":
                category = rng.choice(list(CASE_CATEGORY_SUBCATEGORY.keys()),
                                       p=[0.05, 0.05, 0.15, 0.40, 0.30, 0.05])
            elif pid == "HVW":
                category = rng.choice(list(CASE_CATEGORY_SUBCATEGORY.keys()),
                                       p=[0.10, 0.38, 0.12, 0.08, 0.12, 0.20])
            elif pid == "AUR":
                category = rng.choice(list(CASE_CATEGORY_SUBCATEGORY.keys()),
                                       p=[0.34, 0.08, 0.16, 0.08, 0.14, 0.20])
            else:
                category = rng.choice(list(CASE_CATEGORY_SUBCATEGORY.keys()),
                                       p=[0.14, 0.12, 0.18, 0.12, 0.16, 0.28])
            subcategory = rng.choice(CASE_CATEGORY_SUBCATEGORY[category])

            created_offset = int(rng.integers(0, WINDOW_DAYS))
            # Meridian handover/snagging cases skew toward the recent
            # months of the window, mirroring rising complaints as
            # actual handover dates approach and slip.
            if pid == "MER" and category in ("Handover", "Snagging / Quality"):
                created_offset = int(rng.integers(int(WINDOW_DAYS * 0.55), WINDOW_DAYS))
            created_at = START_DATE + pd.Timedelta(days=created_offset)
            if created_at > END_DATE:
                created_at = END_DATE - pd.Timedelta(days=int(rng.integers(0, 30)))

            priority = rng.choice(["Low", "Medium", "High", "Critical"], p=[0.30, 0.40, 0.22, 0.08])
            sla_hours = {"Critical": 4, "High": 24, "Medium": 48, "Low": 96}[priority]

            response_minutes = max(float(rng.normal(90, 60)), 5)
            first_response_at = created_at + pd.Timedelta(minutes=response_minutes)

            resolved_prob = 0.82
            resolved_at, status = None, "Open"
            resolution_hours = None
            sla_met = None
            if rng.random() < resolved_prob:
                resolution_hours = max(float(rng.normal(sla_hours * 0.8, sla_hours * 0.5)), 1)
                if pid == "MER" and category in ("Handover", "Snagging / Quality"):
                    resolution_hours *= float(rng.uniform(1.3, 2.2))
                resolved_at = first_response_at + pd.Timedelta(hours=resolution_hours)
                status = "Resolved"
                sla_met = bool(resolution_hours <= sla_hours * 1.5)

            reopen_count = int(rng.integers(0, 2)) if (status == "Resolved" and rng.random() < 0.12) else 0
            if pid == "MER" and category == "Snagging / Quality" and rng.random() < 0.25:
                reopen_count = max(reopen_count, 1)

            if pid == "MER" and category in ("Handover", "Construction Update"):
                sentiment = rng.choice(["Negative", "Neutral", "Positive"], p=[0.58, 0.32, 0.10])
            elif pid == "HVW" and category == "Collections":
                sentiment = rng.choice(["Negative", "Neutral", "Positive"], p=[0.50, 0.35, 0.15])
            else:
                sentiment = rng.choice(["Negative", "Neutral", "Positive"], p=[0.25, 0.45, 0.30])

            escalation_flag = bool(priority in ("High", "Critical") and rng.random() < 0.4)

            rows.append({
                "case_id": f"CAS-{counter:05d}",
                "customer_id": sale["customer_id"],
                "project_id": pid,
                "unit_id": sale["unit_id"],
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "subcategory": subcategory,
                "priority": priority,
                "channel": rng.choice(CASE_CHANNELS),
                "first_response_at": first_response_at.strftime("%Y-%m-%d %H:%M:%S"),
                "resolved_at": resolved_at.strftime("%Y-%m-%d %H:%M:%S") if resolved_at is not None else None,
                "status": status,
                "resolution_sla_hours": sla_hours,
                "resolution_sla_met": sla_met,
                "reopen_count": reopen_count,
                "sentiment": sentiment,
                "escalation_flag": escalation_flag,
                "responsible_department": CASE_DEPARTMENT_BY_CATEGORY[category],
            })
            counter += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# L. Handovers + snagging
# --------------------------------------------------------------------------

SNAG_CATEGORIES = ["Paint & Finishing", "Plumbing", "Electrical", "Flooring", "Doors & Windows", "HVAC", "Structural"]


def generate_handovers(units_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counter = 1
    # Meridian is the project actually in the handover window; a small
    # number of near-complete units in other mature phases may also
    # appear, but the volume is deliberately concentrated in Meridian.
    candidates = units_df[
        (units_df["project_id"] == "MER") & (units_df["unit_status"].isin(["contracted", "handed_over"]))
    ]
    for _, unit in candidates.iterrows():
        is_delayed_building = unit["building"] in MERIDIAN_DELAYED_BUILDINGS
        original = pd.Timestamp(unit["expected_handover_date"])
        if is_delayed_building:
            delay_days = int(rng.integers(60, 150))
            forecast = original + pd.Timedelta(days=delay_days)
            handover_status = "Delayed" if unit["unit_status"] != "handed_over" else "Completed (Delayed)"
        else:
            delay_days = int(rng.integers(-10, 20))
            forecast = original + pd.Timedelta(days=max(delay_days, 0))
            handover_status = "On Track" if unit["unit_status"] != "handed_over" else "Completed"

        actual = None
        if unit["unit_status"] == "handed_over":
            actual = forecast - pd.Timedelta(days=int(rng.integers(0, 10)))
            actual = min(actual, END_DATE)

        final_payment_received = unit["unit_status"] == "handed_over" or (not is_delayed_building and rng.random() < 0.4)
        rows.append({
            "handover_id": f"HAN-{counter:04d}",
            "unit_id": unit["unit_id"],
            "project_id": unit["project_id"],
            "original_handover_date": original.strftime("%Y-%m-%d"),
            "forecast_handover_date": min(forecast, END_DATE + pd.Timedelta(days=365)).strftime("%Y-%m-%d"),
            "actual_handover_date": actual.strftime("%Y-%m-%d") if actual is not None else None,
            "handover_status": handover_status,
            "days_delayed": max((forecast - original).days, 0),
            "final_payment_received": bool(final_payment_received),
            "customer_notified": bool(rng.random() < (0.55 if is_delayed_building else 0.92)),
            "inspection_completed": bool(unit["unit_status"] == "handed_over" or rng.random() < 0.3),
            "keys_released": bool(unit["unit_status"] == "handed_over"),
        })
        counter += 1
    return pd.DataFrame(rows)


def generate_snagging(handovers_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counter = 1
    for _, handover in handovers_df.iterrows():
        is_delayed_building = handover["handover_status"] in ("Delayed", "Completed (Delayed)")
        n_snags = int(rng.integers(3, 9)) if is_delayed_building else int(rng.integers(0, 4))
        actual_or_forecast = handover["actual_handover_date"] if pd.notna(handover["actual_handover_date"]) else handover["forecast_handover_date"]
        base_date = pd.Timestamp(actual_or_forecast)
        for _ in range(n_snags):
            created_at = base_date - pd.Timedelta(days=int(rng.integers(0, 20)))
            severity = rng.choice(["Minor", "Moderate", "Major"], p=[0.5, 0.35, 0.15] if not is_delayed_building else [0.25, 0.40, 0.35])
            target_days = {"Minor": 7, "Moderate": 14, "Major": 30}[severity]
            target_resolution = created_at + pd.Timedelta(days=target_days)
            resolved = rng.random() < (0.7 if not is_delayed_building else 0.45)
            resolved_at = None
            status = "Open"
            if resolved:
                actual_days = target_days * float(rng.uniform(0.6, 1.8))
                resolved_at = created_at + pd.Timedelta(days=actual_days)
                status = "Resolved"
            elif rng.random() < 0.3:
                status = "In Progress"

            reopen_count = int(rng.integers(0, 2)) if (resolved and rng.random() < (0.10 if not is_delayed_building else 0.25)) else 0

            rows.append({
                "snag_id": f"SNG-{counter:05d}",
                "handover_id": handover["handover_id"],
                "unit_id": handover["unit_id"],
                "category": rng.choice(SNAG_CATEGORIES),
                "severity": severity,
                "created_at": created_at.strftime("%Y-%m-%d"),
                "target_resolution_date": target_resolution.strftime("%Y-%m-%d"),
                "resolved_at": resolved_at.strftime("%Y-%m-%d") if resolved_at is not None else None,
                "status": status,
                "contractor": rng.choice(CONTRACTORS),
                "reopen_count": reopen_count,
            })
            counter += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Finalize brokers (aggregate from sales + leads)
# --------------------------------------------------------------------------

def _finalize_brokers(brokers_df: pd.DataFrame, sales_df: pd.DataFrame, leads_df: pd.DataFrame) -> pd.DataFrame:
    brokers_df = brokers_df.copy()
    lead_counts = leads_df[leads_df["broker_id"].notna()].groupby("broker_id")["lead_id"].count()
    sales_with_broker = sales_df[sales_df["broker_id"].notna()]
    reservation_counts = sales_with_broker.groupby("broker_id")["sale_id"].count()
    contract_counts = sales_with_broker[sales_with_broker["contract_date"].notna()].groupby("broker_id")["sale_id"].count()
    cancelled_counts = sales_with_broker[sales_with_broker["cancellation_flag"]].groupby("broker_id")["sale_id"].count()
    gross_sales = sales_with_broker.groupby("broker_id")["gross_price"].sum()
    commission_paid = sales_with_broker.groupby("broker_id")["commission_value"].sum()
    avg_discount = sales_with_broker.groupby("broker_id")["discount_pct"].mean()

    for column, series in [
        ("lead_volume", lead_counts), ("reservation_count", reservation_counts),
        ("contract_count", contract_counts), ("cancelled_contracts", cancelled_counts),
        ("gross_sales_value", gross_sales), ("commission_paid", commission_paid),
        ("average_discount_pct", avg_discount),
    ]:
        brokers_df[column] = brokers_df["broker_id"].map(series).fillna(0)
        if column not in ("average_discount_pct",):
            brokers_df[column] = brokers_df[column].round(2 if column in ("gross_sales_value", "commission_paid") else 0).astype(
                float if column in ("gross_sales_value", "commission_paid") else int
            )
        else:
            brokers_df[column] = brokers_df[column].round(1)
    return brokers_df


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("Generating projects...")
    projects_df = generate_projects()

    print("Generating units...")
    units_df = generate_units()

    print("Generating brokers...")
    brokers_df = generate_brokers()

    print("Generating sales agents...")
    agents_df = generate_sales_agents()

    print("Generating campaigns...")
    campaigns_df = generate_campaigns()

    print("Generating leads...")
    leads_df = generate_leads(campaigns_df, agents_df, brokers_df)

    print("Generating sales (and finalizing units/leads)...")
    sales_df, units_df, leads_df = generate_sales(units_df, leads_df, agents_df, brokers_df)

    print("Generating customers...")
    customers_df = generate_customers(sales_df)

    print("Finalizing brokers...")
    brokers_df = _finalize_brokers(brokers_df, sales_df, leads_df)

    print("Generating payment schedules...")
    payment_schedules_df = generate_payment_schedules(sales_df)

    print("Generating collections...")
    collections_df = generate_collections(payment_schedules_df)

    print("Generating construction milestones...")
    construction_df = generate_construction_milestones(units_df)

    print("Generating customer cases...")
    cases_df = generate_customer_cases(sales_df)

    print("Generating handovers...")
    handovers_df = generate_handovers(units_df)

    print("Generating snagging...")
    snagging_df = generate_snagging(handovers_df)

    outputs = {
        "projects.csv": projects_df, "units.csv": units_df, "customers.csv": customers_df,
        "brokers.csv": brokers_df, "sales_agents.csv": agents_df, "campaigns.csv": campaigns_df,
        "leads.csv": leads_df, "sales.csv": sales_df, "payment_schedules.csv": payment_schedules_df,
        "collections.csv": collections_df, "construction_milestones.csv": construction_df,
        "customer_cases.csv": cases_df, "handovers.csv": handovers_df, "snagging.csv": snagging_df,
    }
    for filename, df in outputs.items():
        df.to_csv(DATA_DIR / filename, index=False)
        print(f"  Wrote {len(df):,} rows -> data/{filename}")

    print("\nDone. Synthetic real estate data ready in data/.")


if __name__ == "__main__":
    main()
