"""Synthetic business data generator for the Solven Operations
Intelligence Platform.

Generates three realistic, internally-consistent datasets that mirror
what a medium-sized logistics company would actually export from its
ERP/TMS/WMS systems:

    - data/shipments.csv   (~5,000 rows) -- TMS shipment/delivery records
    - data/complaints.csv  (~500 rows)   -- CRM customer complaint tickets
    - data/inventory.csv   (~300 rows)   -- WMS warehouse stock records

The generated data intentionally encodes a realistic operational
problem: the "Cairo East" warehouse becomes overloaded during July,
causing delivery times to increase, delay rates to spike, driver
ratings to drop, loading times to lengthen, and customer complaints
to rise. Inventory is also tightest at Cairo East. Every other field
(driver, vehicle, route, priority, customer tier, weight, fuel cost,
etc.) is generated to be causally consistent with this story rather
than sampled independently, so the pattern is discoverable by the
analytics layer (analysis/) rather than injected as isolated noise.

Run directly to (re)generate all CSV files:

    python generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 42
N_SHIPMENTS = 5_000
N_COMPLAINTS = 500

DATA_DIR = Path(__file__).parent / "data"

START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2025-12-31")

WAREHOUSES = ["Cairo East", "Cairo West", "Alexandria", "Mansoura"]
WAREHOUSE_CODES = {
    "Cairo East": "CAE", "Cairo West": "CAW", "Alexandria": "ALX", "Mansoura": "MNS",
}

REGIONS = ["Cairo", "Giza", "Alexandria", "Delta", "Upper Egypt", "Suez"]
REGION_CODES = {
    "Cairo": "CAI", "Giza": "GIZ", "Alexandria": "ALXR", "Delta": "DLT",
    "Upper Egypt": "UPE", "Suez": "SUZ",
}
REGION_WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]

# Which warehouses typically serve each region, and how often.
REGION_WAREHOUSE_MAP: dict[str, tuple[list[str], list[float]]] = {
    "Cairo": (["Cairo East", "Cairo West"], [0.6, 0.4]),
    "Giza": (["Cairo West", "Cairo East"], [0.7, 0.3]),
    "Alexandria": (["Alexandria", "Cairo West"], [0.85, 0.15]),
    "Delta": (["Mansoura", "Cairo East"], [0.6, 0.4]),
    "Upper Egypt": (["Mansoura", "Cairo East"], [0.5, 0.5]),
    "Suez": (["Cairo East", "Cairo West"], [0.75, 0.25]),
}

# Baseline (min, max) actual transit days by region under normal load.
REGION_DELIVERY_DAYS: dict[str, tuple[int, int]] = {
    "Cairo": (1, 2),
    "Giza": (1, 3),
    "Alexandria": (2, 4),
    "Delta": (2, 4),
    "Upper Egypt": (3, 6),
    "Suez": (2, 4),
}

# Standard SLA commitment (days) communicated to the customer at
# booking time, before the priority-tier adjustment is applied.
REGION_BASE_PROMISE_DAYS: dict[str, int] = {
    "Cairo": 2, "Giza": 3, "Alexandria": 4, "Delta": 4, "Upper Egypt": 6, "Suez": 4,
}

# Road distance (km) from the regional hub, used to scale fuel cost.
REGION_DISTANCE_KM: dict[str, int] = {
    "Cairo": 15, "Giza": 25, "Alexandria": 220, "Delta": 120,
    "Upper Egypt": 650, "Suez": 140,
}

COMPLAINT_CATEGORIES = [
    "Late Delivery",
    "Damaged Package",
    "Wrong Item",
    "Poor Communication",
    "Lost Shipment",
]
SENTIMENTS = ["Negative", "Neutral", "Positive"]

# The injected operational problem: this warehouse becomes overloaded
# during this month, degrading delivery performance, driver ratings,
# fuel efficiency, and dock throughput.
PROBLEM_WAREHOUSE = "Cairo East"
PROBLEM_MONTH = 7
OVERLOAD_EXTRA_DAYS_RANGE = (3, 7)
OVERLOAD_COST_MULTIPLIER_RANGE = (1.15, 1.35)
OVERLOAD_FUEL_MULTIPLIER_RANGE = (1.10, 1.25)
OVERLOAD_LOADING_DELAY_MINUTES_RANGE = (10, 20)
OVERLOAD_INCIDENT_RATE_MULTIPLIER = 2.2
OVERLOAD_RATING_PENALTY_RANGE = (0.3, 0.6)

# Warehouses that intentionally run leaner (lower) inventory buffers,
# expressed as the probability that a given product is understocked
# relative to its reorder threshold. Cairo East is deliberately the
# weakest, echoing the same overload story told by the shipment data.
WAREHOUSE_LOW_STOCK_RATE: dict[str, float] = {
    "Cairo East": 0.35,
    "Cairo West": 0.15,
    "Alexandria": 0.10,
    "Mansoura": 0.12,
}

# Physical storage capacity (units) per warehouse. Cairo East is
# undersized relative to the demand it serves, which is the root
# cause of its July overload rather than a coincidence of the data.
WAREHOUSE_CAPACITY_UNITS: dict[str, int] = {
    "Cairo East": 45_000,
    "Cairo West": 60_000,
    "Alexandria": 50_000,
    "Mansoura": 40_000,
}

# --- Customer segmentation -------------------------------------------------

CUSTOMER_TYPE_WEIGHTS: dict[str, float] = {"Retail": 0.50, "SME": 0.35, "Enterprise": 0.15}

# Probability that a shipment for this customer tier ends up Lost or
# Cancelled. Enterprise accounts get dedicated account management and
# white-glove handling, so incidents are rarer.
CUSTOMER_INCIDENT_RATE: dict[str, float] = {"Enterprise": 0.02, "SME": 0.035, "Retail": 0.05}

# Gamma-distribution scale (kg) controlling typical shipment weight
# per customer tier. Enterprise accounts ship in bulk.
CUSTOMER_WEIGHT_SCALE_KG: dict[str, float] = {"Retail": 8.0, "SME": 20.0, "Enterprise": 55.0}
WEIGHT_GAMMA_SHAPE = 2.0

# Fraction of the raw weight-based transit penalty that actually
# lands on each tier. Enterprise bulk freight moves on dedicated
# capacity and is largely shielded from it; Retail bears it in full.
CUSTOMER_WEIGHT_PENALTY_FACTOR: dict[str, float] = {"Enterprise": 0.3, "SME": 0.7, "Retail": 1.0}

# Priority mix per customer tier: Enterprise accounts more frequently
# pay for Express/Critical handling.
PRIORITY_WEIGHTS_BY_CUSTOMER: dict[str, tuple[list[str], list[float]]] = {
    "Enterprise": (["Standard", "Express", "Critical"], [0.40, 0.35, 0.25]),
    "SME": (["Standard", "Express", "Critical"], [0.60, 0.28, 0.12]),
    "Retail": (["Standard", "Express", "Critical"], [0.75, 0.20, 0.05]),
}

# Days added to (positive) or subtracted from (negative) the region's
# base SLA promise depending on the purchased priority tier.
PRIORITY_PROMISE_ADJUSTMENT_DAYS: dict[str, int] = {"Critical": -1, "Express": 0, "Standard": 1}
MIN_PROMISE_DAYS = 1

DELIVERY_WINDOWS_BY_PRIORITY: dict[str, list[str]] = {
    "Standard": ["09:00-12:00", "12:00-15:00", "15:00-18:00", "18:00-21:00"],
    "Express": ["08:00-11:00", "13:00-16:00"],
    "Critical": ["08:00-10:00", "10:00-12:00", "14:00-16:00"],
}

# --- Fleet & routing ---------------------------------------------------

VEHICLE_TYPES = ["Motorcycle", "Van", "Small Truck", "Large Truck"]
# Upper weight bound (kg) for each vehicle type, in ascending order;
# the last type is the catch-all for anything heavier.
VEHICLE_WEIGHT_UPPER_BOUND_KG = [5, 50, 200]
VEHICLES_PER_TYPE_PER_WAREHOUSE: dict[str, int] = {
    "Motorcycle": 6, "Van": 8, "Small Truck": 5, "Large Truck": 3,
}
LOADING_TIME_RANGE_BY_VEHICLE: dict[str, tuple[int, int]] = {
    "Motorcycle": (3, 8), "Van": (8, 15), "Small Truck": (15, 25), "Large Truck": (25, 40),
}
UNLOADING_TIME_RANGE_BY_VEHICLE: dict[str, tuple[int, int]] = {
    "Motorcycle": (3, 6), "Van": (6, 12), "Small Truck": (12, 20), "Large Truck": (20, 35),
}

DRIVERS_PER_WAREHOUSE = 15
ROUTES_PER_WAREHOUSE_REGION_PAIR = 3

FUEL_RATE_PER_KM = 0.9  # currency units per km, baseline light-vehicle rate
WEIGHT_FUEL_FACTOR_DIVISOR = 400  # heavier loads burn proportionally more fuel

DRIVER_FIRST_NAMES = [
    "Ahmed", "Mohamed", "Mahmoud", "Youssef", "Omar", "Karim", "Hassan",
    "Sherif", "Tarek", "Amr", "Khaled", "Sameh", "Ibrahim", "Wael",
    "Adel", "Hany", "Ashraf", "Fady", "Ramy", "Emad",
]
DRIVER_LAST_NAMES = [
    "Abdallah", "El-Sayed", "Farouk", "Hussein", "Mostafa", "Kamal",
    "Zaki", "Fahmy", "Rashad", "Naguib", "El-Masry", "Shawky", "Gomaa",
    "Salem", "Aziz", "Nour", "Fathy", "Reda", "Anwar", "Mansour",
]

# --- Complaint resolution workflow ------------------------------------

CATEGORY_ASSIGNED_TEAM: dict[str, str] = {
    "Late Delivery": "Logistics Support",
    "Damaged Package": "Claims & Damage",
    "Wrong Item": "Fulfillment QA",
    "Poor Communication": "Customer Success",
    "Lost Shipment": "Claims & Damage",
}
CATEGORY_BASE_ESCALATION_LEVEL: dict[str, int] = {
    "Late Delivery": 1, "Damaged Package": 2, "Wrong Item": 1,
    "Poor Communication": 1, "Lost Shipment": 3,
}
CATEGORY_BASE_RESOLUTION_HOURS: dict[str, float] = {
    "Late Delivery": 36, "Damaged Package": 48, "Wrong Item": 24,
    "Poor Communication": 12, "Lost Shipment": 96,
}
CATEGORY_RESOLUTION_RATE: dict[str, float] = {
    "Late Delivery": 0.92, "Damaged Package": 0.88, "Wrong Item": 0.90,
    "Poor Communication": 0.95, "Lost Shipment": 0.72,
}
MAX_ESCALATION_LEVEL = 3
SATISFACTION_HOURS_DIVISOR = 40  # slower resolutions erode satisfaction

# --- Inventory / supply chain -------------------------------------------

SUPPLIER_BASE_LEAD_TIME_DAYS: dict[str, int] = {
    "Nile Manufacturing Co.": 7,
    "Delta Industrial Supply": 10,
    "Cairo Wholesale Partners": 5,
    "Red Sea Import Group": 14,
    "Alexandria Trading House": 9,
    "Mediterranean Sourcing Ltd.": 12,
    "Upper Egypt Producers": 11,
    "Sinai Logistics Supply": 15,
}
UNIT_COST_RANGE = (5.0, 150.0)

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# Generic vectorized helpers
# --------------------------------------------------------------------------

def _grouped_choice(
    group_values: np.ndarray,
    options_by_group: dict[str, list],
    weights_by_group: dict[str, list] | None = None,
) -> np.ndarray:
    """Apply ``rng.choice`` independently within each group.

    For every distinct value found in ``group_values``, samples that
    many entries from the matching entry in ``options_by_group``
    (optionally weighted). This avoids a per-row Python loop when a
    categorical attribute depends on another column, e.g. sampling a
    driver from the roster of the shipment's own warehouse.

    Args:
        group_values: Array of group labels, one per output row.
        options_by_group: Mapping of group label to the list of
            choices available to that group.
        weights_by_group: Optional mapping of group label to sampling
            probabilities for that group's choices.

    Returns:
        An object array the same length as ``group_values``.
    """
    result = np.empty(len(group_values), dtype=object)
    for group, options in options_by_group.items():
        mask = group_values == group
        count = int(mask.sum())
        if count:
            weights = weights_by_group[group] if weights_by_group else None
            result[mask] = rng.choice(options, size=count, p=weights)
    return result


def _grouped_randint(
    group_values: np.ndarray, range_by_group: dict[str, tuple[int, int]]
) -> np.ndarray:
    """Apply an inclusive ``rng.integers`` draw independently within each group.

    Args:
        group_values: Array of group labels, one per output row.
        range_by_group: Mapping of group label to an inclusive
            ``(low, high)`` integer range.

    Returns:
        An int array the same length as ``group_values``.
    """
    result = np.empty(len(group_values), dtype=int)
    for group, (low, high) in range_by_group.items():
        mask = group_values == group
        count = int(mask.sum())
        if count:
            result[mask] = rng.integers(low, high + 1, size=count)
    return result


def _combo_key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build a ``"a|b"`` composite key array for two-column grouping."""
    return (pd.Series(a).astype(str) + "|" + pd.Series(b).astype(str)).to_numpy()


# --------------------------------------------------------------------------
# Shared reference pools
# --------------------------------------------------------------------------

def _generate_customer_pool(size: int = 250) -> list[str]:
    """Build a pool of realistic-looking company customer names.

    Names are generated by combining place-inspired prefixes with
    common business suffixes, then deduplicated. Shipments sample
    from this pool so customers naturally repeat across orders.
    """
    prefixes = [
        "Nile", "Delta", "Pyramids", "Sphinx", "Zamalek", "Maadi",
        "Heliopolis", "Horus", "Al Amal", "Sahara", "Cleopatra",
        "Luxor", "Aswan", "Nasr City", "Al Fustat", "Giza Plaza",
        "Corniche", "Mediterranean", "Red Sea", "Oasis",
    ]
    suffixes = [
        "Trading Co.", "Retail Group", "Logistics", "Supplies Co.",
        "Distribution", "Market", "Stores", "Enterprises",
        "Wholesale", "Import & Export", "Holdings", "Commercial Co.",
    ]
    combos = [f"{p} {s}" for p in prefixes for s in suffixes]
    size = min(size, len(combos))
    return list(rng.choice(combos, size=size, replace=False))


def _assign_customer_types(customer_pool: list[str]) -> dict[str, str]:
    """Assign each customer a fixed tier so a given company is always
    the same tier across all of its shipments."""
    types = list(CUSTOMER_TYPE_WEIGHTS.keys())
    weights = list(CUSTOMER_TYPE_WEIGHTS.values())
    assigned = rng.choice(types, size=len(customer_pool), p=weights)
    return dict(zip(customer_pool, assigned))


def _generate_product_catalog(size: int = 75) -> list[str]:
    """Build a pool of realistic-looking warehouse product names."""
    adjectives = [
        "Wireless", "Portable", "Compact", "Premium", "Standard",
        "Heavy-Duty", "Eco", "Smart", "Rechargeable", "Foldable",
        "Stainless Steel", "Ergonomic", "Industrial", "Adjustable",
        "Multi-Purpose",
    ]
    nouns = [
        "Mouse", "Keyboard", "Speaker", "Desk Lamp", "Office Chair",
        "Monitor Stand", "Power Bank", "Water Bottle", "Backpack",
        "Storage Box", "Printer", "Router", "Headset", "Webcam",
        "Filing Cabinet", "Whiteboard", "Extension Cord",
        "Air Purifier", "Coffee Maker", "Paper Shredder",
    ]
    combos = [f"{a} {n}" for a in adjectives for n in nouns]
    size = min(size, len(combos))
    return list(rng.choice(combos, size=size, replace=False))


def _build_driver_pool() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Build a fixed driver roster per warehouse.

    Drivers are based out of a single warehouse, mirroring how a real
    TMS assigns delivery staff to a home depot.

    Returns:
        A tuple of (driver_ids_by_warehouse, driver_name_by_id).
    """
    driver_ids_by_warehouse: dict[str, list[str]] = {}
    driver_name_by_id: dict[str, str] = {}
    for warehouse in WAREHOUSES:
        code = WAREHOUSE_CODES[warehouse]
        ids = []
        for n in range(1, DRIVERS_PER_WAREHOUSE + 1):
            driver_id = f"DRV-{code}-{n:03d}"
            name = f"{rng.choice(DRIVER_FIRST_NAMES)} {rng.choice(DRIVER_LAST_NAMES)}"
            driver_name_by_id[driver_id] = name
            ids.append(driver_id)
        driver_ids_by_warehouse[warehouse] = ids
    return driver_ids_by_warehouse, driver_name_by_id


def _build_vehicle_fleet() -> dict[str, list[str]]:
    """Build a fixed vehicle fleet per (warehouse, vehicle_type) pair."""
    fleet: dict[str, list[str]] = {}
    for warehouse in WAREHOUSES:
        code = WAREHOUSE_CODES[warehouse]
        for vehicle_type, count in VEHICLES_PER_TYPE_PER_WAREHOUSE.items():
            type_code = vehicle_type.replace(" ", "").upper()[:4]
            key = f"{warehouse}|{vehicle_type}"
            fleet[key] = [f"VEH-{code}-{type_code}-{n:02d}" for n in range(1, count + 1)]
    return fleet


def _build_route_pool() -> dict[str, list[str]]:
    """Build a fixed set of route codes for every serving (warehouse, region) pair."""
    routes: dict[str, list[str]] = {}
    for region, (warehouse_options, _weights) in REGION_WAREHOUSE_MAP.items():
        for warehouse in warehouse_options:
            key = f"{warehouse}|{region}"
            wcode, rcode = WAREHOUSE_CODES[warehouse], REGION_CODES[region]
            routes[key] = [
                f"RT-{wcode}-{rcode}-{n}" for n in range(1, ROUTES_PER_WAREHOUSE_REGION_PAIR + 1)
            ]
    return routes


def _assign_vehicle_type(shipment_weight: np.ndarray) -> np.ndarray:
    """Map shipment weight to the smallest vehicle class that can carry it."""
    conditions = [shipment_weight < bound for bound in VEHICLE_WEIGHT_UPPER_BOUND_KG]
    return np.select(conditions, VEHICLE_TYPES[:-1], default=VEHICLE_TYPES[-1])


# --------------------------------------------------------------------------
# shipments.csv
# --------------------------------------------------------------------------

def generate_shipments(n: int = N_SHIPMENTS) -> pd.DataFrame:
    """Generate synthetic shipment records with an embedded overload event.

    Every field is derived causally rather than sampled independently:
    customer tier drives priority mix, weight, and incident rate;
    priority drives the promised SLA date; weight drives vehicle type,
    fuel cost, and handling time; and shipments routed through
    PROBLEM_WAREHOUSE during PROBLEM_MONTH are uniformly degraded
    across transit time, incident rate, fuel efficiency, dock
    throughput, and driver ratings, simulating a warehouse operating
    beyond its physical capacity.

    Args:
        n: Number of shipment rows to generate.

    Returns:
        A DataFrame with one row per shipment.
    """
    customer_pool = _generate_customer_pool()
    customer_type_by_customer = _assign_customer_types(customer_pool)
    driver_ids_by_warehouse, driver_name_by_id = _build_driver_pool()
    vehicle_fleet = _build_vehicle_fleet()
    route_pool = _build_route_pool()

    shipment_ids = [f"SHP{100000 + i}" for i in range(n)]

    order_offsets = rng.integers(0, (END_DATE - START_DATE).days + 1, size=n)
    order_dates = START_DATE + pd.to_timedelta(order_offsets, unit="D")

    regions = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    warehouses = _grouped_choice(
        regions,
        {r: opts for r, (opts, _w) in REGION_WAREHOUSE_MAP.items()},
        {r: w for r, (_opts, w) in REGION_WAREHOUSE_MAP.items()},
    )

    customers = rng.choice(customer_pool, size=n)
    customer_type = pd.Series(customers).map(customer_type_by_customer).to_numpy()

    priority = _grouped_choice(
        customer_type,
        {c: opts for c, (opts, _w) in PRIORITY_WEIGHTS_BY_CUSTOMER.items()},
        {c: w for c, (_opts, w) in PRIORITY_WEIGHTS_BY_CUSTOMER.items()},
    )

    # Shipment weight: gamma-distributed around a per-tier scale so
    # Enterprise accounts skew toward bulk shipments.
    weight_scale = np.array([CUSTOMER_WEIGHT_SCALE_KG[c] for c in customer_type])
    shipment_weight = rng.gamma(shape=WEIGHT_GAMMA_SHAPE, scale=weight_scale)
    shipment_weight = np.clip(shipment_weight, 0.5, 800.0).round(1)

    vehicle_type = _assign_vehicle_type(shipment_weight)
    driver_id = _grouped_choice(warehouses, driver_ids_by_warehouse)
    driver_name = pd.Series(driver_id).map(driver_name_by_id).to_numpy()
    vehicle_id = _grouped_choice(_combo_key(warehouses, vehicle_type), vehicle_fleet)
    route_id = _grouped_choice(_combo_key(warehouses, regions), route_pool)

    # Promised delivery date: the SLA committed to the customer at
    # booking time, tightened for higher-priority shipments.
    promise_base_days = _grouped_choice(
        regions, {r: [d] for r, d in REGION_BASE_PROMISE_DAYS.items()}
    ).astype(int)
    promise_adjustment = _grouped_choice(
        priority, {p: [adj] for p, adj in PRIORITY_PROMISE_ADJUSTMENT_DAYS.items()}
    ).astype(int)
    promised_days = np.maximum(promise_base_days + promise_adjustment, MIN_PROMISE_DAYS)

    # Actual transit days: region baseline, plus a weight penalty for
    # heavy loads (dampened for Enterprise accounts, who consolidate
    # bulk freight onto dedicated capacity rather than queuing it),
    # plus tier-driven handling noise, plus the July Cairo East
    # overload penalty. Enterprise accounts also get a guaranteed
    # priority-handling bonus: despite choosing tighter SLA tiers
    # more often, dedicated account management keeps their actual
    # execution reliably ahead of schedule, which is what ultimately
    # gives them the lowest delay rate of any tier.
    base_days = _grouped_randint(regions, REGION_DELIVERY_DAYS)
    raw_weight_penalty_days = np.minimum(shipment_weight // 150, 3)
    weight_penalty_reliability = _grouped_choice(
        customer_type, {c: [f] for c, f in CUSTOMER_WEIGHT_PENALTY_FACTOR.items()}
    ).astype(float)
    weight_penalty_days = np.floor(raw_weight_penalty_days * weight_penalty_reliability).astype(int)

    tier_noise_days = np.zeros(n, dtype=int)
    retail_mask = customer_type == "Retail"
    tier_noise_days[retail_mask] = rng.integers(0, 2, size=int(retail_mask.sum()))

    actual_days = base_days + weight_penalty_days + tier_noise_days

    enterprise_mask = customer_type == "Enterprise"
    actual_days = np.where(enterprise_mask, np.maximum(actual_days - 1, 1), actual_days)

    sme_bonus_mask = (customer_type == "SME") & (rng.random(n) < 0.20)
    actual_days = np.where(sme_bonus_mask, np.maximum(actual_days - 1, 1), actual_days)

    overload_mask = (warehouses == PROBLEM_WAREHOUSE) & (order_dates.month == PROBLEM_MONTH)
    overload_count = int(overload_mask.sum())
    if overload_count:
        actual_days[overload_mask] += rng.integers(*OVERLOAD_EXTRA_DAYS_RANGE, size=overload_count)

    # Incident rate (Lost / Cancelled) depends on customer tier and
    # spikes sharply for overloaded shipments.
    incident_prob = np.array([CUSTOMER_INCIDENT_RATE[c] for c in customer_type])
    incident_prob[overload_mask] *= OVERLOAD_INCIDENT_RATE_MULTIPLIER
    is_incident = rng.random(n) < incident_prob
    incident_label = np.where(rng.random(n) < 0.45, "Cancelled", "Lost")

    status = np.where(
        is_incident, incident_label, np.where(actual_days > promised_days, "Delayed", "Delivered")
    )

    # Delivery cost scales with transit time and shipment weight, with
    # a rush-shipping surcharge when the origin warehouse is overloaded.
    delivery_cost = 50 + actual_days * 15 + shipment_weight * 0.3 + rng.normal(0, 10, size=n)
    if overload_count:
        delivery_cost[overload_mask] *= rng.uniform(*OVERLOAD_COST_MULTIPLIER_RANGE, size=overload_count)
    delivery_cost = np.clip(delivery_cost, 20, None).round(2)

    # Fuel cost scales with regional road distance and load weight.
    distance_km = _grouped_choice(regions, {r: [d] for r, d in REGION_DISTANCE_KM.items()}).astype(float)
    fuel_cost = distance_km * FUEL_RATE_PER_KM * (1 + shipment_weight / WEIGHT_FUEL_FACTOR_DIVISOR)
    fuel_cost += rng.normal(0, 15, size=n)
    if overload_count:
        fuel_cost[overload_mask] *= rng.uniform(*OVERLOAD_FUEL_MULTIPLIER_RANGE, size=overload_count)
    fuel_cost = np.clip(fuel_cost, 10, None).round(2)

    # Dock handling times scale with vehicle size and load weight.
    # Congestion from the overload event slows loading at the origin
    # warehouse specifically; unloading at the destination is unaffected.
    loading_time = _grouped_randint(vehicle_type, LOADING_TIME_RANGE_BY_VEHICLE).astype(float)
    loading_time += shipment_weight / 50
    if overload_count:
        loading_time[overload_mask] += rng.uniform(*OVERLOAD_LOADING_DELAY_MINUTES_RANGE, size=overload_count)
    loading_time_minutes = np.clip(loading_time, 1, None).round().astype(int)

    unloading_time = _grouped_randint(vehicle_type, UNLOADING_TIME_RANGE_BY_VEHICLE).astype(float)
    unloading_time += shipment_weight / 70
    unloading_time_minutes = np.clip(unloading_time, 1, None).round().astype(int)

    # Driver rating reflects the customer's overall experience: it
    # drops with lateness, with warehouse overload stress, and with
    # outright incidents (a lost or cancelled shipment is a bad
    # experience regardless of whose "fault" it was).
    lateness_days = np.clip(actual_days - promised_days, 0, 10)
    driver_rating = rng.normal(4.4, 0.25, size=n)
    driver_rating -= lateness_days * 0.15
    if overload_count:
        driver_rating[overload_mask] -= rng.uniform(*OVERLOAD_RATING_PENALTY_RANGE, size=overload_count)
    driver_rating = np.where(status == "Lost", driver_rating - 1.5, driver_rating)
    driver_rating = np.where(status == "Cancelled", driver_rating - 0.8, driver_rating)
    driver_rating = np.clip(driver_rating, 1.0, 5.0).round(1)

    delivery_window = _grouped_choice(priority, DELIVERY_WINDOWS_BY_PRIORITY)

    promised_delivery_date = order_dates + pd.to_timedelta(promised_days, unit="D")
    delivery_date = order_dates + pd.to_timedelta(actual_days, unit="D")
    delivery_date = delivery_date.where(~np.isin(status, ["Lost", "Cancelled"]))
    # actual_delivery_date is the standardized TMS field; delivery_date
    # is retained for backward compatibility with the original schema.
    actual_delivery_date = delivery_date

    return pd.DataFrame(
        {
            "shipment_id": shipment_ids,
            "order_date": order_dates,
            "delivery_date": delivery_date,
            "region": regions,
            "warehouse": warehouses,
            "customer": customers,
            "delivery_days": actual_days,
            "status": status,
            "delivery_cost": delivery_cost,
            "driver_rating": driver_rating,
            "driver_id": driver_id,
            "driver_name": driver_name,
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "route_id": route_id,
            "promised_delivery_date": promised_delivery_date,
            "actual_delivery_date": actual_delivery_date,
            "shipment_weight": shipment_weight,
            "priority": priority,
            "customer_type": customer_type,
            "delivery_window": delivery_window,
            "fuel_cost": fuel_cost,
            "loading_time_minutes": loading_time_minutes,
            "unloading_time_minutes": unloading_time_minutes,
        }
    )


# --------------------------------------------------------------------------
# complaints.csv
# --------------------------------------------------------------------------

def generate_complaints(
    shipments_df: pd.DataFrame, n: int = N_COMPLAINTS
) -> pd.DataFrame:
    """Generate customer complaints correlated with shipment problems.

    Shipments that were delayed, lost, cancelled, or routed through
    the overloaded warehouse/month are sampled with much higher
    probability, so complaint volume and category naturally spike
    where operational data shows real problems. Resolution workflow
    fields (team, escalation, resolution time, post-resolution CSAT)
    are derived from the complaint category and the originating
    customer's tier, so Enterprise tickets escalate faster and
    longer-running tickets produce lower satisfaction scores.

    Args:
        shipments_df: The shipments DataFrame produced by
            generate_shipments().
        n: Number of complaint rows to generate.

    Returns:
        A DataFrame with one row per complaint.
    """
    overload_mask = (
        (shipments_df["warehouse"] == PROBLEM_WAREHOUSE)
        & (shipments_df["order_date"].dt.month == PROBLEM_MONTH)
    ).to_numpy()

    weights = np.ones(len(shipments_df))
    weights[shipments_df["status"] == "Delayed"] *= 6
    weights[shipments_df["status"] == "Lost"] *= 5
    weights[shipments_df["status"] == "Cancelled"] *= 3
    weights[overload_mask] *= 2.5
    probabilities = weights / weights.sum()

    sampled_idx = rng.choice(shipments_df.index, size=n, replace=False, p=probabilities)
    sampled = shipments_df.loc[sampled_idx].reset_index(drop=True)

    categories = np.empty(n, dtype=object)
    sentiments = np.empty(n, dtype=object)
    resolution_hours = np.empty(n, dtype=float)
    escalation_level = np.empty(n, dtype=int)
    resolved = np.empty(n, dtype=bool)
    satisfaction = np.full(n, np.nan)

    for i, row in sampled.iterrows():
        is_overloaded = bool(
            row["warehouse"] == PROBLEM_WAREHOUSE and row["order_date"].month == PROBLEM_MONTH
        )

        if row["status"] == "Lost":
            category = rng.choice(["Lost Shipment", "Poor Communication"], p=[0.85, 0.15])
        elif row["status"] == "Delayed" or is_overloaded:
            category = rng.choice(["Late Delivery", "Poor Communication"], p=[0.75, 0.25])
        elif row["status"] == "Cancelled":
            category = rng.choice(["Poor Communication", "Wrong Item"], p=[0.6, 0.4])
        else:
            category = rng.choice(
                ["Damaged Package", "Wrong Item", "Poor Communication"], p=[0.4, 0.35, 0.25]
            )

        if category in ("Lost Shipment", "Late Delivery", "Damaged Package"):
            sentiment = rng.choice(["Negative", "Neutral"], p=[0.85, 0.15])
        else:
            sentiment = rng.choice(["Negative", "Neutral", "Positive"], p=[0.55, 0.35, 0.10])

        is_resolved = bool(rng.random() < CATEGORY_RESOLUTION_RATE[category])
        base_hours = CATEGORY_BASE_RESOLUTION_HOURS[category]
        if is_resolved:
            hours = base_hours * rng.uniform(0.6, 1.6)
        else:
            # Still open: elapsed time so far, necessarily less than a
            # typical full resolution.
            hours = base_hours * rng.uniform(0.2, 0.6)

        level = CATEGORY_BASE_ESCALATION_LEVEL[category]
        if row["customer_type"] == "Enterprise":
            level = min(level + 1, MAX_ESCALATION_LEVEL)

        categories[i] = category
        sentiments[i] = sentiment
        resolution_hours[i] = round(hours, 1)
        escalation_level[i] = level
        resolved[i] = is_resolved
        if is_resolved:
            csat = 5 - (hours / SATISFACTION_HOURS_DIVISOR) + rng.normal(0, 0.4)
            satisfaction[i] = round(float(np.clip(csat, 1, 5)))

    complaint_offsets = rng.integers(2, 12, size=n)
    complaint_dates = sampled["order_date"] + pd.to_timedelta(complaint_offsets, unit="D")
    assigned_team = pd.Series(categories).map(CATEGORY_ASSIGNED_TEAM).to_numpy()

    return pd.DataFrame(
        {
            "complaint_id": [f"CMP{200000 + i}" for i in range(n)],
            "shipment_id": sampled["shipment_id"],
            "complaint_date": complaint_dates,
            "category": categories,
            "sentiment": sentiments,
            "region": sampled["region"],
            "resolution_time_hours": resolution_hours,
            "assigned_team": assigned_team,
            "escalation_level": escalation_level,
            "resolved": resolved,
            "customer_satisfaction_after_resolution": satisfaction,
        }
    )


# --------------------------------------------------------------------------
# inventory.csv
# --------------------------------------------------------------------------

def generate_inventory() -> pd.DataFrame:
    """Generate warehouse inventory records with intentional stock risk.

    Every product is stocked at every warehouse, but each warehouse
    has a different probability of running understocked (Cairo East
    weakest, consistent with its overload story in the shipment
    data). SKU, supplier, and unit cost are fixed per product so the
    same item is priced and sourced consistently across warehouses;
    lead time varies with the assigned supplier, and days-of-supply
    is derived directly from stock and sales velocity.

    Returns:
        A DataFrame with one row per (warehouse, product) combination.
    """
    products = _generate_product_catalog()
    suppliers = list(SUPPLIER_BASE_LEAD_TIME_DAYS.keys())

    sku_by_product = {p: f"SKU-{i:05d}" for i, p in enumerate(products, start=1)}
    supplier_by_product = {p: rng.choice(suppliers) for p in products}
    unit_cost_by_product = {
        p: round(float(rng.uniform(*UNIT_COST_RANGE)), 2) for p in products
    }

    rows = []
    for warehouse in WAREHOUSES:
        low_stock_rate = WAREHOUSE_LOW_STOCK_RATE[warehouse]
        for product in products:
            monthly_sales = int(rng.integers(20, 400))
            reorder_threshold = int(round(monthly_sales * rng.uniform(0.15, 0.35)))

            if rng.random() < low_stock_rate:
                stock_level = int(round(reorder_threshold * rng.uniform(0.2, 0.95)))
            else:
                stock_level = reorder_threshold + int(round(monthly_sales * rng.uniform(0.3, 1.5)))
            stock_level = max(stock_level, 0)

            supplier = supplier_by_product[product]
            unit_cost = unit_cost_by_product[product]
            daily_sales_rate = monthly_sales / 30

            rows.append(
                {
                    "warehouse": warehouse,
                    "product": product,
                    "stock_level": stock_level,
                    "monthly_sales": monthly_sales,
                    "reorder_threshold": reorder_threshold,
                    "sku": sku_by_product[product],
                    "supplier": supplier,
                    "lead_time_days": max(
                        SUPPLIER_BASE_LEAD_TIME_DAYS[supplier] + int(rng.integers(-1, 3)), 1
                    ),
                    "warehouse_capacity": WAREHOUSE_CAPACITY_UNITS[warehouse],
                    "inventory_value": round(stock_level * unit_cost, 2),
                    "days_of_inventory_remaining": round(stock_level / daily_sales_rate, 1),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    """Generate all datasets and write them to the data/ directory."""
    DATA_DIR.mkdir(exist_ok=True)

    print("Generating shipments...")
    shipments_df = generate_shipments()
    shipments_df.to_csv(DATA_DIR / "shipments.csv", index=False)
    print(f"  Wrote {len(shipments_df):,} rows -> data/shipments.csv")

    print("Generating complaints...")
    complaints_df = generate_complaints(shipments_df)
    complaints_df.to_csv(DATA_DIR / "complaints.csv", index=False)
    print(f"  Wrote {len(complaints_df):,} rows -> data/complaints.csv")

    print("Generating inventory...")
    inventory_df = generate_inventory()
    inventory_df.to_csv(DATA_DIR / "inventory.csv", index=False)
    print(f"  Wrote {len(inventory_df):,} rows -> data/inventory.csv")

    print("\nDone. Synthetic data ready in data/.")


if __name__ == "__main__":
    main()
