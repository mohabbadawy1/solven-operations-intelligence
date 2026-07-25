"""Tests for generate_real_estate_data.py: determinism and relational integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_generation_is_deterministic():
    """Re-running the generator with the same seed produces byte-identical output.

    generate_units() draws from the module's shared, stateful `rng` --
    calling it twice back-to-back would advance that state between
    calls and *should* differ, so this resets rng to a fresh instance
    with the same seed before each call, exactly mirroring what
    actually happens across two separate `python generate_real_estate_data.py`
    invocations (a fresh process each time).
    """
    import generate_real_estate_data as gen

    projects_a = gen.generate_projects()
    projects_b = gen.generate_projects()
    pd.testing.assert_frame_equal(projects_a, projects_b)

    gen.rng = np.random.default_rng(gen.SEED)
    units_a = gen.generate_units()
    gen.rng = np.random.default_rng(gen.SEED)
    units_b = gen.generate_units()
    pd.testing.assert_frame_equal(units_a, units_b)


def test_units_reference_valid_projects(datasets):
    valid_projects = set(datasets["projects"]["project_id"])
    assert set(datasets["units"]["project_id"]).issubset(valid_projects)


def test_sales_reference_valid_units_and_projects(datasets):
    valid_units = set(datasets["units"]["unit_id"])
    valid_projects = set(datasets["projects"]["project_id"])
    assert set(datasets["sales"]["unit_id"]).issubset(valid_units)
    assert set(datasets["sales"]["project_id"]).issubset(valid_projects)


def test_payment_schedules_reference_valid_sales(datasets):
    valid_sales = set(datasets["sales"]["sale_id"])
    assert set(datasets["payment_schedules"]["sale_id"]).issubset(valid_sales)


def test_collections_reference_valid_installments(datasets):
    valid_installments = set(datasets["payment_schedules"]["installment_id"])
    assert set(datasets["collections"]["installment_id"]).issubset(valid_installments)


def test_snagging_references_valid_handovers(datasets):
    valid_handovers = set(datasets["handovers"]["handover_id"])
    assert set(datasets["snagging"]["handover_id"]).issubset(valid_handovers)


def test_five_engineered_projects_present(datasets):
    assert set(datasets["projects"]["project_id"]) == {"AUR", "CST", "VTX", "HVW", "MER"}


def test_no_negative_monetary_values(datasets):
    sales = datasets["sales"]
    assert (sales["gross_price"] >= 0).all()
    assert (sales["net_sales_value"] >= 0).all()
    assert (datasets["payment_schedules"]["amount_due"] >= 0).all()


def test_contract_date_never_before_reservation_date(datasets):
    sales = datasets["sales"].dropna(subset=["contract_date"])
    assert (sales["contract_date"] >= sales["reservation_date"]).all()
