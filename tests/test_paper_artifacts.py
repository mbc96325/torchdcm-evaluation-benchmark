from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"


def load(name: str):
    return json.loads((GENERATED / name).read_text(encoding="utf-8"))


def test_main_paper_case_counts():
    expected = {
        "synthetic_mnl_single_core_office.json": 15,
        "generated_choice_battery_table4_nl_office.json": 15,
        "generated_choice_battery_table4_mixl_office.json": 15,
        "solver_attempt_matrix_mnl_single_core_office.json": 17,
        "nested_real_battery_single_core_office.json": 12,
        "mixed_real_battery_apollo_office.json": 16,
        "torch_device_stress_battery.json": 9,
    }
    for filename, count in expected.items():
        assert len(load(filename)) == count, filename


def test_electronic_companion_case_counts():
    for kind in ("logit", "probit"):
        synthetic = load(
            f"ordered_{kind}_synthetic_threeway_single_core_office.json"
        )
        actual = load(f"ordered_{kind}_real_threeway_single_core_office.json")
        assert len(synthetic["rows"]) == 10
        assert len(actual["rows"]) == 54
    assert len(load("advanced_full_estimation_office.json")["cases"]) == 18


def test_dataset_catalog_matches_paper_scope():
    with (ROOT / "datasets" / "dataset_index.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18
    committed = [row for row in rows if row["storage"] == "github_small"]
    assert len(committed) == 16
    for row in committed:
        assert (ROOT / "datasets" / "small" / row["dataset_id"] / "data.csv").is_file()
