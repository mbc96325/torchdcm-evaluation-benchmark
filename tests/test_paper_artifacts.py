from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import torchdcm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_main_paper_case_counts():
    expected = {
        "synthetic_mnl_single_core.json": 15,
        "generated_choice_battery_table4_nl.json": 15,
        "generated_choice_battery_table4_mixl.json": 15,
        "solver_attempt_matrix_mnl_single_core.json": 17,
        "nested_real_battery_single_core.json": 12,
        "mixed_real_battery_apollo.json": 16,
        "torch_device_stress_battery.json": 9,
    }
    for filename, count in expected.items():
        assert len(load(filename)) == count, filename


def test_electronic_companion_case_counts():
    for kind in ("logit", "probit"):
        synthetic = load(
            f"ordered_{kind}_synthetic_threeway_single_core.json"
        )
        actual = load(f"ordered_{kind}_real_threeway_single_core.json")
        assert len(synthetic["rows"]) == 10
        assert len(actual["rows"]) == 54
    assert len(load("advanced_full_estimation.json")["cases"]) == 18


def test_all_paper_datasets_are_archived():
    with (ROOT / "data" / "dataset_index.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18
    assert all(row["status"] == "included" for row in rows)
    small = [row for row in rows if row["storage"] == "github_small"]
    assert len(small) == 16
    for row in small:
        assert (ROOT / "data" / "small" / row["dataset_id"] / "data.csv").is_file()

    lpmc = ROOT / "data" / "raw" / "lpmc_london"
    assert sha256(lpmc / "data.csv") == (
        "777b6fb7c2db219582cc318238f23864d642621ef4ae40bdb9e1e7e27b278d0b"
    )
    assert sha256(lpmc / "data.dat") == (
        "033b9eda6742ffa40716bcc4c32bf05cf3a9787058fb8335b65468ac48455338"
    )

    nhts = ROOT / "data" / "raw" / "nhts_2022" / "csv.zip"
    assert sha256(nhts) == (
        "64530c396d5f164d2259a22f7042f27bee5147babcd367568ddbfafe6c8bf34c"
    )
    with zipfile.ZipFile(nhts) as archive:
        assert "tripv2pub.csv" in archive.namelist()


def test_intermediate_outputs_are_archived():
    intermediate = RESULTS / "intermediate"
    mnl = intermediate / "mnl"
    assert len(list(mnl.glob("*.json"))) == 9
    assert len(list(mnl.glob("*.md"))) == 9
    assert len(list((intermediate / "advanced_full_logs").glob("*.log"))) == 18
    base = json.loads(
        (intermediate / "advanced_torchdcm_apollo.json").read_text(
            encoding="utf-8"
        )
    )
    replacement = json.loads(
        (intermediate / "advanced_swissmetro_3500.json").read_text(
            encoding="utf-8"
        )
    )
    sources = {
        (case["kind"], case["case"]): case
        for case in [*base["cases"], *replacement["cases"]]
    }
    final = load("advanced_full_estimation.json")
    for case in final["cases"]:
        source = sources[(case["kind"], case["case"])]
        for backend in ("torchdcm", "apollo"):
            assert source["results"][backend]["loglike"] == (
                case["results"][backend]["loglike"]
            )
            assert source["results"][backend]["seconds"] == (
                case["results"][backend]["seconds"]
            )


def test_vendored_torchdcm_snapshot():
    source = ROOT / "torchdcm"
    assert (source / "__init__.py").is_file()
    assert torchdcm.__version__ == "0.1.1"
    assert Path(torchdcm.__file__).resolve().parent == source.resolve()
