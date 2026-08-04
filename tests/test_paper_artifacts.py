from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torchdcm

from experiments.summarize_ordered_results import domain_summaries, maxima


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


def test_torch_choice_results_are_archived():
    for filename in (
        "synthetic_mnl_single_core.json",
        "generated_choice_battery_table4_nl.json",
        "nested_real_battery_single_core.json",
    ):
        for row in load(filename):
            result = next(
                item for item in row["backends"]
                if item["backend"] == "torch_choice"
            )
            assert result["available"] is True
            assert result["worse_loglike"] is False

    for row in load("solver_attempt_matrix_mnl_single_core.json"):
        torchdcm = row["backend_rows"]["torchdcm"]
        result = row["backend_rows"]["torch_choice"]
        assert torchdcm["available"] is True
        assert result["available"] is True
        assert result["worse_loglike"] is False
        for backend in (torchdcm, result):
            assert backend["estimate_s"] > 0
            assert backend["covariance_s"] > 0
            assert backend["total_s"] == pytest.approx(
                backend["estimate_s"] + backend["covariance_s"]
            )
        assert row["solver_status"]["torch-choice"]["status"] == "ok"


def test_electronic_companion_case_counts():
    for kind in ("logit", "probit"):
        synthetic = load(
            f"ordered_{kind}_synthetic_threeway_single_core.json"
        )
        actual = load(f"ordered_{kind}_real_threeway_single_core.json")
        assert len(synthetic["rows"]) == 10
        assert len(actual["rows"]) == 54
    assert len(load("advanced_full_estimation.json")["cases"]) == 18


def test_advanced_table_diagnostics_are_archived():
    expected = {
        ("latent_class", "Synthetic 2,000"): (2.2340864529532545e-4, 4.5071749136260664e-5, 7.324154968080254e-5),
        ("latent_class", "Synthetic 5,000"): (2.93890082735504e-4, 3.191804808422294e-5, 5.9071982590641614e-5),
        ("latent_class", "Synthetic 10,000"): (2.1281393966510187e-6, 4.6975296330575844e-7, 2.6002561959170833e-7),
        ("latent_class", "Swissmetro 2,000"): (1.8983474101736952e-3, 9.568040243168596e-5, 5.692539107286787e-5),
        ("latent_class", "Swissmetro 3,500"): (1.7415695220179828e-5, 2.16464298607999e-6, 3.373994409583414e-6),
        ("latent_class", "Swissmetro 5,000"): (1.0368461083682945e-3, 2.195495428791716e-5, 3.437941201328565e-5),
        ("hybrid_choice", "Synthetic 500"): (2.2780071182726402e-6, 9.906976097262543e-7),
        ("hybrid_choice", "Synthetic 2,000"): (3.961017049691762e-6, 3.120366864561852e-6),
        ("hybrid_choice", "Synthetic 10,000"): (2.386953181776619e-7, 6.279999970049133e-8),
        ("hybrid_choice", "Optima 500"): (2.8849174396050614e-6, 5.271626530856111e-7),
        ("hybrid_choice", "Optima 1,000"): (1.2139216381656937e-5, 3.5568421347387247e-6),
        ("hybrid_choice", "Optima 1,298"): (5.127314497865854e-6, 1.2421327281186834e-6),
        ("panel_likelihood", "Synthetic 250x2"): (6.571974518920776e-6, 3.176976268171039e-6),
        ("panel_likelihood", "Synthetic 500x4"): (1.9627221253815108e-7, 2.870089654827268e-8),
        ("panel_likelihood", "Synthetic 1,250x8"): (4.5622417077506583e-7, 3.517458146279351e-8),
        ("panel_likelihood", "Electricity 100"): (3.882486106360217e-7, 2.357490050885428e-11),
        ("panel_likelihood", "Electricity 250"): (8.600253248225442e-6, 4.483663299035927e-10),
        ("panel_likelihood", "Electricity 348"): (4.285030679729296e-7, 6.090926880905948e-11),
    }
    for case in load("advanced_full_estimation.json")["cases"]:
        diagnostics = case["diagnostics"]
        values = [diagnostics["max_parameter_difference"]]
        if case["kind"] == "latent_class":
            values.extend(
                [
                    diagnostics["max_mean_class_share_difference"],
                    diagnostics["max_probability_difference"],
                ]
            )
        elif case["kind"] == "hybrid_choice":
            values.append(diagnostics["max_probability_difference"])
        else:
            values.append(diagnostics["max_sequence_probability_difference"])
        assert values == pytest.approx(expected[(case["kind"], case["case"])])


def test_ordered_summary_diagnostics_are_reproducible():
    synthetic_logit = load("ordered_logit_synthetic_threeway_single_core.json")
    synthetic_probit = load("ordered_probit_synthetic_threeway_single_core.json")
    assert list(maxima(synthetic_logit).values()) == pytest.approx(
        [1.1569045454962179e-7, 2.1107926071906036e-5, 2.691609219840352e-6]
    )
    assert list(maxima(synthetic_probit).values()) == pytest.approx(
        [5.015135684516281e-8, 6.927409984536226e-6, 3.2711932814732947e-6]
    )

    domains = domain_summaries(
        {
            "logit": load("ordered_logit_real_threeway_single_core.json"),
            "probit": load("ordered_probit_real_threeway_single_core.json"),
        }
    )
    expected = {
        "Environmental": (1.130393229686888e-6, 8.644273809599312e-5, 1.992352649032858e-5),
        "Mobility": (1.4002062016515993e-6, 3.2797483777091685e-4, 3.117287214615683e-5),
        "Residential choice": (5.45376224181382e-7, 1.3153482763117452e-4, 1.1484619032609e-5),
        "Lifestyle": (8.527717909601051e-7, 5.193664154561073e-5, 1.4180785255279993e-5),
    }
    for domain, values in expected.items():
        assert list(domains[domain]["maxima"].values()) == pytest.approx(values)


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
    assert torchdcm.__version__ == "0.1.2"
    assert Path(torchdcm.__file__).resolve().parent == source.resolve()
