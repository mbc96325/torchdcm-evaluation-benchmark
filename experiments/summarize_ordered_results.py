from __future__ import annotations

import argparse
import json
from pathlib import Path


BACKENDS = ("torchdcm_fit", "biogeme", "apollo")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime(row: dict, backend: str) -> float:
    result = next(item for item in row["backends"] if item["backend"] == backend)
    return float(result["total_s"])


def maxima(payload: dict) -> dict[str, float]:
    rows = payload["rows"]
    return {
        "abs_loglike_diff": max(float(row["abs_loglike_diff"]) for row in rows),
        "max_param_diff": max(float(row["max_param_diff"]) for row in rows),
        "max_probability_diff": max(
            float(row["max_probability_diff"]) for row in rows
        ),
    }


def runtime_ranges(payload: dict) -> dict[str, list[float]]:
    return {
        backend: [
            min(runtime(row, backend) for row in payload["rows"]),
            max(runtime(row, backend) for row in payload["rows"]),
        ]
        for backend in BACKENDS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    args = parser.parse_args()
    root = args.generated_dir
    synthetic = {
        kind: load(
            root
            / f"ordered_{kind}_synthetic_threeway_single_core_office.json"
        )
        for kind in ("logit", "probit")
    }
    actual = {
        kind: load(root / f"ordered_{kind}_real_threeway_single_core_office.json")
        for kind in ("logit", "probit")
    }

    print("SYNTHETIC_ROWS")
    synthetic_by_case = {
        kind: {row["case"]: row for row in payload["rows"]}
        for kind, payload in synthetic.items()
    }
    for case, logit_row in synthetic_by_case["logit"].items():
        probit_row = synthetic_by_case["probit"][case]
        fields = [
            case,
            f'{logit_row["n_obs"]:,}',
            str(logit_row["n_categories"]),
            str(logit_row["n_variables"]),
            f'{logit_row["rho"]:.2f}',
            *[
                f'{logit_row[backend]["runtime"]:.3f}'
                for backend in ("torchdcm", "biogeme", "apollo")
            ],
            *[
                f'{probit_row[backend]["runtime"]:.3f}'
                for backend in ("torchdcm", "biogeme", "apollo")
            ],
            "Yes" if logit_row["consistent"] and probit_row["consistent"] else "No",
        ]
        print(" & ".join(fields) + r" \\")

    print("ACTUAL_ROWS")
    actual_by_indicator = {
        kind: {row["indicator"]: row for row in payload["rows"]}
        for kind, payload in actual.items()
    }
    current_group = None
    for indicator, logit_row in actual_by_indicator["logit"].items():
        probit_row = actual_by_indicator["probit"][indicator]
        if logit_row["group"] != current_group:
            current_group = logit_row["group"]
            print(f"% {current_group}")
        fields = [
            indicator,
            f'{logit_row["n_obs"]:,}',
            *[f'{runtime(logit_row, backend):.3f}' for backend in BACKENDS],
            *[f'{runtime(probit_row, backend):.3f}' for backend in BACKENDS],
            "Yes" if logit_row["consistent"] and probit_row["consistent"] else "No",
        ]
        print(" & ".join(fields) + r" \\")

    summary = {
        "synthetic": {
            kind: {
                "completed": sum(row["status"] == "completed" for row in payload["rows"]),
                "consistent": sum(bool(row["consistent"]) for row in payload["rows"]),
                "maxima": maxima(payload),
            }
            for kind, payload in synthetic.items()
        },
        "actual": {
            kind: {
                "completed": sum(row["status"] == "completed" for row in payload["rows"]),
                "consistent": sum(bool(row["consistent"]) for row in payload["rows"]),
                "maxima": maxima(payload),
                "runtime_ranges": runtime_ranges(payload),
            }
            for kind, payload in actual.items()
        },
    }
    print("SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
