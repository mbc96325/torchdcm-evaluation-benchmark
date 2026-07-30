from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


CASES = [
    ("latent_class", "Synthetic 2,000"),
    ("latent_class", "Synthetic 5,000"),
    ("latent_class", "Synthetic 10,000"),
    ("latent_class", "Swissmetro 2,000"),
    ("latent_class", "Swissmetro 3,500"),
    ("latent_class", "Swissmetro 5,000"),
    ("hybrid_choice", "Synthetic 500"),
    ("hybrid_choice", "Synthetic 2,000"),
    ("hybrid_choice", "Synthetic 10,000"),
    ("hybrid_choice", "Optima 500"),
    ("hybrid_choice", "Optima 1,000"),
    ("hybrid_choice", "Optima 1,298"),
    ("panel_likelihood", "Synthetic 250x2"),
    ("panel_likelihood", "Synthetic 500x4"),
    ("panel_likelihood", "Synthetic 1,250x8"),
    ("panel_likelihood", "Electricity 100"),
    ("panel_likelihood", "Electricity 250"),
    ("panel_likelihood", "Electricity 348"),
]


def _failed_biogeme(message: str) -> dict:
    return {
        "backend": "biogeme",
        "available": False,
        "seconds": None,
        "loglike": None,
        "params": None,
        "covariance_available": None,
        "convergence": None,
        "worse_loglike": None,
        "message": message,
    }


def _refresh_summary(case: dict) -> None:
    available = [
        result
        for result in case["results"].values()
        if result.get("available") and result.get("loglike") is not None
    ]
    best = max((float(result["loglike"]) for result in available), default=None)
    tolerance = (
        max(0.25, 1e-5 * abs(best), 0.01 * case["n_obs"])
        if best is not None
        else None
    )
    for result in available:
        result["worse_loglike"] = bool(
            best - float(result["loglike"]) > tolerance
        )
    comparable = [
        result for result in available if not result.get("worse_loglike", False)
    ]
    case["best_loglike"] = best
    case["loglike_tolerance"] = tolerance
    case["consistent"] = (
        None
        if len(comparable) < 2
        else all(
            best - float(result["loglike"]) <= tolerance
            for result in comparable
        )
    )


def _run(command: list[str], log_path: Path, timeout: int | None = None) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-json", type=Path)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()

    runner = Path(__file__).with_name("compare_advanced_estimators.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or args.output.parent / "advanced_full_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="torchdcm_advanced_full_") as tmp:
        temporary = Path(tmp)
        if args.base_json:
            payload = json.loads(args.base_json.read_text(encoding="utf-8"))
        else:
            base_path = temporary / "torch_apollo.json"
            print("[suite] running TorchDCM and Apollo cases", flush=True)
            _run(
                [
                    sys.executable,
                    "-u",
                    str(runner),
                    "--profile",
                    "full",
                    "--backends",
                    "torchdcm",
                    "apollo",
                    "--max-iter",
                    str(args.max_iter),
                    "--timeout",
                    str(args.timeout),
                    "--output",
                    str(base_path),
                ],
                log_dir / "torch_apollo.log",
            )
            payload = json.loads(base_path.read_text(encoding="utf-8"))

        indexed = {(case["kind"], case["case"]): case for case in payload["cases"]}
        for index, (kind, case_name) in enumerate(CASES, start=1):
            print(f"[suite] Biogeme {index}/{len(CASES)}: {case_name}", flush=True)
            case_path = temporary / f"biogeme_{index:02d}.json"
            log_path = log_dir / f"biogeme_{index:02d}.log"
            command = [
                sys.executable,
                "-u",
                str(runner),
                "--profile",
                "full",
                "--kinds",
                kind,
                "--backends",
                "biogeme",
                "--cases",
                case_name,
                "--max-iter",
                str(args.max_iter),
                "--timeout",
                str(args.timeout),
                "--output",
                str(case_path),
            ]
            try:
                _run(command, log_path, timeout=args.timeout)
                result_payload = json.loads(case_path.read_text(encoding="utf-8"))
                biogeme = result_payload["cases"][0]["results"]["biogeme"]
            except subprocess.TimeoutExpired:
                biogeme = _failed_biogeme(
                    f"full estimation exceeded {args.timeout} seconds"
                )
                print(f"[suite] timeout: {case_name}", flush=True)
            except Exception as exc:
                biogeme = _failed_biogeme(f"{type(exc).__name__}: {exc}")
                print(f"[suite] failed: {case_name}: {exc}", flush=True)
            indexed[(kind, case_name)]["results"]["biogeme"] = biogeme
            _refresh_summary(indexed[(kind, case_name)])

        payload["timing_scope"] = "full optimization plus classic covariance construction"
        payload["solver_timeout_seconds"] = args.timeout
        payload["cases"] = [indexed[key] for key in CASES]
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[suite] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
