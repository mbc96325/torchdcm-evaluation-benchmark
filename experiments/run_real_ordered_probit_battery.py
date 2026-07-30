from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmark_runtime import configure_single_thread_cpu, runtime_policy_metadata


configure_single_thread_cpu(configure_torch=False)


INDICATOR_GROUPS = {
    "Environmental": [f"Envir{index:02d}" for index in range(1, 7)],
    "Mobility": [f"Mobil{index:02d}" for index in range(1, 28)],
    "Residential choice": [f"ResidCh{index:02d}" for index in range(1, 8)],
    "Lifestyle": [f"LifSty{index:02d}" for index in range(1, 15)],
}


def consistency_metrics(payload: dict) -> dict:
    backends = {row["backend"]: row for row in payload["backends"]}
    torch_result = backends.get("torchdcm_fit", {})
    comparisons = {}
    for backend in ("biogeme", "apollo"):
        result = backends.get(backend, {})
        required = (
            torch_result.get("loglike"),
            result.get("loglike"),
            result.get("ll_diff"),
            result.get("max_param_diff"),
            result.get("max_probability_diff"),
        )
        if not result.get("available") or not all(
            value is not None for value in required
        ):
            comparisons[backend] = {
                "available": False,
                "consistent": False,
            }
            continue
        ll_diff = abs(float(result["ll_diff"]))
        param_diff = float(result["max_param_diff"])
        probability_diff = float(result["max_probability_diff"])
        comparisons[backend] = {
            "available": True,
            "abs_loglike_diff": ll_diff,
            "max_param_diff": param_diff,
            "max_probability_diff": probability_diff,
            "max_covariance_diff": result.get("max_covariance_diff"),
            "consistent": bool(
                ll_diff
                <= max(1e-5, 1e-8 * abs(float(torch_result["loglike"])))
                and param_diff <= 1e-3
                and probability_diff <= 1e-4
            ),
        }
    available = [item for item in comparisons.values() if item["available"]]
    if len(available) != 2:
        return {
            "abs_loglike_diff": None,
            "max_param_diff": None,
            "max_probability_diff": None,
            "max_covariance_diff": None,
            "comparisons_to_torchdcm": comparisons,
            "consistent": False,
        }
    covariance_differences = [
        float(item["max_covariance_diff"])
        for item in available
        if item.get("max_covariance_diff") is not None
    ]
    return {
        "abs_loglike_diff": max(item["abs_loglike_diff"] for item in available),
        "max_param_diff": max(item["max_param_diff"] for item in available),
        "max_probability_diff": max(
            item["max_probability_diff"] for item in available
        ),
        "max_covariance_diff": max(covariance_differences)
        if covariance_differences
        else None,
        "comparisons_to_torchdcm": comparisons,
        "consistent": bool(all(item["consistent"] for item in available)),
    }


def run_indicator(
    benchmark: Path,
    indicator: str,
    group: str,
    kind: str,
    max_iter: int,
    timeout: int,
    temp_dir: Path,
) -> dict:
    output = temp_dir / f"{indicator}.json"
    command = [
        sys.executable,
        str(benchmark),
        "--kind",
        kind,
        "--indicator",
        indicator,
        "--mode",
        "full-estimation",
        "--max-iter",
        str(max_iter),
        "--json-output",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=temp_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "indicator": indicator,
            "group": group,
            "status": "timeout",
            "consistent": False,
        }
    if completed.returncode != 0 or not output.exists():
        return {
            "indicator": indicator,
            "group": group,
            "status": "failed",
            "message": completed.stderr.strip() or completed.stdout.strip(),
            "consistent": False,
        }
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload.update(
        {
            "indicator": indicator,
            "group": group,
            "status": "completed",
            **consistency_metrics(payload),
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--kind", choices=["logit", "probit"], default="probit")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indicators", nargs="*")
    args = parser.parse_args()

    benchmark = Path(__file__).with_name("compare_ordered_estimators.py")
    selected = set(args.indicators or [])
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_ordered_{args.kind}_") as temp:
        temp_dir = Path(temp)
        for group, indicators in INDICATOR_GROUPS.items():
            for indicator in indicators:
                if selected and indicator not in selected:
                    continue
                print(f"running {indicator}", flush=True)
                rows.append(
                    run_indicator(
                        benchmark,
                        indicator,
                        group,
                        args.kind,
                        args.max_iter,
                        args.timeout,
                        temp_dir,
                    )
                )

    result = {
        "benchmark": f"real_ordered_{args.kind}_battery",
        "dataset": "Biogeme Optima",
        "model": f"Ordered {args.kind}",
        "indicator_groups": INDICATOR_GROUPS,
        "runtime_policy": runtime_policy_metadata(),
        "tolerances": {
            "loglike": "max(1e-5, 1e-8 * abs(torchdcm_loglike))",
            "parameter": 1e-3,
            "probability": 1e-4,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
