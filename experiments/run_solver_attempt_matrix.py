from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark_runtime import configure_single_thread_cpu, runtime_policy_metadata

if __name__ == "__main__":
    configure_single_thread_cpu(configure_torch=False)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS = ROOT / "results"

SOLVERS = {
    "torchdcm": {"torchdcm", "torchdcm_fit", "torchdcm_fixed"},
    "torch-choice": {"torch_choice"},
    "scipy_bfgs": {"scipy_bfgs"},
    "biogeme": {"biogeme", "biogeme_fixed"},
    "apollo": {"apollo", "apollo_r_fixed"},
    "mlogit": {"mlogit"},
    "gmnl": {"gmnl"},
    "xlogit": {"xlogit"},
}

BACKEND_PATTERN = re.compile(
    r"^(torchdcm|torchdcm_fit|torchdcm_fixed|torch_choice|scipy_bfgs|biogeme|biogeme_fixed|apollo|apollo_r_fixed|mlogit|gmnl|xlogit)\s+"
)

LL_ABS_TOL = 0.25
LL_REL_TOL = 1e-5
LL_PER_OBSERVATION_TOL = 0.01


@dataclass(frozen=True)
class MatrixCase:
    case: str
    dataset: str
    model: str
    alignment_mode: str
    command: list[str]


def benchmark_cases() -> list[MatrixCase]:
    cases = [
        MatrixCase(
            "swissmetro_mnl",
            "Swissmetro",
            "MNL",
            "full_estimation",
            ["compare_mnl_estimators.py", "--case", "swissmetro", "--n-obs", "10719", "--initial", "zero"],
        ),
        MatrixCase(
            "nhts_2022_mnl",
            "NHTS 2022",
            "MNL",
            "full_estimation",
            ["compare_nhts_mnl.py", "--max-iter", "500"],
        ),
    ]
    for case in ["airline", "parking", "telephone", "lpmc"]:
        cases.append(
            MatrixCase(
                f"biogeme_public_{case}",
                case,
                "MNL",
                "full_estimation",
                ["compare_biogeme_public_mnl.py", "--case", case],
            )
        )
    for dataset in [
        "catsup",
        "cracker",
        "electricity",
        "hc",
        "heating",
        "mode",
        "nox",
        "risky_transport",
        "train",
    ]:
        cases.append(
            MatrixCase(
                f"mlogit_{dataset}",
                dataset,
                "MNL",
                "full_estimation",
                ["run_mlogit_dataset_battery.py", "--profile", f"solver_matrix_{dataset}", "--datasets", dataset, "--max-iter", "180"],
            )
        )
    for dataset in ["fishing", "modecanada"]:
        cases.append(
            MatrixCase(
                f"mlogit_{dataset}",
                dataset,
                "MNL",
                "full_estimation",
                ["compare_mlogit_estimators.py", "--dataset", dataset],
            )
        )
    return cases


def run_case(case: MatrixCase, python: str) -> dict:
    command = [python, str(EXPERIMENTS_DIR / case.command[0]), *case.command[1:]]
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=EXPERIMENTS_DIR, text=True, capture_output=True)
    wall_s = time.perf_counter() - start
    backend_rows = parse_backend_rows(proc.stdout)
    backend_rows.update(parse_sidecar_backend_rows(case))
    n_obs = parse_n_obs(proc.stdout)
    mark_worse_loglikes(backend_rows, n_obs)
    solver_status = summarize_solvers(backend_rows)
    if proc.returncode != 0:
        for solver in solver_status.values():
            if solver["status"] == "unsupported":
                continue
            solver["status"] = "failed"
        solver_status["case_runner"] = {
            "status": "failed",
            "seconds": wall_s,
            "message": (proc.stderr or proc.stdout).strip()[-1000:],
        }
    else:
        solver_status["case_runner"] = {"status": "ok", "seconds": wall_s}
    return {
        "case": case.case,
        "dataset": case.dataset,
        "model": case.model,
        "alignment_mode": case.alignment_mode,
        "runtime_policy": runtime_policy_metadata(),
        "command": [
            "python",
            f"experiments/{case.command[0]}",
            *case.command[1:],
        ],
        "returncode": proc.returncode,
        "wall_seconds": wall_s,
        "solver_status": solver_status,
        "backend_rows": backend_rows,
        "n_obs": n_obs,
        "consistent": consistency_label(backend_rows),
        "stderr_tail": proc.stderr.strip()[-1000:],
    }


def parse_backend_rows(text: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in text.splitlines():
        if not BACKEND_PATTERN.match(line):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[1] not in {"True", "False"}:
            continue
        row = {"backend": parts[0], "available": parts[1] == "True", "raw": line}
        if row["available"] and len(parts) >= 3:
            row["total_s"] = parse_float(parts[2])
        if row["available"] and len(parts) >= 6:
            row["loglike"] = parse_float(parts[5])
        if not row["available"]:
            row["message"] = " ".join(parts[2:])
        rows[parts[0]] = row
    return rows


def parse_sidecar_backend_rows(case: MatrixCase) -> dict[str, dict]:
    if not case.command or case.command[0] != "run_mlogit_dataset_battery.py":
        return {}
    profile = None
    if "--profile" in case.command:
        profile = case.command[case.command.index("--profile") + 1]
    if not profile:
        return {}
    path = RESULTS / f"mlogit_dataset_battery_{profile}.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not rows:
        return {}
    row = rows[0]
    backend_rows: dict[str, dict] = {}
    torch_row = row.get("torchdcm")
    if torch_row:
        backend_rows["torchdcm"] = {
            "backend": "torchdcm",
            "available": bool(torch_row.get("available")),
            "total_s": parse_float(str(torch_row.get("total_seconds", "NA"))),
            "loglike": parse_float(str(torch_row.get("loglike", "NA"))),
            "raw": json.dumps(torch_row),
            "message": torch_row.get("message", ""),
        }
    torch_choice_row = row.get("torch_choice")
    if torch_choice_row:
        backend_rows["torch_choice"] = {
            "backend": "torch_choice",
            "available": bool(torch_choice_row.get("available")),
            "total_s": parse_float(str(torch_choice_row.get("total_seconds", "NA"))),
            "loglike": parse_float(str(torch_choice_row.get("loglike", "NA"))),
            "raw": json.dumps(torch_choice_row),
            "message": torch_choice_row.get("message", ""),
        }
    scipy_row = row.get("scipy_bfgs")
    if scipy_row:
        backend_rows["scipy_bfgs"] = {
            "backend": "scipy_bfgs",
            "available": bool(scipy_row.get("available")),
            "total_s": parse_float(str(scipy_row.get("total_seconds", "NA"))),
            "loglike": parse_float(str(scipy_row.get("loglike", "NA"))),
            "raw": json.dumps(scipy_row),
            "message": scipy_row.get("message", ""),
        }
    ref_row = row.get("mlogit")
    if ref_row:
        backend_rows["mlogit"] = {
            "backend": "mlogit",
            "available": bool(ref_row.get("available", row.get("status") == "ok")),
            "total_s": parse_float(str(ref_row.get("total_seconds", "NA"))),
            "loglike": parse_float(str(ref_row.get("loglike", "NA"))),
            "raw": json.dumps(ref_row),
            "message": ref_row.get("message", ""),
        }
    biogeme_row = row.get("biogeme")
    if biogeme_row:
        backend_rows["biogeme"] = {
            "backend": "biogeme",
            "available": bool(biogeme_row.get("available")),
            "total_s": parse_float(str(biogeme_row.get("total_seconds", "NA"))),
            "loglike": parse_float(str(biogeme_row.get("loglike", "NA"))),
            "raw": json.dumps(biogeme_row),
            "message": biogeme_row.get("message", ""),
        }
    apollo_row = row.get("apollo")
    if apollo_row:
        backend_rows["apollo"] = {
            "backend": "apollo",
            "available": bool(apollo_row.get("available")),
            "total_s": parse_float(str(apollo_row.get("total_seconds", "NA"))),
            "loglike": parse_float(str(apollo_row.get("loglike", "NA"))),
            "raw": json.dumps(apollo_row),
            "message": apollo_row.get("message", ""),
        }
    gmnl_row = row.get("gmnl")
    if gmnl_row:
        backend_rows["gmnl"] = {
            "backend": "gmnl",
            "available": bool(gmnl_row.get("available")),
            "total_s": parse_float(str(gmnl_row.get("total_seconds", "NA"))),
            "loglike": parse_float(str(gmnl_row.get("loglike", "NA"))),
            "raw": json.dumps(gmnl_row),
            "message": gmnl_row.get("message", ""),
        }
    xlogit_row = row.get("xlogit")
    if xlogit_row:
        backend_rows["xlogit"] = {
            "backend": "xlogit",
            "available": bool(xlogit_row.get("available")),
            "total_s": parse_float(str(xlogit_row.get("total_seconds", "NA"))),
            "loglike": parse_float(str(xlogit_row.get("loglike", "NA"))),
            "raw": json.dumps(xlogit_row),
            "message": xlogit_row.get("message", ""),
        }
    return backend_rows


def summarize_solvers(backend_rows: dict[str, dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for solver, backend_names in SOLVERS.items():
        matching = [backend_rows[name] for name in backend_names if name in backend_rows]
        if not matching:
            result[solver] = {"status": "unsupported", "message": "No aligned wrapper for this solver/model case."}
            continue
        row = matching[0]
        if row.get("available"):
            result[solver] = {
                "status": "ok",
                "seconds": row.get("total_s"),
                "loglike": row.get("loglike"),
                "worse_loglike": row.get("worse_loglike", False),
                "backend": row["backend"],
            }
        else:
            result[solver] = {"status": "failed", "backend": row["backend"], "message": row.get("message", row.get("raw", ""))}
    return result


def mark_worse_loglikes(backend_rows: dict[str, dict], n_obs: int | None = None) -> None:
    completed = [
        row
        for row in backend_rows.values()
        if row.get("available")
        and isinstance(row.get("loglike"), (int, float))
        and math.isfinite(float(row["loglike"]))
    ]
    if not completed:
        return
    best_loglike = max(float(row["loglike"]) for row in completed)
    tolerance = max(
        LL_ABS_TOL,
        LL_REL_TOL * abs(best_loglike),
        LL_PER_OBSERVATION_TOL * n_obs if n_obs is not None else 0.0,
    )
    for row in completed:
        row["worse_loglike"] = float(row["loglike"]) < best_loglike - tolerance


def consistency_label(backend_rows: dict[str, dict]) -> str:
    comparable = [
        row
        for row in backend_rows.values()
        if row.get("available")
        and isinstance(row.get("loglike"), (int, float))
        and math.isfinite(float(row["loglike"]))
        and not row.get("worse_loglike", False)
    ]
    return "Yes" if len(comparable) >= 2 else "N.A."


def parse_n_obs(text: str) -> int | None:
    match = re.search(r"^n_obs:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_float(text: str):
    try:
        if text == "NA":
            return None
        return float(text)
    except ValueError:
        return None


def render_markdown(rows: list[dict]) -> str:
    columns = ["case", "dataset", "model", *SOLVERS.keys()]
    lines = [
        "# Solver Attempt Matrix",
        "",
        "Each benchmark case is attempted with every configured solver where an aligned wrapper exists. Runtimes report estimation plus covariance on one logical CPU. A dagger marks a clearly inferior final log likelihood; its runtime is retained but its estimate is excluded from consistency.",
        "`ok` means the solver completed the aligned case, `failed` means the wrapper was attempted but failed, and `unsupported` means no aligned wrapper exists for that solver/model case.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = [row["case"], row["dataset"], row["model"]]
        for solver in SOLVERS:
            status = row["solver_status"][solver]
            label = status["status"]
            if status["status"] == "ok" and status.get("seconds") is not None:
                label = f"ok ({status['seconds']:.3f}s)"
                if status.get("worse_loglike"):
                    label += "†"
            cells.append(label)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Failures", ""])
    for row in rows:
        for solver in SOLVERS:
            status = row["solver_status"][solver]
            if status["status"] == "failed":
                message = status.get("message") or row.get("stderr_tail") or ""
                lines.append(f"- `{row['case']}` / `{solver}`: {message[:500]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="full")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    cases = benchmark_cases()
    if args.cases:
        wanted = set(args.cases)
        cases = [case for case in cases if case.case in wanted]
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case, args.python) for case in cases]
    json_path = RESULTS / f"solver_attempt_matrix_{args.profile}.json"
    md_path = RESULTS / f"solver_attempt_matrix_{args.profile}.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    for row in rows:
        ok_count = sum(1 for solver in SOLVERS if row["solver_status"][solver]["status"] == "ok")
        fail_count = sum(1 for solver in SOLVERS if row["solver_status"][solver]["status"] == "failed")
        print(f"{row['case']}: ok={ok_count} failed={fail_count}")


if __name__ == "__main__":
    main()
