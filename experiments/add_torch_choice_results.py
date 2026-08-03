from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from torchdcm import MultinomialLogit

from benchmark_runtime import configure_single_thread_cpu, runtime_policy_metadata

configure_single_thread_cpu(configure_torch=True)

import compare_biogeme_public_mnl as public_mnl
import compare_generated_choice_battery as generated
import compare_mlogit_estimators as mlogit_estimators
import compare_mnl_estimators as swissmetro_mnl
import compare_nhts_mnl as nhts_mnl
import compare_real_nested_logit_battery as nested_real
import run_mlogit_dataset_battery as mlogit_battery
import run_solver_attempt_matrix as solver_matrix
from torch_choice_backend import run_mnl, run_nested


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20260709
LAMBDA_MIN = 0.0001


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _insert_backend(backends: list[dict], addition: dict) -> None:
    backends[:] = [row for row in backends if row.get("backend") != addition["backend"]]
    position = next(
        (index + 1 for index, row in enumerate(backends) if row.get("backend") == "torchdcm"),
        0,
    )
    backends.insert(position, addition)


def _mark_json_loglikes(backends: list[dict], n_obs: int) -> str:
    completed = [
        row
        for row in backends
        if row.get("available")
        and isinstance(row.get("loglike"), (int, float))
        and np.isfinite(float(row["loglike"]))
    ]
    if not completed:
        return "N.A."
    best = max(float(row["loglike"]) for row in completed)
    tolerance = max(0.25, 1e-5 * abs(best), 0.01 * n_obs)
    for row in completed:
        row["worse_loglike"] = float(row["loglike"]) < best - tolerance
    comparable = [row for row in completed if not row.get("worse_loglike", False)]
    if len(comparable) < 2:
        return "N.A."
    return "Yes" if all(abs(float(row["loglike"]) - best) <= tolerance for row in comparable) else "No"


def _backend_payload(result, reference=None, parameter_names: list[str] | None = None) -> dict:
    payload = {
        "backend": "torch_choice",
        "available": bool(result.available),
        "total_s": result.total_s,
        "estimate_s": result.estimate_s,
        "covariance_s": result.covariance_s,
        "loglike": result.loglike,
        "ll_diff": None,
        "max_param_diff": None,
        "max_prob_diff": None,
        "max_cov_diff": None,
        "max_se_diff": None,
        "worse_loglike": False,
        "message": result.message,
    }
    if result.available and reference is not None and reference.available:
        payload["ll_diff"] = float(result.loglike - reference.loglike)
        if parameter_names:
            payload["max_param_diff"] = max(
                abs(result.params[name] - reference.params[name]) for name in parameter_names
            )
        if result.probabilities is not None and reference.probabilities is not None:
            payload["max_prob_diff"] = float(
                np.max(np.abs(result.probabilities - reference.probabilities))
            )
        if result.covariance is not None and reference.covariance is not None:
            payload["max_cov_diff"] = float(
                np.max(np.abs(result.covariance - reference.covariance))
            )
            payload["max_se_diff"] = float(
                np.max(
                    np.abs(
                        np.sqrt(np.diag(result.covariance))
                        - np.sqrt(np.diag(reference.covariance))
                    )
                )
            )
    return payload


def add_synthetic_mnl(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        meta_dict = row.get("generated", row.get("results"))
        meta = generated.GeneratedSpec(
            case=row["case"],
            model="mnl",
            n_obs=int(meta_dict["N"]),
            n_alternatives=int(meta_dict["J"]),
            n_variables=int(meta_dict["K"]),
            rho=float(meta_dict["rho"]),
        )
        case = generated.build_mnl_case(meta, SEED)
        runs = [run_mnl(case.data, case.spec, case.initial_values, max_iter=120) for _ in range(3)]
        completed = [result for result in runs if result.available]
        result = sorted(completed, key=lambda item: float(item.total_s))[1] if completed else runs[0]
        result.runtime_repeats_s = [item.total_s for item in runs]
        reference = generated.run_mnl_torch(case, max_iter=120)
        addition = _backend_payload(result, reference, case.parameter_names)
        addition["runtime_repeats_s"] = result.runtime_repeats_s
        _insert_backend(row["backends"], addition)
        row["consistent"] = _mark_json_loglikes(row["backends"], case.data.n_obs)
        _write(path, rows)
        print(f"[torch-choice] synthetic MNL {row['case']}: {result.total_s}", flush=True)


def add_synthetic_nl(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        meta_dict = row.get("results", row.get("generated"))
        meta = generated.GeneratedSpec(
            case=row["case"],
            model="nl",
            n_obs=int(meta_dict["N"]),
            n_alternatives=int(meta_dict["J"]),
            n_variables=int(meta_dict["K"]),
            rho=float(meta_dict["rho"]),
        )
        mnl_case = generated.build_mnl_case(meta, SEED)
        case = generated.build_nested_case(mnl_case, SEED)
        result = run_nested(
            case.data,
            case.spec,
            case.alternatives,
            case.beta_names,
            case.nests,
            case.initial_values,
            case.lambda_names,
            lambda_min=LAMBDA_MIN,
            max_iter=120,
        )
        reference = nested_real.run_torch(case, max_iter=120)
        addition = _backend_payload(result, reference, case.parameter_names)
        _insert_backend(row["backends"], addition)
        row["consistent"] = _mark_json_loglikes(row["backends"], case.data.n_obs)
        _write(path, rows)
        print(f"[torch-choice] synthetic NL {row['case']}: {result.total_s}", flush=True)


def _wide_to_long(
    frame: pd.DataFrame,
    alternatives: list[str],
    choice_column: str,
    variables: list[str],
    column_name,
) -> pd.DataFrame:
    rows = []
    for obs_index, source in frame.iterrows():
        chosen = str(source[choice_column])
        for alternative in alternatives:
            row = {
                "obs_id": obs_index + 1,
                "alt": str(alternative),
                "choice": chosen == str(alternative),
            }
            for variable in variables:
                row[variable] = source[column_name(variable, alternative)]
            rows.append(row)
    return pd.DataFrame(rows)


def load_mlogit_aligned(dataset: str) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(ROOT / "data" / "small" / f"mlogit_{dataset}" / "data.csv")
    if dataset == "catsup":
        variables = ["disp", "feat", "price"]
        aligned = _wide_to_long(raw, ["heinz41", "heinz32", "heinz28", "hunts32"], "choice", variables, lambda variable, alternative: f"{variable}.{alternative}")
    elif dataset == "cracker":
        variables = ["disp", "feat", "price"]
        aligned = _wide_to_long(raw, ["sunshine", "kleebler", "nabisco", "private"], "choice", variables, lambda variable, alternative: f"{variable}.{alternative}")
    elif dataset == "electricity":
        variables = ["pf", "cl", "loc", "wk", "tod", "seas"]
        aligned = _wide_to_long(raw, ["1", "2", "3", "4"], "choice", variables, lambda variable, alternative: f"{variable}{alternative}")
    elif dataset == "fishing":
        variables = ["price", "catch"]
        aligned = _wide_to_long(raw, ["beach", "pier", "boat", "charter"], "mode", variables, lambda variable, alternative: f"{variable}.{alternative}")
    elif dataset == "hc":
        variables = ["ich", "och"]
        aligned = _wide_to_long(raw, ["gcc", "ecc", "erc", "hpc", "gc", "ec", "er"], "depvar", variables, lambda variable, alternative: f"{variable}.{alternative}")
    elif dataset == "heating":
        variables = ["ic", "oc"]
        aligned = _wide_to_long(raw, ["gc", "gr", "ec", "er", "hp"], "depvar", variables, lambda variable, alternative: f"{variable}.{alternative}")
    elif dataset == "mode":
        variables = ["cost", "time"]
        aligned = _wide_to_long(raw, ["car", "carpool", "bus", "rail"], "choice", variables, lambda variable, alternative: f"{variable}.{alternative}")
    elif dataset == "modecanada":
        variables = ["cost", "ivt", "ovt", "freq"]
        aligned = pd.DataFrame({"obs_id": raw["case"], "alt": raw["alt"], "choice": raw["choice"].astype(bool)})
        for variable in variables:
            aligned[variable] = raw[variable]
    elif dataset == "nox":
        variables = ["post", "vcost", "kcost"]
        aligned = pd.DataFrame({"obs_id": raw["chid"], "alt": raw["alt"], "choice": raw["choice"].astype(bool), "availability": raw["available"].astype(bool)})
        for variable in variables:
            aligned[variable] = raw[variable]
    elif dataset == "risky_transport":
        variables = ["cost", "risk", "seats", "noise", "crowdness", "convloc", "clientele"]
        aligned = pd.DataFrame({"obs_id": raw["chid"], "alt": raw["mode"], "choice": raw["choice"].astype(bool)})
        for variable in variables:
            aligned[variable] = raw[variable]
    elif dataset == "train":
        variables = ["price", "time", "change", "comfort"]
        aligned = _wide_to_long(raw, ["A", "B"], "choice", variables, lambda variable, alternative: f"{variable}_{alternative}")
    else:
        raise ValueError(dataset)
    for variable in variables:
        aligned[variable] = pd.to_numeric(aligned[variable], errors="coerce")
    aligned = aligned.dropna(subset=["obs_id", "alt", "choice", *variables]).reset_index(drop=True)
    return aligned, variables


def _real_mnl_case(case_name: str):
    if case_name == "swissmetro_mnl":
        _, data, base_spec, _ = swissmetro_mnl.load_biogeme_swissmetro(10719)
        names = base_spec.parameter_names
        initial = {name: 0.0 for name in names}
        return data, swissmetro_mnl.spec_with_initials(base_spec, initial), initial, 200
    if case_name == "nhts_2022_mnl":
        case = nhts_mnl.make_nhts_2022_case(None)
        return case.data, case.spec, case.initial_values, 500
    if case_name.startswith("biogeme_public_"):
        key = case_name.removeprefix("biogeme_public_")
        case = public_mnl.CASE_BUILDERS[key](None)
        return case.data, case.spec, case.initial_values, 500
    if case_name.startswith("mlogit_"):
        key = case_name.removeprefix("mlogit_")
        aligned, variables = load_mlogit_aligned(key)
        if key in {"fishing", "modecanada"}:
            data, spec, names = mlogit_estimators.make_case(key, aligned)
            return data, spec, {name: 0.0 for name in names}, 200
        data, spec, _ = mlogit_battery.make_torch_case(aligned, variables, max_iter=180)
        names = [f"B_{variable.upper()}" for variable in variables]
        return data, spec, {name: 0.0 for name in names}, 180
    raise ValueError(case_name)


def _run_aligned_torchdcm_mnl(data, spec, initial_values, max_iter: int):
    """Run TorchDCM with the same warm-up and timing scope as Torch-Choice."""
    model = MultinomialLogit(spec, max_iter=max_iter, tolerance_grad=1e-7)
    data = data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    initial = torch.as_tensor(
        [initial_values.get(name, 0.0) for name in compiled.free_names],
        dtype=torch.float64,
    )

    # Both PyTorch packages receive one untimed forward/backward evaluation so
    # their reported optimizer times exclude one-time kernel initialization.
    warmup = initial.clone().requires_grad_(True)
    (-model.loglike(warmup, data, compiled)).backward()

    params = initial.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=max_iter,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = -model.loglike(params, data, compiled)
        loss.backward()
        return loss

    estimate_start = time.perf_counter()
    optimizer.step(closure)
    estimate_s = time.perf_counter() - estimate_start
    final = params.detach().clone()
    final_loglike = float(model.loglike(final, data, compiled).detach())
    covariance_start = time.perf_counter()
    hessian = torch.autograd.functional.hessian(
        lambda value: model.loglike(value, data, compiled), final
    )
    covariance = torch.linalg.pinv(-hessian.detach(), hermitian=True).cpu().numpy()
    covariance_s = time.perf_counter() - covariance_start
    probabilities = model.predict_proba(data, final, compiled).detach().cpu().numpy()
    return SimpleNamespace(
        backend="torchdcm",
        available=True,
        total_s=estimate_s + covariance_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=final_loglike,
        params={
            name: float(final[index])
            for index, name in enumerate(compiled.free_names)
        },
        covariance=covariance,
        probabilities=probabilities,
        message="",
    )


def refresh_real_mnl_torchdcm(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        data, spec, initial, max_iter = _real_mnl_case(row["case"])
        result = _run_aligned_torchdcm_mnl(
            data, spec, initial, max_iter=max_iter
        )
        backend_row = {
            "backend": "torchdcm",
            "available": True,
            "total_s": result.total_s,
            "estimate_s": result.estimate_s,
            "covariance_s": result.covariance_s,
            "loglike": result.loglike,
            "worse_loglike": False,
            "message": "",
            "raw": json.dumps(
                {
                    "backend": "torchdcm",
                    "available": True,
                    "total_s": result.total_s,
                    "estimate_s": result.estimate_s,
                    "covariance_s": result.covariance_s,
                    "loglike": result.loglike,
                }
            ),
        }
        row.setdefault("backend_rows", {})["torchdcm"] = backend_row
        row.setdefault("solver_status", {})["torchdcm"] = {
            "status": "ok",
            "seconds": result.total_s,
            "loglike": result.loglike,
            "worse_loglike": False,
            "backend": "torchdcm",
        }
        label = _mark_json_loglikes(
            list(row["backend_rows"].values()), int(data.n_obs)
        )
        for solver, status in row["solver_status"].items():
            backend_name = "torch_choice" if solver == "torch-choice" else solver
            backend = row["backend_rows"].get(backend_name)
            if backend and status.get("status") == "ok":
                status["loglike"] = backend.get("loglike")
                status["worse_loglike"] = backend.get("worse_loglike", False)
        row["consistent"] = label
        row["runtime_policy"] = runtime_policy_metadata()
        _write(path, rows)
        print(
            f"[torchdcm-aligned] real MNL {row['case']}: {result.total_s}",
            flush=True,
        )
    path.with_suffix(".md").write_text(
        solver_matrix.render_markdown(rows), encoding="utf-8"
    )


def add_real_mnl(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        data, spec, initial, max_iter = _real_mnl_case(row["case"])
        result = run_mnl(data, spec, initial, max_iter=max_iter)
        backend_row = {
            "backend": "torch_choice",
            "available": bool(result.available),
            "total_s": result.total_s,
            "estimate_s": result.estimate_s,
            "covariance_s": result.covariance_s,
            "loglike": result.loglike,
            "worse_loglike": False,
            "message": result.message,
            "raw": json.dumps({
                "backend": "torch_choice",
                "available": bool(result.available),
                "total_s": result.total_s,
                "estimate_s": result.estimate_s,
                "covariance_s": result.covariance_s,
                "loglike": result.loglike,
            }),
        }
        row.setdefault("backend_rows", {})["torch_choice"] = backend_row
        row.setdefault("solver_status", {})["torch-choice"] = (
            {
                "status": "ok",
                "seconds": result.total_s,
                "loglike": result.loglike,
                "worse_loglike": False,
                "backend": "torch_choice",
            }
            if result.available
            else {"status": "failed", "backend": "torch_choice", "message": result.message}
        )
        n_obs = int(data.n_obs)
        row["n_obs"] = n_obs
        for backend in row["backend_rows"].values():
            if backend.get("available") and backend.get("loglike") is None:
                raw = str(backend.get("raw", ""))
                if raw.lstrip().startswith("{"):
                    try:
                        value = json.loads(raw).get("loglike")
                        backend["loglike"] = float(value) if value is not None else None
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                else:
                    parts = raw.split()
                    if len(parts) >= 6:
                        try:
                            backend["loglike"] = float(parts[5])
                        except ValueError:
                            pass
        label = _mark_json_loglikes(list(row["backend_rows"].values()), n_obs)
        for solver, status in row["solver_status"].items():
            backend_name = "torch_choice" if solver == "torch-choice" else solver
            backend = row["backend_rows"].get(backend_name)
            if backend and status.get("status") == "ok":
                status["loglike"] = backend.get("loglike")
                status["worse_loglike"] = backend.get("worse_loglike", False)
        row["consistent"] = label
        row["runtime_policy"] = runtime_policy_metadata()
        _write(path, rows)
        print(f"[torch-choice] real MNL {row['case']}: {result.total_s}", flush=True)


def normalize_real_mnl_timing_fields(path: Path) -> None:
    """Expose estimation and covariance timing stored in legacy raw fields."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        for backend_name in ("torchdcm", "torch_choice"):
            backend = row["backend_rows"][backend_name]
            if "estimate_s" in backend and "covariance_s" in backend:
                continue
            raw = json.loads(backend["raw"])
            backend["estimate_s"] = raw.get(
                "estimate_s", raw.get("estimate_seconds")
            )
            backend["covariance_s"] = raw.get(
                "covariance_s", raw.get("covariance_seconds")
            )
    _write(path, rows)
    path.with_suffix(".md").write_text(
        solver_matrix.render_markdown(rows), encoding="utf-8"
    )


REAL_NL_KEYS = {
    "swissmetro_nested": "swissmetro",
    "lpmc_nested": "lpmc",
    "nhts_2022_nested": "nhts",
    "parking_nested": "parking",
    "airline_nested": "airline",
    "mlogit_catsup_nested": "mlogit_catsup",
    "mlogit_cracker_nested": "mlogit_cracker",
    "mlogit_electricity_nested": "mlogit_electricity",
    "mlogit_fishing_nested": "mlogit_fishing",
    "mlogit_hc_nested": "mlogit_hc",
    "mlogit_heating_nested": "mlogit_heating",
    "mlogit_mode_nested": "mlogit_mode",
}


def add_real_nl(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        builder_key = REAL_NL_KEYS[row["case"]]
        if builder_key.startswith("mlogit_"):
            dataset = builder_key.removeprefix("mlogit_")
            aligned, variables = load_mlogit_aligned(dataset)
            long_df, beta_names = mlogit_battery.make_design_long(aligned, variables)
            long_df["alt"] = long_df["alt"].astype(str)
            alternatives = [str(alternative) for alternative in pd.unique(long_df["alt"])]
            case = nested_real.nested_case_from_design_long(
                case=f"mlogit_{dataset}_nested",
                data_label=nested_real.MLOGIT_DATA_LABELS[dataset],
                model_label="Nested logit",
                source=f"R mlogit::{dataset}",
                long_df=long_df,
                alternatives=alternatives,
                beta_names=beta_names,
                nests=nested_real.MLOGIT_NESTS[dataset],
            )
        else:
            case = nested_real.CASE_BUILDERS[builder_key](None)
        result = run_nested(
            case.data,
            case.spec,
            case.alternatives,
            case.beta_names,
            case.nests,
            case.initial_values,
            case.lambda_names,
            lambda_min=LAMBDA_MIN,
            max_iter=200,
        )
        reference = nested_real.run_torch(case, max_iter=200)
        addition = _backend_payload(result, reference, case.parameter_names)
        _insert_backend(row["backends"], addition)
        label = _mark_json_loglikes(row["backends"], case.data.n_obs)
        row["consistent"] = None if label == "N.A." else label == "Yes"
        row["runtime_policy"] = runtime_policy_metadata()
        _write(path, rows)
        print(f"[torch-choice] real NL {row['case']}: {result.total_s}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchdcm-table7-only", action="store_true")
    parser.add_argument("--normalize-table7-only", action="store_true")
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=["3", "4", "7", "8"],
        default=["3", "4", "7", "8"],
    )
    args = parser.parse_args()
    if args.normalize_table7_only:
        normalize_real_mnl_timing_fields(
            RESULTS / "solver_attempt_matrix_mnl_single_core.json"
        )
        return
    if args.torchdcm_table7_only:
        refresh_real_mnl_torchdcm(
            RESULTS / "solver_attempt_matrix_mnl_single_core.json"
        )
        return
    if "3" in args.tables:
        add_synthetic_mnl(RESULTS / "synthetic_mnl_single_core.json")
    if "4" in args.tables:
        add_synthetic_nl(RESULTS / "generated_choice_battery_table4_nl.json")
    if "7" in args.tables:
        add_real_mnl(RESULTS / "solver_attempt_matrix_mnl_single_core.json")
    if "8" in args.tables:
        add_real_nl(RESULTS / "nested_real_battery_single_core.json")


if __name__ == "__main__":
    main()
