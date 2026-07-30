from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from benchmark_runtime import estimation_covariance_total
from torchdcm import MultinomialLogit


ROOT = Path(__file__).resolve().parents[1]
MLOGIT_GENERIC_SCRIPT = ROOT / "experiments" / "mlogit" / "R" / "run_generic_mnl.R"
GMNL_GENERIC_SCRIPT = ROOT / "experiments" / "mlogit" / "R" / "run_generic_gmnl_mnl.R"


def unavailable(backend: str, message: str, seconds: float | None = None):
    return SimpleNamespace(
        backend=backend,
        available=False,
        total_s=seconds,
        seconds=seconds,
        estimate_s=None,
        estimate_seconds=None,
        covariance_s=None,
        covariance_seconds=None,
        loglike=None,
        params=None,
        covariance=None,
        probabilities=None,
        message=message,
    )


def make_design_long(
    df: pd.DataFrame,
    alternatives: list[str],
    choice_col: str,
    utility_columns: dict[str, dict[str, str | float | int]],
    availability_columns: dict[str, str] | None,
    parameter_names: list[str],
) -> pd.DataFrame:
    rows = []
    for _, source_row in df.iterrows():
        chosen = source_row[choice_col]
        obs = source_row["obs_id"]
        for alt in alternatives:
            row = {"obs_id": obs, "alt": alt, "choice": chosen == alt}
            if availability_columns:
                row["availability"] = bool(source_row[availability_columns[alt]])
            else:
                row["availability"] = True
            terms = utility_columns.get(alt, {})
            for name in parameter_names:
                value = terms.get(name, 0.0)
                row[name] = float(source_row[value]) if isinstance(value, str) else float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def run_scipy_mle(data, spec, initial_values: dict[str, float], target_names: list[str] | None = None, maxiter: int = 200):
    model = MultinomialLogit(spec)
    compiled = model.compile(data)
    names = list(compiled.free_names)
    x0 = np.asarray([initial_values.get(name, 0.0) for name in names], dtype=float)
    design = compiled.design.detach().cpu().numpy()
    fixed_design = compiled.fixed_design.detach().cpu().numpy()
    fixed_values = compiled.fixed_values.detach().cpu().numpy()
    obs_ptr = data.obs_ptr.detach().cpu().numpy()
    chosen_row = data.chosen_row.detach().cpu().numpy()
    availability = data.availability.detach().cpu().numpy().astype(bool)
    weights = data.weights.detach().cpu().numpy()
    fixed_utility = fixed_design @ fixed_values if fixed_values.size else 0.0

    def value_and_grad(beta: np.ndarray):
        utility = design @ beta + fixed_utility
        loglike = 0.0
        grad = np.zeros_like(beta)
        for obs in range(data.n_obs):
            start_row = obs_ptr[obs]
            end_row = obs_ptr[obs + 1]
            rows = slice(start_row, end_row)
            mask = availability[rows]
            available_utility = utility[rows][mask]
            max_utility = np.max(available_utility)
            exp_utility = np.exp(available_utility - max_utility)
            denom = exp_utility.sum()
            probs_available = exp_utility / denom
            probs = np.zeros(end_row - start_row, dtype=float)
            probs[mask] = probs_available
            loglike += weights[obs] * (utility[chosen_row[obs]] - max_utility - np.log(denom))
            chosen_local = chosen_row[obs] - start_row
            y = np.zeros(end_row - start_row, dtype=float)
            y[chosen_local] = 1.0
            grad += weights[obs] * design[rows].T @ (y - probs)
        return -loglike, -grad

    estimate_start = time.perf_counter()
    opt = minimize(
        fun=lambda beta: value_and_grad(beta)[0],
        x0=x0,
        jac=lambda beta: value_and_grad(beta)[1],
        method="BFGS",
        options={"gtol": 1e-7, "maxiter": maxiter},
    )
    estimate_s = time.perf_counter() - estimate_start
    params = dict(zip(names, opt.x))
    covariance_start = time.perf_counter()
    vector = torch.as_tensor([params[name] for name in names], dtype=torch.float64)
    hessian = torch.autograd.functional.hessian(lambda p: model.loglike(p, data, compiled), vector)
    covariance = torch.linalg.pinv(-hessian.detach(), hermitian=True).cpu().numpy()
    covariance_s = time.perf_counter() - covariance_start
    if target_names:
        index = [names.index(name) for name in target_names]
        covariance = covariance[np.ix_(index, index)]
    return SimpleNamespace(
        backend="scipy_bfgs",
        available=True,
        total_s=estimate_s + covariance_s,
        seconds=estimate_s + covariance_s,
        estimate_s=estimate_s,
        estimate_seconds=estimate_s,
        covariance_s=covariance_s,
        covariance_seconds=covariance_s,
        loglike=-float(opt.fun),
        params=params,
        covariance=covariance,
        probabilities=None,
        message="" if opt.success else str(opt.message),
    )


def run_mlogit_generic(long_df: pd.DataFrame, parameter_names: list[str]):
    return run_r_generic("mlogit", MLOGIT_GENERIC_SCRIPT, long_df, parameter_names)


def run_gmnl_generic(long_df: pd.DataFrame, parameter_names: list[str]):
    return run_r_generic("gmnl", GMNL_GENERIC_SCRIPT, long_df, parameter_names)


def run_r_generic(backend: str, script: Path, long_df: pd.DataFrame, parameter_names: list[str]):
    rscript = shutil.which("Rscript")
    if rscript is None:
        return unavailable(backend, "Rscript not found.")
    if not script.exists():
        return unavailable(backend, f"Missing script: {script}")
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_{backend}_generic_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        spec_path = tmp_path / "spec.json"
        result_path = tmp_path / "result.json"
        long_df.to_csv(data_path, index=False)
        spec_path.write_text(json.dumps({"parameters": parameter_names}, indent=2), encoding="utf-8")
        env = os.environ.copy()
        r_user_lib = str(Path.home() / "R" / "site-library")
        existing = env.get("R_LIBS_USER")
        env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
        command = [rscript, str(script), "--data", str(data_path), "--spec", str(spec_path), "--result-output", str(result_path)]
        wall_start = time.perf_counter()
        proc = subprocess.run(command, text=True, capture_output=True, env=env)
        wall = time.perf_counter() - wall_start
        if proc.returncode != 0:
            return unavailable(backend, (proc.stderr or proc.stdout).strip(), wall)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        covariance = reorder_covariance(payload["covariance"], payload["covariance_names"], parameter_names)
        estimate_s = float(payload.get("estimate_seconds", 0.0))
        covariance_s = float(payload.get("covariance_seconds", 0.0))
        return SimpleNamespace(
            backend=backend,
            available=True,
            total_s=estimation_covariance_total(estimate_s, covariance_s),
            seconds=estimation_covariance_total(estimate_s, covariance_s),
            estimate_s=estimate_s,
            estimate_seconds=estimate_s,
            covariance_s=covariance_s,
            covariance_seconds=covariance_s,
            loglike=float(payload["loglike"]),
            params={name: float(payload["params"][name]) for name in parameter_names},
            covariance=covariance,
            probabilities=None,
            message="",
        )


def run_xlogit_generic(long_df: pd.DataFrame, parameter_names: list[str]):
    try:
        from xlogit import MultinomialLogit as XlogitMultinomialLogit
    except ImportError as exc:
        return unavailable("xlogit", f"xlogit unavailable: {exc}")
    try:
        model = XlogitMultinomialLogit()
        estimate_start = time.perf_counter()
        model.fit(
            X=long_df[parameter_names],
            y=long_df["choice"].astype(int),
            varnames=parameter_names,
            alts=long_df["alt"],
            ids=long_df["obs_id"],
            avail=long_df["availability"].astype(int) if "availability" in long_df.columns else None,
            fit_intercept=False,
            verbose=0,
            maxiter=2000,
        )
        estimate_s = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance = np.asarray(model.covariance, dtype=float)
        covariance_s = time.perf_counter() - covariance_start
        params = dict(zip(model.coeff_names, model.coeff_))
        covariance = reorder_covariance(covariance, list(model.coeff_names), parameter_names)
        return SimpleNamespace(
            backend="xlogit",
            available=True,
            total_s=estimate_s + covariance_s,
            seconds=estimate_s + covariance_s,
            estimate_s=estimate_s,
            estimate_seconds=estimate_s,
            covariance_s=covariance_s,
            covariance_seconds=covariance_s,
            loglike=float(model.loglikelihood),
            params={name: float(params[name]) for name in parameter_names},
            covariance=covariance,
            probabilities=None,
            message="",
        )
    except Exception as exc:
        return unavailable("xlogit", f"{type(exc).__name__}: {exc}")


def reorder_covariance(covariance, source_names: list[str], target_names: list[str]) -> np.ndarray:
    matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
    return matrix.loc[target_names, target_names].to_numpy(dtype=float)
