from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xlogit import MultinomialLogit as XlogitMultinomialLogit

from benchmark_runtime import estimation_covariance_total
from torchdcm import Beta, ChoiceDataset, MultinomialLogit, UtilitySpec
from mnl_generic_backends import run_scipy_mle


APOLLO_SCRIPT = Path(__file__).resolve().parent / "apollo" / "R" / "run_generic_mnl.R"


@dataclass
class BackendResult:
    backend: str
    available: bool
    seconds: float | None = None
    estimate_seconds: float | None = None
    covariance_seconds: float | None = None
    loglike: float | None = None
    params: dict[str, float] | None = None
    covariance: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    message: str = ""


def run_mlogit_export(dataset: str) -> tuple[pd.DataFrame, BackendResult]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript not found.")
    script = Path(__file__).resolve().parent / "mlogit" / "R" / "run_mlogit_mnl.R"
    with tempfile.TemporaryDirectory(prefix="torchdcm_mlogit_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        result_path = tmp_path / "result.json"
        env = os.environ.copy()
        r_user_lib = str(Path.home() / "R" / "site-library")
        existing = env.get("R_LIBS_USER")
        env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
        cmd = [
            rscript,
            str(script),
            "--dataset",
            dataset,
            "--data-output",
            str(data_path),
            "--result-output",
            str(result_path),
        ]
        wall_start = time.perf_counter()
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        wall = time.perf_counter() - wall_start
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip())
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = BackendResult(
            backend="mlogit",
            available=True,
            seconds=wall,
            estimate_seconds=float(payload["estimate_seconds"]),
            covariance_seconds=float(payload["covariance_seconds"]),
            loglike=float(payload["loglike"]),
            params=_map_mlogit_params(dataset, {key: float(value) for key, value in payload["params"].items()}),
            covariance=_map_mlogit_covariance(dataset, payload["covariance_names"], np.asarray(payload["covariance"], dtype=float)),
        )
        return pd.read_csv(data_path), result


def run_gmnl(dataset: str) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="gmnl", available=False, message="Rscript not found.")
    script = Path(__file__).resolve().parent / "mlogit" / "R" / "run_gmnl_mnl.R"
    if not script.exists():
        return BackendResult(backend="gmnl", available=False, message=f"Missing script: {script}")
    with tempfile.TemporaryDirectory(prefix="torchdcm_gmnl_") as tmp:
        tmp_path = Path(tmp)
        result_path = tmp_path / "result.json"
        env = os.environ.copy()
        r_user_lib = str(Path.home() / "R" / "site-library")
        existing = env.get("R_LIBS_USER")
        env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
        cmd = [
            rscript,
            str(script),
            "--dataset",
            dataset,
            "--result-output",
            str(result_path),
        ]
        wall_start = time.perf_counter()
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        wall = time.perf_counter() - wall_start
        if proc.returncode != 0:
            return BackendResult(backend="gmnl", available=False, seconds=wall, message=(proc.stderr or proc.stdout).strip())
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return BackendResult(
            backend="gmnl",
            available=True,
            seconds=wall,
            estimate_seconds=float(payload["estimate_seconds"]),
            covariance_seconds=float(payload["covariance_seconds"]),
            loglike=float(payload["loglike"]),
            params=_map_gmnl_params(dataset, {key: float(value) for key, value in payload["params"].items()}),
            covariance=_map_gmnl_covariance(dataset, payload["covariance_names"], np.asarray(payload["covariance"], dtype=float)),
        )


def run_xlogit(dataset: str, df: pd.DataFrame) -> BackendResult:
    if dataset == "modecanada":
        return BackendResult(
            backend="xlogit",
            available=False,
            message="xlogit MultinomialLogit requires consistent alternatives in long format; ModeCanada has ragged choice sets.",
        )
    if dataset != "fishing":
        return BackendResult(backend="xlogit", available=False, message=f"Unsupported xlogit case: {dataset}")
    try:
        model = XlogitMultinomialLogit()
        estimate_start = time.perf_counter()
        model.fit(
            X=df[["price", "catch"]],
            y=df["choice"].astype(int),
            varnames=["price", "catch"],
            alts=df["alt"],
            ids=df["obs_id"],
            fit_intercept=True,
            base_alt="beach",
            verbose=0,
            maxiter=2000,
        )
        estimate_seconds = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance = np.asarray(model.covariance, dtype=float)
        covariance_seconds = time.perf_counter() - covariance_start
        params = _map_xlogit_params(dataset, dict(zip(model.coeff_names, model.coeff_)))
        return BackendResult(
            backend="xlogit",
            available=True,
            seconds=estimate_seconds + covariance_seconds,
            estimate_seconds=estimate_seconds,
            covariance_seconds=covariance_seconds,
            loglike=float(model.loglikelihood),
            params=params,
            covariance=_map_xlogit_covariance(dataset, list(model.coeff_names), covariance),
        )
    except Exception as exc:
        return BackendResult(backend="xlogit", available=False, message=str(exc))


def run_biogeme(dataset: str, df: pd.DataFrame, names: list[str]) -> BackendResult:
    tmp_root = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "torchdcm_matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(tmp_root / "torchdcm_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme import models
        from biogeme.expressions import Beta as BioBeta
        from biogeme.expressions import Variable
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult(backend="biogeme", available=False, message=f"Biogeme unavailable: {exc}")
    try:
        variables = variables_for_dataset(dataset)
        wide_df, alt_columns, choice_col, alt_order = long_to_biogeme_wide(df, variables)
        database = db.Database(f"torchdcm_mlogit_{dataset}", wide_df)
        betas = {name: BioBeta(name, 0.0, None, None, 0) for name in names}
        utility = {}
        availability = {}
        for alt in alt_order:
            columns = alt_columns[alt]
            expr = 0
            asc_name = f"ASC_{str(alt).upper()}"
            if asc_name in betas:
                expr += betas[asc_name]
            for variable in variables:
                expr += betas[f"B_{variable.upper()}"] * Variable(columns[variable])
            utility[columns["code"]] = expr
            availability[columns["code"]] = Variable(columns["availability"])
        logprob = models.loglogit(utility, availability, Variable(choice_col))
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = f"torchdcm_mlogit_{dataset}_{len(wide_df)}"
        biogeme.biogeme_parameters.set_value("save_iterations", False)
        total_start = time.perf_counter()
        estimate_start = time.perf_counter()
        estimates = biogeme.estimate()
        estimate_seconds = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance_obj = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
        covariance_seconds = time.perf_counter() - covariance_start
        covariance = covariance_to_ordered_array(covariance_obj, names)
        beta_values = estimates.get_beta_values()
        return BackendResult(
            backend="biogeme",
            available=True,
            seconds=time.perf_counter() - total_start,
            estimate_seconds=estimate_seconds,
            covariance_seconds=covariance_seconds,
            loglike=float(estimates.final_log_likelihood),
            params={name: float(beta_values[name]) for name in names},
            covariance=covariance,
        )
    except Exception as exc:
        return BackendResult(backend="biogeme", available=False, message=f"{type(exc).__name__}: {exc}")


def run_apollo(dataset: str, df: pd.DataFrame, names: list[str]) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo", available=False, message="Rscript not found.")
    if not APOLLO_SCRIPT.exists():
        return BackendResult(backend="apollo", available=False, message=f"Missing Apollo script: {APOLLO_SCRIPT}")
    try:
        variables = variables_for_dataset(dataset)
        wide_df, alt_columns, choice_col, alt_order = long_to_biogeme_wide(df, variables)
        asc_names = {alt: f"ASC_{str(alt).upper()}" if f"ASC_{str(alt).upper()}" in names else None for alt in alt_order}
        spec = make_apollo_spec(wide_df, alt_columns, variables, names, choice_col, dataset, asc_names)
        with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_{dataset}_") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.csv"
            spec_path = tmp_path / "spec.json"
            output_path = tmp_path / "apollo_result.json"
            wide_df.to_csv(data_path, index=False)
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            env = os.environ.copy()
            r_user_lib = str(Path.home() / "R" / "site-library")
            existing = env.get("R_LIBS_USER")
            env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
            cmd = [
                rscript,
                str(APOLLO_SCRIPT),
                "--data",
                str(data_path),
                "--spec",
                str(spec_path),
                "--output",
                str(output_path),
            ]
            wall_start = time.perf_counter()
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
            wall = time.perf_counter() - wall_start
            if proc.returncode != 0:
                return BackendResult(backend="apollo", available=False, seconds=wall, message=(proc.stderr or proc.stdout).strip())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            covariance_names = payload.get("covariance_names") or names
            covariance = reorder_covariance(payload.get("covariance"), covariance_names, names)
            estimate_seconds = float(payload.get("timing", {}).get("estimate_seconds", 0.0))
            covariance_seconds = float(payload.get("timing", {}).get("covariance_seconds", 0.0))
            return BackendResult(
                backend="apollo",
                available=True,
                seconds=estimation_covariance_total(estimate_seconds, covariance_seconds),
                estimate_seconds=estimate_seconds,
                covariance_seconds=covariance_seconds,
                loglike=float(payload["loglike"]),
                params={name: float(payload["estimates"][name]) for name in names},
                covariance=covariance,
                message=f"apollo_version={payload.get('apollo_version')}",
            )
    except Exception as exc:
        return BackendResult(backend="apollo", available=False, message=f"{type(exc).__name__}: {exc}")


def make_apollo_spec(
    wide_df: pd.DataFrame,
    alt_columns: dict[object, dict[str, str | int]],
    variables: list[str],
    parameter_names: list[str],
    choice_col: str,
    model_token: str,
    asc_names: dict[object, str | None],
) -> dict:
    return {
        "model_name": f"apollo_mlogit_{model_token}_{len(wide_df)}",
        "alternatives": [str(alt) for alt in alt_columns],
        "choice_col": choice_col,
        "parameters": {name: 0.0 for name in parameter_names},
        "utility": {
            str(alt): {
                "code": columns["code"],
                "asc": asc_names.get(alt),
                "availability": columns["availability"],
                "variables": {f"B_{variable.upper()}": columns[variable] for variable in variables},
            }
            for alt, columns in alt_columns.items()
        },
    }


def reorder_covariance(covariance, source_names: list[str], target_names: list[str]) -> np.ndarray:
    matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
    return matrix.loc[target_names, target_names].to_numpy(dtype=float)


def variables_for_dataset(dataset: str) -> list[str]:
    if dataset == "fishing":
        return ["price", "catch"]
    if dataset == "modecanada":
        return ["cost", "ivt", "ovt"]
    raise ValueError(dataset)


def long_to_biogeme_wide(df: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, str | int]], str, list[str]]:
    alt_order = list(pd.unique(df["alt"]))
    safe_by_alt = unique_safe_names(alt_order)
    code_by_alt = {alt: index + 1 for index, alt in enumerate(alt_order)}
    rows = []
    has_availability = "availability" in df.columns
    for obs_id, group in df.groupby("obs_id", sort=False):
        chosen = group.loc[group["choice"].astype(bool), "alt"]
        if chosen.empty:
            continue
        row = {"obs_id": obs_id, "choice_code": code_by_alt[chosen.iloc[0]]}
        group_by_alt = {alt: alt_group.iloc[0] for alt, alt_group in group.groupby("alt", sort=False)}
        for alt in alt_order:
            safe = safe_by_alt[alt]
            source = group_by_alt.get(alt)
            available = bool(source["availability"]) if source is not None and has_availability else source is not None
            row[f"avail_{safe}"] = int(available)
            for variable in variables:
                row[f"{variable}_{safe}"] = float(source[variable]) if source is not None else 0.0
        rows.append(row)
    wide_df = pd.DataFrame(rows)
    alt_columns = {
        alt: {
            "code": code_by_alt[alt],
            "availability": f"avail_{safe_by_alt[alt]}",
            **{variable: f"{variable}_{safe_by_alt[alt]}" for variable in variables},
        }
        for alt in alt_order
    }
    return wide_df, alt_columns, "choice_code", alt_order


def unique_safe_names(values: list[object]) -> dict[object, str]:
    used: set[str] = set()
    result = {}
    for value in values:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_").lower() or "alt"
        if base[0].isdigit():
            base = f"alt_{base}"
        candidate = base
        cursor = 2
        while candidate in used:
            candidate = f"{base}_{cursor}"
            cursor += 1
        used.add(candidate)
        result[value] = candidate
    return result


def covariance_to_ordered_array(covariance_obj, parameter_names: list[str]) -> np.ndarray:
    if hasattr(covariance_obj, "loc"):
        return np.asarray(covariance_obj.loc[parameter_names, parameter_names], dtype=float)
    return np.asarray(covariance_obj, dtype=float)


def make_case(dataset: str, df: pd.DataFrame) -> tuple[ChoiceDataset, UtilitySpec, list[str]]:
    if dataset == "fishing":
        alternatives = ["beach", "boat", "charter", "pier"]
        data = ChoiceDataset.from_long(
            df,
            obs_id="obs_id",
            alt_id="alt",
            choice="choice",
            variables=["price", "catch"],
            alt_order=alternatives,
        )
        spec = UtilitySpec()
        spec.utility("beach", Beta("B_PRICE") * "price" + Beta("B_CATCH") * "catch")
        spec.utility("boat", Beta("ASC_BOAT") + Beta("B_PRICE") * "price" + Beta("B_CATCH") * "catch")
        spec.utility("charter", Beta("ASC_CHARTER") + Beta("B_PRICE") * "price" + Beta("B_CATCH") * "catch")
        spec.utility("pier", Beta("ASC_PIER") + Beta("B_PRICE") * "price" + Beta("B_CATCH") * "catch")
        names = ["ASC_BOAT", "ASC_CHARTER", "ASC_PIER", "B_PRICE", "B_CATCH"]
        return data, spec, names
    if dataset == "modecanada":
        alternatives = list(pd.unique(df["alt"]))
        data = ChoiceDataset.from_long(
            df,
            obs_id="obs_id",
            alt_id="alt",
            choice="choice",
            variables=["cost", "ivt", "ovt"],
            alt_order=alternatives,
        )
        spec = UtilitySpec()
        for alt in alternatives:
            spec.utility(alt, Beta("B_COST") * "cost" + Beta("B_IVT") * "ivt" + Beta("B_OVT") * "ovt")
        names = ["B_COST", "B_IVT", "B_OVT"]
        return data, spec, names
    raise ValueError(f"Unknown dataset: {dataset}")


def run_torch(data: ChoiceDataset, spec: UtilitySpec, names: list[str]) -> BackendResult:
    model = MultinomialLogit(spec)
    data = data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    name_to_index = {name: index for index, name in enumerate(compiled.free_names)}
    initial = torch.zeros(len(compiled.free_names), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [initial],
        max_iter=model.max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = -model.loglike(initial, data, compiled)
        loss.backward()
        return loss

    estimate_start = time.perf_counter()
    optimizer.step(closure)
    estimate_seconds = time.perf_counter() - estimate_start
    params = initial.detach().clone()
    loglike = float(model.loglike(params, data, compiled).detach().cpu())
    covariance_start = time.perf_counter()
    hessian = torch.autograd.functional.hessian(lambda p: model.loglike(p, data, compiled), params)
    covariance = torch.linalg.pinv(-hessian.detach(), hermitian=True).cpu().numpy()
    covariance_seconds = time.perf_counter() - covariance_start
    probabilities = model.predict_proba(data, params, compiled).detach().cpu().numpy()
    ordered_indices = [name_to_index[name] for name in names]
    return BackendResult(
        backend="torchdcm",
        available=True,
        seconds=estimate_seconds + covariance_seconds,
        estimate_seconds=estimate_seconds,
        covariance_seconds=covariance_seconds,
        loglike=loglike,
        params={name: float(params[name_to_index[name]].detach().cpu()) for name in names},
        covariance=covariance[np.ix_(ordered_indices, ordered_indices)],
        probabilities=probabilities,
    )


def _map_mlogit_params(dataset: str, params: dict[str, float]) -> dict[str, float]:
    if dataset == "fishing":
        return {
            "ASC_BOAT": params["(Intercept):boat"],
            "ASC_CHARTER": params["(Intercept):charter"],
            "ASC_PIER": params["(Intercept):pier"],
            "B_PRICE": params["price"],
            "B_CATCH": params["catch"],
        }
    if dataset == "modecanada":
        return {
            "B_COST": params["cost"],
            "B_IVT": params["ivt"],
            "B_OVT": params["ovt"],
        }
    raise ValueError(dataset)


def _map_mlogit_covariance(dataset: str, names: list[str], covariance: np.ndarray) -> np.ndarray:
    if dataset == "fishing":
        target = ["(Intercept):boat", "(Intercept):charter", "(Intercept):pier", "price", "catch"]
    elif dataset == "modecanada":
        target = ["cost", "ivt", "ovt"]
    else:
        raise ValueError(dataset)
    index = [names.index(name) for name in target]
    return covariance[np.ix_(index, index)]


def _map_gmnl_params(dataset: str, params: dict[str, float]) -> dict[str, float]:
    if dataset == "fishing":
        return {
            "ASC_BOAT": params["boat:(intercept)"],
            "ASC_CHARTER": params["charter:(intercept)"],
            "ASC_PIER": params["pier:(intercept)"],
            "B_PRICE": params["price"],
            "B_CATCH": params["catch"],
        }
    if dataset == "modecanada":
        return {
            "B_COST": params["cost"],
            "B_IVT": params["ivt"],
            "B_OVT": params["ovt"],
        }
    raise ValueError(dataset)


def _map_gmnl_covariance(dataset: str, names: list[str], covariance: np.ndarray) -> np.ndarray:
    if dataset == "fishing":
        target = ["boat:(intercept)", "charter:(intercept)", "pier:(intercept)", "price", "catch"]
    elif dataset == "modecanada":
        target = ["cost", "ivt", "ovt"]
    else:
        raise ValueError(dataset)
    index = [names.index(name) for name in target]
    return covariance[np.ix_(index, index)]


def _map_xlogit_params(dataset: str, params: dict[str, float]) -> dict[str, float]:
    if dataset == "fishing":
        return {
            "ASC_BOAT": params["_intercept.boat"],
            "ASC_CHARTER": params["_intercept.charter"],
            "ASC_PIER": params["_intercept.pier"],
            "B_PRICE": params["price"],
            "B_CATCH": params["catch"],
        }
    raise ValueError(dataset)


def _map_xlogit_covariance(dataset: str, names: list[str], covariance: np.ndarray) -> np.ndarray:
    if dataset == "fishing":
        target = ["_intercept.boat", "_intercept.charter", "_intercept.pier", "price", "catch"]
    else:
        raise ValueError(dataset)
    index = [names.index(name) for name in target]
    return covariance[np.ix_(index, index)]


def compare(results: list[BackendResult], names: list[str]) -> None:
    ref = next(result for result in results if result.backend == "torchdcm")
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        result.max_abs_param_diff = max(abs(result.params[name] - ref.params[name]) for name in names)  # type: ignore[attr-defined]
        result.max_abs_covariance_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
        se = np.sqrt(np.diag(result.covariance))
        ref_se = np.sqrt(np.diag(ref.covariance))
        result.max_abs_se_diff = float(np.max(np.abs(se - ref_se)))  # type: ignore[attr-defined]
        t_values = np.asarray([result.params[name] for name in names]) / se
        ref_t = np.asarray([ref.params[name] for name in names]) / ref_se
        result.max_abs_t_diff = float(np.max(np.abs(t_values - ref_t)))  # type: ignore[attr-defined]


def print_results(dataset: str, data: ChoiceDataset, names: list[str], results: list[BackendResult]) -> None:
    compare(results, names)
    print(f"case: mlogit_{dataset}_mnl")
    print(f"n_obs: {data.n_obs}")
    print("alignment:")
    print("  benchmark_mode: full_estimation")
    print("  data_source: R package mlogit built-in data")
    print("  estimator_reference: R mlogit::mlogit")
    print("  initial_values: zeros for TorchDCM; mlogit default start")
    print("  covariance: classic inverse observed information / vcov")
    print("  reference: torchdcm")
    print()
    print(
        f"{'backend':<12}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'param_diff':>14}{'cov_diff':>14}{'se_diff':>14}{'t_diff':>14}"
    )
    for result in results:
        if result.available:
            print(
                f"{result.backend:<12}{str(result.available):>10}"
                f"{_fmt_seconds(result.seconds):>12}"
                f"{_fmt_seconds(result.estimate_seconds):>12}"
                f"{_fmt_seconds(result.covariance_seconds):>12}"
                f"{result.loglike:>18.10f}"
                f"{getattr(result, 'll_diff'):>14.3e}"
                f"{getattr(result, 'max_abs_param_diff'):>14.3e}"
                f"{getattr(result, 'max_abs_covariance_diff'):>14.3e}"
                f"{getattr(result, 'max_abs_se_diff'):>14.3e}"
                f"{getattr(result, 'max_abs_t_diff'):>14.3e}"
            )
        else:
            print(
                f"{result.backend:<12}{str(result.available):>10}"
                f"{'NA':>12}{'NA':>12}{'NA':>12}{'NA':>18}{'NA':>14}{'NA':>14}{'NA':>14}{'NA':>14}{'NA':>14}  {result.message}"
            )
    print()
    print(f"{'parameter':<18}{'torch_beta':>14}{'mlogit_beta':>14}{'beta_diff':>14}{'torch_se':>14}{'mlogit_se':>14}{'torch_t':>14}{'mlogit_t':>14}")
    torch_result = results[0]
    mlogit_result = next(result for result in results if result.backend == "mlogit")
    torch_se = np.sqrt(np.diag(torch_result.covariance))
    mlogit_se = np.sqrt(np.diag(mlogit_result.covariance))
    for i, name in enumerate(names):
        torch_beta = torch_result.params[name]
        mlogit_beta = mlogit_result.params[name]
        print(
            f"{name:<18}{torch_beta:>14.6g}{mlogit_beta:>14.6g}{torch_beta - mlogit_beta:>14.3e}"
            f"{torch_se[i]:>14.6g}{mlogit_se[i]:>14.6g}{torch_beta / torch_se[i]:>14.6g}{mlogit_beta / mlogit_se[i]:>14.6g}"
        )


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fishing", "modecanada"], default="fishing")
    args = parser.parse_args()

    df, mlogit_result = run_mlogit_export(args.dataset)
    data, spec, names = make_case(args.dataset, df)
    torch_result = run_torch(data, spec, names)
    scipy_result = run_scipy_mle(data, spec, {name: 0.0 for name in names}, target_names=names)
    gmnl_result = run_gmnl(args.dataset)
    xlogit_result = run_xlogit(args.dataset, df)
    biogeme_result = run_biogeme(args.dataset, df, names)
    apollo_result = run_apollo(args.dataset, df, names)
    print_results(args.dataset, data, names, [torch_result, scipy_result, mlogit_result, biogeme_result, apollo_result, gmnl_result, xlogit_result])


if __name__ == "__main__":
    main()
