from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

from benchmark_runtime import estimation_covariance_total
from torchdcm import Beta, ChoiceDataset, MultinomialLogit, UtilitySpec
from mnl_generic_backends import run_gmnl_generic, run_scipy_mle, run_xlogit_generic
from torch_choice_backend import run_mnl as run_torch_choice_mnl


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_ROOT = ROOT / "data" / "small"
R_SCRIPT = ROOT / "experiments" / "mlogit" / "R" / "run_mlogit_dataset_mnl.R"
APOLLO_SCRIPT = ROOT / "experiments" / "apollo" / "R" / "run_generic_mnl.R"

DEFAULT_DATASETS = [
    "catsup",
    "cracker",
    "electricity",
    "fishing",
    "hc",
    "heating",
    "mode",
    "modecanada",
    "nox",
    "risky_transport",
    "train",
]


def run_r_reference(dataset: str) -> tuple[pd.DataFrame | None, dict]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None, {"backend": "mlogit", "available": False, "message": "Rscript not found."}

    with tempfile.TemporaryDirectory(prefix=f"torchdcm_{dataset}_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        result_path = tmp_path / "result.json"
        env = os.environ.copy()
        r_user_lib = str(Path.home() / "R" / "site-library")
        existing = env.get("R_LIBS_USER")
        env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
        command = [
            rscript,
            str(R_SCRIPT),
            "--dataset",
            dataset,
            "--data-input",
            str(DATA_ROOT / f"mlogit_{dataset}" / "data.csv"),
            "--data-output",
            str(data_path),
            "--result-output",
            str(result_path),
        ]
        start = time.perf_counter()
        proc = subprocess.run(command, text=True, capture_output=True, env=env)
        wall_s = time.perf_counter() - start
        if proc.returncode != 0:
            return None, {
                "backend": "mlogit",
                "available": False,
                "total_seconds": wall_s,
                "message": (proc.stderr or proc.stdout).strip(),
            }
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["available"] = True
        payload["wall_seconds"] = wall_s
        return pd.read_csv(data_path), payload


def run_torch(df: pd.DataFrame, variables: list[str], max_iter: int) -> dict:
    alt_order = list(pd.unique(df["alt"]))
    data = ChoiceDataset.from_long(
        df,
        obs_id="obs_id",
        alt_id="alt",
        choice="choice",
        variables=variables,
        availability="availability" if "availability" in df.columns else None,
        alt_order=alt_order,
    )
    spec = UtilitySpec()
    for alt in alt_order:
        utility = None
        for variable in variables:
            term = Beta(f"B_{variable.upper()}") * variable
            utility = term if utility is None else utility + term
        spec.utility(alt, utility)

    model = MultinomialLogit(spec, max_iter=max_iter)
    data = data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    initial = torch.zeros(len(compiled.free_names), dtype=torch.float64)
    # Match Torch-Choice by excluding one-time tensor/autograd initialization.
    warmup = initial.clone().requires_grad_(True)
    (-model.loglike(warmup, data, compiled)).backward()
    params = initial.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
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
    loglike = float(model.loglike(final, data, compiled).detach().cpu())
    covariance_start = time.perf_counter()
    hessian = torch.autograd.functional.hessian(
        lambda p: model.loglike(p, data, compiled), final
    )
    torch.linalg.pinv(-hessian.detach(), hermitian=True)
    covariance_s = time.perf_counter() - covariance_start
    return {
        "backend": "torchdcm",
        "available": True,
        "total_seconds": estimate_s + covariance_s,
        "estimate_seconds": estimate_s,
        "covariance_seconds": covariance_s,
        "loglike": loglike,
        "n_obs": data.n_obs,
        "n_rows": data.n_rows,
        "n_parameters": len(compiled.free_names),
        "parameters": compiled.free_names,
    }


def make_torch_case(df: pd.DataFrame, variables: list[str], max_iter: int):
    alt_order = list(pd.unique(df["alt"]))
    data = ChoiceDataset.from_long(
        df,
        obs_id="obs_id",
        alt_id="alt",
        choice="choice",
        variables=variables,
        availability="availability" if "availability" in df.columns else None,
        alt_order=alt_order,
    )
    spec = UtilitySpec()
    for alt in alt_order:
        utility = None
        for variable in variables:
            term = Beta(f"B_{variable.upper()}") * variable
            utility = term if utility is None else utility + term
        spec.utility(alt, utility)
    model = MultinomialLogit(spec, max_iter=max_iter)
    return data, spec, model


def make_design_long(df: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, list[str]]:
    long_df = df[["obs_id", "alt", "choice"] + (["availability"] if "availability" in df.columns else [])].copy()
    if "availability" not in long_df.columns:
        long_df["availability"] = True
    parameter_names = [f"B_{variable.upper()}" for variable in variables]
    for variable, parameter in zip(variables, parameter_names):
        long_df[parameter] = df[variable].astype(float)
    return long_df, parameter_names


def result_to_dict(result) -> dict:
    return {
        "backend": result.backend,
        "available": bool(result.available),
        "total_seconds": getattr(result, "total_s", getattr(result, "seconds", None)),
        "estimate_seconds": getattr(result, "estimate_s", getattr(result, "estimate_seconds", None)),
        "covariance_seconds": getattr(result, "covariance_s", getattr(result, "covariance_seconds", None)),
        "loglike": getattr(result, "loglike", None),
        "params": getattr(result, "params", None),
        "message": getattr(result, "message", ""),
    }


def run_biogeme(df: pd.DataFrame, variables: list[str]) -> dict:
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
        return {"backend": "biogeme", "available": False, "message": f"Biogeme unavailable: {exc}"}

    try:
        wide_df, alt_columns, choice_col = long_to_biogeme_wide(df, variables)
        database = db.Database("torchdcm_mlogit_generic_mnl", wide_df)
        parameter_names = [f"B_{variable.upper()}" for variable in variables]
        betas = {name: BioBeta(name, 0.0, None, None, 0) for name in parameter_names}
        utility = {}
        availability = {}
        for alt, columns in alt_columns.items():
            expr = 0
            for variable in variables:
                expr += betas[f"B_{variable.upper()}"] * Variable(columns[variable])
            utility[columns["code"]] = expr
            availability[columns["code"]] = Variable(columns["availability"])
        logprob = models.loglogit(utility, availability, Variable(choice_col))
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = f"torchdcm_mlogit_generic_{len(wide_df)}"
        biogeme.biogeme_parameters.set_value("save_iterations", False)
        total_start = time.perf_counter()
        estimate_start = time.perf_counter()
        estimates = biogeme.estimate()
        estimate_s = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance_obj = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
        covariance_s = time.perf_counter() - covariance_start
        covariance = covariance_to_ordered_array(covariance_obj, parameter_names)
        beta_values = estimates.get_beta_values()
        return {
            "backend": "biogeme",
            "available": True,
            "total_seconds": time.perf_counter() - total_start,
            "estimate_seconds": estimate_s,
            "covariance_seconds": covariance_s,
            "loglike": float(estimates.final_log_likelihood),
            "parameters": parameter_names,
            "params": {name: float(beta_values[name]) for name in parameter_names},
            "covariance": covariance.tolist(),
        }
    except Exception as exc:
        return {"backend": "biogeme", "available": False, "message": f"{type(exc).__name__}: {exc}"}


def run_apollo(df: pd.DataFrame, variables: list[str]) -> dict:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {"backend": "apollo", "available": False, "message": "Rscript not found."}
    if not APOLLO_SCRIPT.exists():
        return {"backend": "apollo", "available": False, "message": f"Missing Apollo script: {APOLLO_SCRIPT}"}

    try:
        wide_df, alt_columns, choice_col = long_to_biogeme_wide(df, variables)
        parameter_names = [f"B_{variable.upper()}" for variable in variables]
        spec = make_apollo_spec(wide_df, alt_columns, variables, parameter_names, choice_col, model_token="mlogit_generic")
        with tempfile.TemporaryDirectory(prefix="torchdcm_apollo_generic_") as tmp:
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
            command = [
                rscript,
                str(APOLLO_SCRIPT),
                "--data",
                str(data_path),
                "--spec",
                str(spec_path),
                "--output",
                str(output_path),
            ]
            total_start = time.perf_counter()
            proc = subprocess.run(command, text=True, capture_output=True, env=env)
            total_s = time.perf_counter() - total_start
            if proc.returncode != 0:
                return {
                    "backend": "apollo",
                    "available": False,
                    "total_seconds": total_s,
                    "message": (proc.stderr or proc.stdout).strip(),
                }
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            covariance_names = payload.get("covariance_names") or parameter_names
            covariance = reorder_covariance(payload.get("covariance"), covariance_names, parameter_names)
            estimate_seconds = payload.get("timing", {}).get("estimate_seconds")
            covariance_seconds = payload.get("timing", {}).get("covariance_seconds")
            return {
                "backend": "apollo",
                "available": True,
                "total_seconds": estimation_covariance_total(estimate_seconds, covariance_seconds),
                "estimate_seconds": estimate_seconds,
                "covariance_seconds": covariance_seconds,
                "loglike": float(payload["loglike"]),
                "parameters": parameter_names,
                "params": {name: float(payload["estimates"][name]) for name in parameter_names},
                "covariance": covariance.tolist() if covariance is not None else None,
                "message": f"apollo_version={payload.get('apollo_version')}",
            }
    except Exception as exc:
        return {"backend": "apollo", "available": False, "message": f"{type(exc).__name__}: {exc}"}


def make_apollo_spec(
    wide_df: pd.DataFrame,
    alt_columns: dict[object, dict[str, str | int]],
    variables: list[str],
    parameter_names: list[str],
    choice_col: str,
    model_token: str,
    asc_names: dict[object, str | None] | None = None,
) -> dict:
    asc_names = asc_names or {}
    alternatives = list(alt_columns.keys())
    return {
        "model_name": f"apollo_{model_token}_{len(wide_df)}",
        "alternatives": [str(alt) for alt in alternatives],
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


def reorder_covariance(covariance, source_names: list[str], target_names: list[str]):
    if covariance is None:
        return None
    matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
    return matrix.loc[target_names, target_names].to_numpy(dtype=float)


def long_to_biogeme_wide(df: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, str | int]], str]:
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
    return wide_df, alt_columns, "choice_code"


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


def covariance_to_ordered_array(covariance_obj, parameter_names: list[str]):
    if hasattr(covariance_obj, "loc"):
        return covariance_obj.loc[parameter_names, parameter_names].to_numpy(dtype=float)
    return pd.DataFrame(covariance_obj).to_numpy(dtype=float)


def benchmark_dataset(dataset: str, max_iter: int) -> dict:
    df, ref = run_r_reference(dataset)
    result = {
        "case": dataset,
        "dataset_id": f"mlogit_{dataset}",
        "model": "MNL",
        "reference": "mlogit",
        "mlogit": ref,
    }
    if df is None or not ref.get("available"):
        result["status"] = "failed"
        result["consistent"] = "No"
        return result
    variables_raw = ref["variables"]
    variables = [variables_raw] if isinstance(variables_raw, str) else list(variables_raw)
    try:
        torch_result = run_torch(df, variables, max_iter=max_iter)
    except Exception as exc:
        result["status"] = "failed"
        result["torchdcm"] = {"available": False, "message": str(exc)}
        result["consistent"] = "No"
        return result
    data, spec, _ = make_torch_case(df, variables, max_iter=max_iter)
    parameter_names = [f"B_{variable.upper()}" for variable in variables]
    scipy_result = run_scipy_mle(data, spec, {name: 0.0 for name in parameter_names}, target_names=parameter_names)
    torch_choice_result = result_to_dict(
        run_torch_choice_mnl(
            data,
            spec,
            {name: 0.0 for name in parameter_names},
            max_iter=max_iter,
        )
    )
    long_design, design_parameters = make_design_long(df, variables)
    biogeme_result = run_biogeme(df, variables)
    apollo_result = run_apollo(df, variables)
    gmnl_result = result_to_dict(run_gmnl_generic(long_design, design_parameters))
    xlogit_result = result_to_dict(run_xlogit_generic(long_design, design_parameters))
    ll_diff = torch_result["loglike"] - float(ref["loglike"])
    biogeme_ll_diff = (
        torch_result["loglike"] - float(biogeme_result["loglike"])
        if biogeme_result.get("available") and biogeme_result.get("loglike") is not None
        else None
    )
    apollo_ll_diff = (
        torch_result["loglike"] - float(apollo_result["loglike"])
        if apollo_result.get("available") and apollo_result.get("loglike") is not None
        else None
    )
    result["torchdcm"] = torch_result
    result["torch_choice"] = torch_choice_result
    result["scipy_bfgs"] = result_to_dict(scipy_result)
    result["biogeme"] = biogeme_result
    result["apollo"] = apollo_result
    result["gmnl"] = gmnl_result
    result["xlogit"] = xlogit_result
    result["n_obs"] = int(torch_result["n_obs"])
    result["n_rows"] = int(torch_result["n_rows"])
    result["n_parameters"] = int(torch_result["n_parameters"])
    result["ll_diff"] = ll_diff
    result["biogeme_ll_diff"] = biogeme_ll_diff
    result["apollo_ll_diff"] = apollo_ll_diff
    biogeme_consistent = not biogeme_result.get("available") or (biogeme_ll_diff is not None and abs(biogeme_ll_diff) <= 1e-4)
    apollo_consistent = not apollo_result.get("available") or (apollo_ll_diff is not None and abs(apollo_ll_diff) <= 1e-4)
    scipy_consistent = abs(torch_result["loglike"] - float(scipy_result.loglike)) <= 1e-4
    torch_choice_consistent = (
        not torch_choice_result.get("available")
        or abs(torch_result["loglike"] - float(torch_choice_result["loglike"])) <= 1e-4
    )
    gmnl_consistent = not gmnl_result.get("available") or abs(torch_result["loglike"] - float(gmnl_result["loglike"])) <= 1e-4
    xlogit_consistent = not xlogit_result.get("available") or abs(torch_result["loglike"] - float(xlogit_result["loglike"])) <= 1e-4
    result["consistent"] = "Yes" if abs(ll_diff) <= 1e-4 and torch_choice_consistent and scipy_consistent and biogeme_consistent and apollo_consistent and gmnl_consistent and xlogit_consistent else "No"
    result["status"] = "ok"
    return result


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# R mlogit Dataset Battery",
        "",
        "Baseline MNL with generic coefficients and no alternative-specific constants.",
        "",
        "| case | N | K | TorchDCM total_s | mlogit total_s | Biogeme total_s | Apollo total_s | mlogit LL diff | Biogeme LL diff | Apollo LL diff | Consistent? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        torch_row = row.get("torchdcm", {})
        ref = row.get("mlogit", {})
        biogeme = row.get("biogeme", {})
        apollo = row.get("apollo", {})
        lines.append(
            "| {case} | {n} | {k} | {torch_s} | {mlogit_s} | {biogeme_s} | {apollo_s} | {ll_diff} | {biogeme_ll_diff} | {apollo_ll_diff} | {consistent} |".format(
                case=row["case"],
                n=row.get("n_obs", "NA"),
                k=row.get("n_parameters", "NA"),
                torch_s=_fmt(torch_row.get("total_seconds")),
                mlogit_s=_fmt(ref.get("total_seconds")),
                biogeme_s=_fmt(biogeme.get("total_seconds")),
                apollo_s=_fmt(apollo.get("total_seconds")),
                ll_diff=_sci(row.get("ll_diff")),
                biogeme_ll_diff=_sci(row.get("biogeme_ll_diff")),
                apollo_ll_diff=_sci(row.get("apollo_ll_diff")),
                consistent=row.get("consistent", "No"),
            )
        )
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    for row in results:
        if row.get("status") == "ok":
            continue
        lines.append(
            f"- `{row['case']}`: "
            f"{row.get('mlogit', {}).get('message') or row.get('torchdcm', {}).get('message') or row.get('biogeme', {}).get('message')}"
        )
    return "\n".join(lines)


def _fmt(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "NA"


def _sci(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2e}"
    return "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--max-iter", type=int, default=120)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    results = [benchmark_dataset(dataset, max_iter=args.max_iter) for dataset in args.datasets]
    json_path = RESULTS / f"mlogit_dataset_battery_{args.profile}.json"
    md_path = RESULTS / f"mlogit_dataset_battery_{args.profile}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    for row in results:
        print(f"{row['case']}: {row.get('status')} consistent={row.get('consistent')}")


if __name__ == "__main__":
    main()
