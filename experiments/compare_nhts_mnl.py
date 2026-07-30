from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark_runtime import estimation_covariance_total
from torchdcm import Beta, ChoiceDataset, MultinomialLogit, UtilitySpec
from mnl_generic_backends import (
    make_design_long,
    run_gmnl_generic,
    run_mlogit_generic,
    run_scipy_mle,
    run_xlogit_generic,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
APOLLO_SCRIPT = ROOT / "experiments" / "apollo" / "R" / "run_generic_mnl.R"


@dataclass
class BenchmarkCase:
    case: str
    dataset_id: str
    model_name: str
    df: pd.DataFrame
    data: ChoiceDataset
    spec: UtilitySpec
    alternatives: list[str]
    feature_names: list[str]
    feature_columns: dict[str, dict[str, str]]
    availability_columns: dict[str, str]
    parameter_names: list[str]
    initial_values: dict[str, float]
    source: str


@dataclass
class BackendResult:
    backend: str
    available: bool
    total_s: float | None = None
    estimate_s: float | None = None
    covariance_s: float | None = None
    loglike: float | None = None
    params: dict[str, float] | None = None
    covariance: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    message: str = ""


def ensure_nhts_2022_zip() -> Path:
    archive = ROOT / "data" / "raw" / "nhts_2022" / "csv.zip"
    if not archive.exists():
        raise FileNotFoundError(
            f"Archived NHTS input not found: {archive}. "
            "Restore data/raw/nhts_2022/csv.zip from the repository checkout."
        )
    return archive


def load_nhts_2022_trip_data(n_obs: int | None = None) -> pd.DataFrame:
    archive = ensure_nhts_2022_zip()
    usecols = [
        "TRIPMODE",
        "TRPMILES",
        "HHFAMINC",
        "HHVEHCNT",
        "NUMADLT",
        "URBAN",
        "R_AGE",
        "WORKER",
    ]
    with zipfile.ZipFile(archive) as zf:
        with zf.open("tripv2pub.csv") as fh:
            raw = pd.read_csv(fh, usecols=usecols)
    if n_obs:
        raw = raw.head(n_obs).copy()
    return raw


def make_nhts_2022_case(n_obs: int | None = None) -> BenchmarkCase:
    raw = load_nhts_2022_trip_data(n_obs)
    tripmode = raw["TRIPMODE"].replace({1: "AUTO", 2: "WALK", 3: "BIKE", 4: "TRANSIT", 5: "OTHER", 6: "OTHER", 7: "OTHER"})
    keep = (
        tripmode.isin(["AUTO", "WALK", "BIKE", "TRANSIT", "OTHER"])
        & (raw["TRPMILES"] >= 0)
        & (raw["HHFAMINC"] > 0)
        & (raw["HHVEHCNT"] >= 0)
        & (raw["NUMADLT"] > 0)
        & (raw["URBAN"] > 0)
        & (raw["R_AGE"] > 0)
        & (raw["WORKER"] > 0)
    )
    clean = raw.loc[keep].copy()
    tripmode = tripmode.loc[keep]

    alternatives = ["AUTO", "WALK", "BIKE", "TRANSIT", "OTHER"]
    features = ["log_miles", "veh_per_adult", "urban"]
    df = pd.DataFrame({"obs_id": np.arange(len(clean)), "choice": tripmode.to_numpy()})
    df["log_miles_base"] = np.log1p(clean["TRPMILES"].to_numpy(dtype=float))
    df["veh_per_adult_base"] = clean["HHVEHCNT"].to_numpy(dtype=float) / clean["NUMADLT"].to_numpy(dtype=float)
    df["urban_base"] = (clean["URBAN"].to_numpy(dtype=int) == 1).astype(float)
    for feature in ["log_miles_base", "veh_per_adult_base"]:
        values = df[feature].to_numpy(dtype=float)
        df[feature] = (values - values.mean()) / values.std(ddof=0)

    feature_columns: dict[str, dict[str, str]] = {feature: {} for feature in features}
    availability_columns: dict[str, str] = {}
    for alt in alternatives:
        alt_key = alt.lower()
        for feature in features:
            source_column = f"{feature}_base"
            target_column = f"{feature}_{alt_key}"
            df[target_column] = df[source_column]
            feature_columns[feature][alt] = target_column
        df[f"avail_{alt_key}"] = True
        availability_columns[alt] = f"avail_{alt_key}"
    df = df.drop(columns=[f"{feature}_base" for feature in features])

    data = ChoiceDataset.from_wide(
        df,
        alternatives=alternatives,
        choice="choice",
        variables=feature_columns,
        availability=availability_columns,
        obs_id="obs_id",
    )

    spec = UtilitySpec()
    parameter_names: list[str] = []
    spec.utility("AUTO", Beta("ASC_AUTO", fixed=True))
    for alt in alternatives[1:]:
        expr = Beta(f"ASC_{alt}")
        parameter_names.append(f"ASC_{alt}")
        for feature in features:
            name = f"B_{feature.upper()}_{alt}"
            expr = expr + Beta(name) * feature
            parameter_names.append(name)
        spec.utility(alt, expr)

    return BenchmarkCase(
        case="nhts_2022_mode",
        dataset_id="nhts_2022",
        model_name="NHTS 2022 trip mode MNL",
        df=df,
        data=data,
        spec=spec,
        alternatives=alternatives,
        feature_names=features,
        feature_columns=feature_columns,
        availability_columns=availability_columns,
        parameter_names=parameter_names,
        initial_values={name: 0.0 for name in parameter_names},
        source="NHTS 2022 official public-use CSV zip",
    )


def run_torch(case: BenchmarkCase, max_iter: int) -> BackendResult:
    model = MultinomialLogit(case.spec, max_iter=max_iter, tolerance_grad=1e-9)
    data = case.data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    params = compiled.free_initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=model.max_iter,
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
    hessian = torch.autograd.functional.hessian(lambda p: model.loglike(p, data, compiled), final)
    covariance = torch.linalg.pinv(-hessian.detach(), hermitian=True).cpu().numpy()
    covariance_s = time.perf_counter() - covariance_start
    probabilities = model.predict_proba(data, final, compiled).detach().cpu().numpy()
    return BackendResult(
        backend="torchdcm",
        available=True,
        total_s=estimate_s + covariance_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=loglike,
        params={name: float(final[i].detach().cpu()) for i, name in enumerate(compiled.free_names)},
        covariance=covariance,
        probabilities=probabilities,
    )


def run_biogeme(case: BenchmarkCase) -> BackendResult:
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

    df = case.df.copy()
    code_by_alt = {alt: i + 1 for i, alt in enumerate(case.alternatives)}
    df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
    df = df.drop(columns=["choice"])
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    database = db.Database(f"torchdcm_{case.case}", df)
    betas = {name: BioBeta(name, case.initial_values.get(name, 0.0), None, None, 0) for name in case.parameter_names}
    utility = {}
    availability = {}
    for alt, code in code_by_alt.items():
        expr = 0
        if alt != "AUTO":
            expr += betas[f"ASC_{alt}"]
            for feature in case.feature_names:
                expr += betas[f"B_{feature.upper()}_{alt}"] * Variable(case.feature_columns[feature][alt])
        utility[code] = expr
        availability[code] = Variable(case.availability_columns[alt])
    logprob = models.loglogit(utility, availability, Variable("choice_code"))
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.model_name = f"torchdcm_{case.case}_{case.data.n_obs}"
    biogeme.biogeme_parameters.set_value("save_iterations", False)
    total_start = time.perf_counter()
    estimate_start = time.perf_counter()
    estimates = biogeme.estimate()
    estimate_s = time.perf_counter() - estimate_start
    covariance_start = time.perf_counter()
    covariance_obj = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
    covariance_s = time.perf_counter() - covariance_start
    if hasattr(covariance_obj, "loc"):
        covariance = np.asarray(covariance_obj.loc[case.parameter_names, case.parameter_names], dtype=float)
    else:
        covariance = np.asarray(covariance_obj, dtype=float)
    beta_values = estimates.get_beta_values()
    total_s = time.perf_counter() - total_start
    return BackendResult(
        backend="biogeme",
        available=True,
        total_s=total_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=float(estimates.final_log_likelihood),
        params={name: float(beta_values[name]) for name in case.parameter_names},
        covariance=covariance,
    )


def run_apollo(case: BenchmarkCase) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo", available=False, message="Rscript not found.")
    if not APOLLO_SCRIPT.exists():
        return BackendResult(backend="apollo", available=False, message=f"Missing Apollo script: {APOLLO_SCRIPT}")
    try:
        df, spec = make_apollo_inputs(case)
        with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_{case.case}_") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.csv"
            spec_path = tmp_path / "spec.json"
            output_path = tmp_path / "apollo_result.json"
            df.to_csv(data_path, index=False)
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
                return BackendResult(backend="apollo", available=False, total_s=total_s, message=(proc.stderr or proc.stdout).strip())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            covariance_names = payload.get("covariance_names") or case.parameter_names
            covariance = reorder_covariance(payload.get("covariance"), covariance_names, case.parameter_names)
            estimate_s = payload.get("timing", {}).get("estimate_seconds")
            covariance_s = payload.get("timing", {}).get("covariance_seconds")
            return BackendResult(
                backend="apollo",
                available=True,
                total_s=estimation_covariance_total(estimate_s, covariance_s),
                estimate_s=estimate_s,
                covariance_s=covariance_s,
                loglike=float(payload["loglike"]),
                params={name: float(payload["estimates"][name]) for name in case.parameter_names},
                covariance=covariance,
                message=f"apollo_version={payload.get('apollo_version')}",
            )
    except Exception as exc:
        return BackendResult(backend="apollo", available=False, message=f"{type(exc).__name__}: {exc}")


def run_scipy(case: BenchmarkCase):
    return run_scipy_mle(case.data, case.spec, case.initial_values, target_names=case.parameter_names, maxiter=500)


def run_mlogit(case: BenchmarkCase):
    return run_mlogit_generic(make_case_design_long(case), case.parameter_names)


def run_gmnl(case: BenchmarkCase):
    return run_gmnl_generic(make_case_design_long(case), case.parameter_names)


def run_xlogit(case: BenchmarkCase):
    return run_xlogit_generic(make_case_design_long(case), case.parameter_names)


def make_case_design_long(case: BenchmarkCase) -> pd.DataFrame:
    utility_columns: dict[str, dict[str, str | float]] = {}
    for alt in case.alternatives:
        terms: dict[str, str | float] = {}
        if alt != "AUTO":
            terms[f"ASC_{alt}"] = 1.0
            for feature in case.feature_names:
                terms[f"B_{feature.upper()}_{alt}"] = case.feature_columns[feature][alt]
        utility_columns[alt] = terms
    return make_design_long(
        case.df,
        case.alternatives,
        "choice",
        utility_columns,
        case.availability_columns,
        case.parameter_names,
    )


def make_apollo_inputs(case: BenchmarkCase) -> tuple[pd.DataFrame, dict]:
    code_by_alt = {alt: i + 1 for i, alt in enumerate(case.alternatives)}
    df = case.df.copy()
    df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
    df = df.drop(columns=["choice"])
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    utility = {}
    for alt in case.alternatives:
        if alt == "AUTO":
            variables = {}
            asc_name = None
        else:
            variables = {
                f"B_{feature.upper()}_{alt}": case.feature_columns[feature][alt]
                for feature in case.feature_names
            }
            asc_name = f"ASC_{alt}"
        utility[alt] = {
            "code": code_by_alt[alt],
            "asc": asc_name,
            "availability": case.availability_columns[alt],
            "variables": variables,
        }
    spec = {
        "model_name": f"apollo_{case.case}_{case.data.n_obs}",
        "alternatives": case.alternatives,
        "choice_col": "choice_code",
        "parameters": {name: case.initial_values.get(name, 0.0) for name in case.parameter_names},
        "utility": utility,
    }
    return df, spec


def reorder_covariance(covariance, source_names: list[str], target_names: list[str]) -> np.ndarray:
    matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
    return matrix.loc[target_names, target_names].to_numpy(dtype=float)


def predict_probabilities(case: BenchmarkCase, params: dict[str, float]) -> np.ndarray:
    model = MultinomialLogit(case.spec)
    vector = torch.as_tensor([params[name] for name in case.parameter_names], dtype=torch.float64)
    return model.predict_proba(case.data, vector).detach().cpu().numpy()


def compare(results: list[BackendResult], case: BenchmarkCase) -> None:
    for result in results:
        if result.available and result.probabilities is None:
            result.probabilities = predict_probabilities(case, result.params or {})
    ref = next(result for result in results if result.backend == "torchdcm" and result.available)
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        result.max_param_diff = max(abs(result.params[name] - ref.params[name]) for name in case.parameter_names)  # type: ignore[attr-defined]
        result.max_prob_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))  # type: ignore[attr-defined]
        result.max_cov_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
        result.max_se_diff = float(np.max(np.abs(np.sqrt(np.diag(result.covariance)) - np.sqrt(np.diag(ref.covariance)))))  # type: ignore[attr-defined]


def is_consistent(results: list[BackendResult]) -> bool:
    for result in results:
        if result.backend == "torchdcm" or not result.available:
            continue
        if abs(getattr(result, "ll_diff")) > 1e-4:
            return False
        if getattr(result, "max_param_diff") > 5e-4:
            return False
        if getattr(result, "max_prob_diff") > 2e-4:
            return False
        if getattr(result, "max_se_diff") > 3e-2:
            return False
    return True


def result_payload(case: BenchmarkCase, results: list[BackendResult]) -> dict:
    compare(results, case)
    return {
        "case": case.case,
        "dataset_id": case.dataset_id,
        "model_name": case.model_name,
        "n_obs": case.data.n_obs,
        "n_alternatives": len(case.alternatives),
        "n_parameters": len(case.parameter_names),
        "source": case.source,
        "parameters": case.parameter_names,
        "consistent": is_consistent(results),
        "backends": [
            {
                "backend": result.backend,
                "available": result.available,
                "total_s": result.total_s,
                "estimate_s": result.estimate_s,
                "covariance_s": result.covariance_s,
                "loglike": result.loglike,
                "ll_diff": getattr(result, "ll_diff", None),
                "max_param_diff": getattr(result, "max_param_diff", None),
                "max_prob_diff": getattr(result, "max_prob_diff", None),
                "max_cov_diff": getattr(result, "max_cov_diff", None),
                "max_se_diff": getattr(result, "max_se_diff", None),
                "message": result.message,
            }
            for result in results
        ],
    }


def print_payload(payload: dict) -> None:
    print(f"case: {payload['case']}")
    print(f"dataset_id: {payload['dataset_id']}")
    print(f"model: {payload['model_name']}")
    print(f"n_obs: {payload['n_obs']}")
    print(f"n_alternatives: {payload['n_alternatives']}")
    print(f"n_parameters: {payload['n_parameters']}")
    print(f"consistent: {payload['consistent']}")
    print("alignment:")
    print("  benchmark_mode: full_estimation")
    print(f"  data_source: {payload['source']}")
    print("  choice_set: AUTO, WALK, BIKE, TRANSIT, OTHER from NHTS TRIPMODE")
    print("  features: alternative-specific coefficients on standardized trip/person/household covariates")
    print("  initial_values: zeros shared across TorchDCM, SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit")
    print("  covariance: classic inverse observed information / Rao-Cramer")
    print("  reference: torchdcm")
    print()
    print(
        f"{'backend':<12}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'param_diff':>14}{'prob_diff':>14}{'cov_diff':>14}{'se_diff':>14}"
    )
    for result in payload["backends"]:
        if result["available"]:
            print(
                f"{result['backend']:<12}{str(result['available']):>10}"
                f"{_fmt(result['total_s']):>12}{_fmt(result['estimate_s']):>12}{_fmt(result['covariance_s']):>12}"
                f"{result['loglike']:>18.10f}{result['ll_diff']:>14.3e}{result['max_param_diff']:>14.3e}"
                f"{result['max_prob_diff']:>14.3e}{result['max_cov_diff']:>14.3e}{result['max_se_diff']:>14.3e}"
            )
        else:
            print(f"{result['backend']:<12}{str(result['available']):>10}  {result['message']}")


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    case = make_nhts_2022_case(args.n_obs)
    results = [
        run_torch(case, args.max_iter),
        run_scipy(case),
        run_biogeme(case),
        run_apollo(case),
        run_mlogit(case),
        run_gmnl(case),
        run_xlogit(case),
    ]
    payload = result_payload(case, results)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_payload(payload)


if __name__ == "__main__":
    main()
