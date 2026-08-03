from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
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
from torch_choice_backend import run_mnl as run_torch_choice_mnl


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
APOLLO_SCRIPT = ROOT / "experiments" / "apollo" / "R" / "run_generic_mnl.R"


@dataclass
class CaseSpec:
    case: str
    dataset_id: str
    model_name: str
    df: pd.DataFrame
    data: ChoiceDataset
    spec: UtilitySpec
    alternatives: list[str]
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


def _read_dataset(dataset_id: str) -> pd.DataFrame:
    small_path = REPO_ROOT / "data" / "small" / dataset_id / "data.csv"
    raw_path = ROOT / "data" / "raw" / dataset_id / "data.csv"
    if small_path.exists():
        return pd.read_csv(small_path)
    if raw_path.exists():
        return pd.read_csv(raw_path)
    raise FileNotFoundError(f"Dataset not found: {dataset_id}")


def make_airline(n_obs: int | None) -> CaseSpec:
    raw = _read_dataset("biogeme_airline_itinerary")
    if n_obs:
        raw = raw.head(n_obs).copy()
    df = pd.DataFrame({"obs_id": np.arange(len(raw))})
    alternatives = ["ALT1", "ALT2", "ALT3"]
    choice_idx = raw[["BestAlternative_1", "BestAlternative_2", "BestAlternative_3"]].to_numpy().argmax(axis=1) + 1
    df["choice"] = [f"ALT{i}" for i in choice_idx]
    feature_columns = {"trip_time": {}, "fare": {}, "legroom": {}}
    availability_columns = {}
    for i, alt in enumerate(alternatives, start=1):
        df[f"trip_time_{alt.lower()}"] = raw[f"TripTimeHours_{i}"]
        df[f"fare_{alt.lower()}"] = raw[f"Fare_{i}"] / 100.0
        df[f"legroom_{alt.lower()}"] = raw[f"Legroom_{i}"]
        df[f"avail_{alt.lower()}"] = True
        feature_columns["trip_time"][alt] = f"trip_time_{alt.lower()}"
        feature_columns["fare"][alt] = f"fare_{alt.lower()}"
        feature_columns["legroom"][alt] = f"legroom_{alt.lower()}"
        availability_columns[alt] = f"avail_{alt.lower()}"
    data = ChoiceDataset.from_wide(
        df,
        alternatives=alternatives,
        choice="choice",
        variables=feature_columns,
        availability=availability_columns,
        obs_id="obs_id",
    )
    spec = UtilitySpec()
    spec.utility("ALT1", Beta("B_TRIP_TIME") * "trip_time" + Beta("B_FARE") * "fare" + Beta("B_LEGROOM") * "legroom")
    spec.utility("ALT2", Beta("ASC_ALT2") + Beta("B_TRIP_TIME") * "trip_time" + Beta("B_FARE") * "fare" + Beta("B_LEGROOM") * "legroom")
    spec.utility("ALT3", Beta("ASC_ALT3") + Beta("B_TRIP_TIME") * "trip_time" + Beta("B_FARE") * "fare" + Beta("B_LEGROOM") * "legroom")
    names = ["B_TRIP_TIME", "B_FARE", "B_LEGROOM", "ASC_ALT2", "ASC_ALT3"]
    return CaseSpec(
        case="airline",
        dataset_id="biogeme_airline_itinerary",
        model_name="Airline itinerary MNL",
        df=df,
        data=data,
        spec=spec,
        alternatives=alternatives,
        feature_columns=feature_columns,
        availability_columns=availability_columns,
        parameter_names=names,
        initial_values={name: 0.0 for name in names},
        source="Biogeme data page airline.dat",
    )


def make_parking(n_obs: int | None) -> CaseSpec:
    raw = _read_dataset("biogeme_parking_spain")
    if n_obs:
        raw = raw.head(n_obs).copy()
    alternatives = ["FSP", "PSP", "PUP"]
    df = pd.DataFrame({"obs_id": np.arange(len(raw)), "choice": raw["CHOICE"].map({1: "FSP", 2: "PSP", 3: "PUP"})})
    feature_columns = {"access_time": {}, "search_time": {}, "fee": {}}
    availability_columns = {}
    for i, alt in enumerate(alternatives, start=1):
        df[f"access_time_{alt.lower()}"] = raw[f"AT{i}"] / 10.0
        df[f"search_time_{alt.lower()}"] = raw[f"TD{i}"] / 10.0
        df[f"fee_{alt.lower()}"] = raw[f"FEE{i}"]
        df[f"avail_{alt.lower()}"] = True
        feature_columns["access_time"][alt] = f"access_time_{alt.lower()}"
        feature_columns["search_time"][alt] = f"search_time_{alt.lower()}"
        feature_columns["fee"][alt] = f"fee_{alt.lower()}"
        availability_columns[alt] = f"avail_{alt.lower()}"
    data = ChoiceDataset.from_wide(
        df,
        alternatives=alternatives,
        choice="choice",
        variables=feature_columns,
        availability=availability_columns,
        obs_id="obs_id",
    )
    spec = UtilitySpec()
    for alt in alternatives:
        asc = 0 if alt == "FSP" else Beta(f"ASC_{alt}")
        spec.utility(
            alt,
            asc + Beta("B_ACCESS_TIME") * "access_time" + Beta("B_SEARCH_TIME") * "search_time" + Beta("B_FEE") * "fee",
        )
    names = ["B_ACCESS_TIME", "B_SEARCH_TIME", "B_FEE", "ASC_PSP", "ASC_PUP"]
    return CaseSpec(
        case="parking",
        dataset_id="biogeme_parking_spain",
        model_name="Parking Spain MNL",
        df=df,
        data=data,
        spec=spec,
        alternatives=alternatives,
        feature_columns=feature_columns,
        availability_columns=availability_columns,
        parameter_names=names,
        initial_values={name: 0.0 for name in names},
        source="Biogeme data page parking.dat",
    )


def make_telephone(n_obs: int | None) -> CaseSpec:
    raw = _read_dataset("biogeme_telephone")
    if n_obs:
        raw = raw.head(n_obs).copy()
    alternatives = ["A1", "A2", "A3", "A4", "A5"]
    df = pd.DataFrame({"obs_id": np.arange(len(raw)), "choice": raw["choice"].map({i: f"A{i}" for i in range(1, 6)})})
    feature_columns = {"cost": {}}
    availability_columns = {}
    for i, alt in enumerate(alternatives, start=1):
        df[f"cost_{alt.lower()}"] = raw[f"cost{i}"] / 10.0
        df[f"avail_{alt.lower()}"] = raw[f"avail{i}"].astype(bool)
        feature_columns["cost"][alt] = f"cost_{alt.lower()}"
        availability_columns[alt] = f"avail_{alt.lower()}"
    data = ChoiceDataset.from_wide(
        df,
        alternatives=alternatives,
        choice="choice",
        variables=feature_columns,
        availability=availability_columns,
        obs_id="obs_id",
    )
    spec = UtilitySpec()
    for alt in alternatives:
        asc = 0 if alt == "A1" else Beta(f"ASC_{alt}")
        spec.utility(alt, asc + Beta("B_COST") * "cost")
    names = ["B_COST", "ASC_A2", "ASC_A3", "ASC_A4", "ASC_A5"]
    return CaseSpec(
        case="telephone",
        dataset_id="biogeme_telephone",
        model_name="Telephone service MNL",
        df=df,
        data=data,
        spec=spec,
        alternatives=alternatives,
        feature_columns=feature_columns,
        availability_columns=availability_columns,
        parameter_names=names,
        initial_values={name: 0.0 for name in names},
        source="Biogeme data page telephone.dat",
    )


def make_lpmc(n_obs: int | None) -> CaseSpec:
    raw_path = ROOT / "data" / "raw" / "lpmc_london" / "data.csv"
    raw = pd.read_csv(raw_path)
    if n_obs:
        raw = raw.head(n_obs).copy()
    alternatives = ["walk", "cycle", "pt", "drive"]
    choice_map = {1: "walk", 2: "cycle", 3: "pt", 4: "drive"}
    df = pd.DataFrame(
        {
            "obs_id": raw["trip_id"].astype(int),
            "choice": raw["travel_mode"].map(choice_map),
        }
    )
    df["time_walk"] = raw["dur_walking"]
    df["time_cycle"] = raw["dur_cycling"]
    df["time_pt"] = raw["dur_pt_access"] + raw["dur_pt_rail"] + raw["dur_pt_bus"] + raw["dur_pt_int"]
    df["time_drive"] = raw["dur_driving"]
    df["cost_walk"] = 0.0
    df["cost_cycle"] = 0.0
    df["cost_pt"] = raw["cost_transit"]
    df["cost_drive"] = raw["cost_driving_fuel"] + raw["cost_driving_ccharge"]
    for alt in alternatives:
        df[f"avail_{alt}"] = True
    feature_columns = {
        "time": {alt: f"time_{alt}" for alt in alternatives},
        "cost": {alt: f"cost_{alt}" for alt in alternatives},
    }
    availability_columns = {alt: f"avail_{alt}" for alt in alternatives}
    data = ChoiceDataset.from_wide(
        df,
        alternatives=alternatives,
        choice="choice",
        variables=feature_columns,
        availability=availability_columns,
        obs_id="obs_id",
    )
    spec = UtilitySpec()
    for alt in alternatives:
        asc = 0 if alt == "walk" else Beta(f"ASC_{alt.upper()}")
        spec.utility(alt, asc + Beta("B_TIME") * "time" + Beta("B_COST") * "cost")
    names = ["B_TIME", "B_COST", "ASC_CYCLE", "ASC_PT", "ASC_DRIVE"]
    return CaseSpec(
        case="lpmc",
        dataset_id="lpmc_london",
        model_name="London Passenger Mode Choice MNL",
        df=df,
        data=data,
        spec=spec,
        alternatives=alternatives,
        feature_columns=feature_columns,
        availability_columns=availability_columns,
        parameter_names=names,
        initial_values={name: 0.0 for name in names},
        source="Biogeme data page lpmc.dat",
    )


CASE_BUILDERS = {
    "airline": make_airline,
    "parking": make_parking,
    "telephone": make_telephone,
    "lpmc": make_lpmc,
}


def run_torch(case: CaseSpec) -> BackendResult:
    model = MultinomialLogit(case.spec)
    data = case.data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    initial = torch.as_tensor(
        [case.initial_values[name] for name in compiled.free_names],
        dtype=torch.float64,
    )
    # Warm up the tensor and autograd path before timing, as in Torch-Choice.
    warmup = initial.clone().requires_grad_(True)
    (-model.loglike(warmup, data, compiled)).backward()
    params = initial.clone().requires_grad_(True)
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


def run_biogeme(case: CaseSpec) -> BackendResult:
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
        if alt != case.alternatives[0]:
            expr += betas[f"ASC_{alt.upper()}"]
        for feature, columns in case.feature_columns.items():
            param_name = {
                "trip_time": "B_TRIP_TIME",
                "fare": "B_FARE",
                "legroom": "B_LEGROOM",
                "access_time": "B_ACCESS_TIME",
                "search_time": "B_SEARCH_TIME",
                "fee": "B_FEE",
                "time": "B_TIME",
                "cost": "B_COST",
            }[feature]
            expr += betas[param_name] * Variable(columns[alt])
        utility[code] = expr
        availability[code] = Variable(case.availability_columns[alt])
    logprob = models.loglogit(utility, availability, Variable("choice_code"))
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.model_name = f"torchdcm_public_{case.case}_{len(df)}"
    biogeme.biogeme_parameters.set_value("save_iterations", False)
    total_start = time.perf_counter()
    estimate_start = time.perf_counter()
    estimates = biogeme.estimate()
    estimate_s = time.perf_counter() - estimate_start
    covariance_start = time.perf_counter()
    covariance = np.asarray(
        estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER),
        dtype=float,
    )
    covariance_s = time.perf_counter() - covariance_start
    total_s = time.perf_counter() - total_start
    return BackendResult(
        backend="biogeme",
        available=True,
        total_s=total_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=float(estimates.final_log_likelihood),
        params={name: float(estimates.get_beta_values()[name]) for name in case.parameter_names},
        covariance=covariance,
    )


def run_apollo(case: CaseSpec) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo", available=False, message="Rscript not found.")
    if not APOLLO_SCRIPT.exists():
        return BackendResult(backend="apollo", available=False, message=f"Missing Apollo script: {APOLLO_SCRIPT}")
    try:
        df, spec = make_apollo_inputs(case)
        with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_public_{case.case}_") as tmp:
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


def run_scipy(case: CaseSpec):
    return run_scipy_mle(case.data, case.spec, case.initial_values, target_names=case.parameter_names)


def run_torch_choice(case: CaseSpec):
    return run_torch_choice_mnl(case.data, case.spec, case.initial_values, max_iter=500)


def run_mlogit(case: CaseSpec):
    return run_mlogit_generic(make_case_design_long(case), case.parameter_names)


def run_gmnl(case: CaseSpec):
    return run_gmnl_generic(make_case_design_long(case), case.parameter_names)


def run_xlogit(case: CaseSpec):
    return run_xlogit_generic(make_case_design_long(case), case.parameter_names)


def make_case_design_long(case: CaseSpec) -> pd.DataFrame:
    utility_columns: dict[str, dict[str, str | float]] = {}
    for alt in case.alternatives:
        terms: dict[str, str | float] = {}
        asc_name = f"ASC_{alt.upper()}"
        if asc_name in case.parameter_names:
            terms[asc_name] = 1.0
        for feature, columns in case.feature_columns.items():
            terms[param_for_feature(feature)] = columns[alt]
        utility_columns[alt] = terms
    return make_design_long(
        case.df,
        case.alternatives,
        "choice",
        utility_columns,
        case.availability_columns,
        case.parameter_names,
    )


def make_apollo_inputs(case: CaseSpec) -> tuple[pd.DataFrame, dict]:
    code_by_alt = {alt: i + 1 for i, alt in enumerate(case.alternatives)}
    df = case.df.copy()
    df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
    df = df.drop(columns=["choice"])
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    utility = {}
    for alt in case.alternatives:
        asc_name = f"ASC_{alt.upper()}"
        utility[alt] = {
            "code": code_by_alt[alt],
            "asc": asc_name if asc_name in case.parameter_names else None,
            "availability": case.availability_columns[alt],
            "variables": {
                param_for_feature(feature): columns[alt]
                for feature, columns in case.feature_columns.items()
            },
        }
    spec = {
        "model_name": f"apollo_public_{case.case}_{case.data.n_obs}",
        "alternatives": case.alternatives,
        "choice_col": "choice_code",
        "parameters": {name: case.initial_values.get(name, 0.0) for name in case.parameter_names},
        "utility": utility,
    }
    return df, spec


def param_for_feature(feature: str) -> str:
    return {
        "trip_time": "B_TRIP_TIME",
        "fare": "B_FARE",
        "legroom": "B_LEGROOM",
        "access_time": "B_ACCESS_TIME",
        "search_time": "B_SEARCH_TIME",
        "fee": "B_FEE",
        "time": "B_TIME",
        "cost": "B_COST",
    }[feature]


def reorder_covariance(covariance, source_names: list[str], target_names: list[str]) -> np.ndarray:
    matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
    return matrix.loc[target_names, target_names].to_numpy(dtype=float)


def predict_probabilities(case: CaseSpec, params: dict[str, float]) -> np.ndarray:
    model = MultinomialLogit(case.spec)
    vector = torch.as_tensor([params[name] for name in case.parameter_names], dtype=torch.float64)
    return model.predict_proba(case.data, vector).detach().cpu().numpy()


def compare(results: list[BackendResult], case: CaseSpec, reference: str = "torchdcm") -> None:
    for result in results:
        if result.available and result.probabilities is None:
            result.probabilities = predict_probabilities(case, result.params or {})
    ref = next(result for result in results if result.backend == reference and result.available)
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        result.max_param_diff = max(abs(result.params[name] - ref.params[name]) for name in case.parameter_names)  # type: ignore[attr-defined]
        result.max_prob_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))  # type: ignore[attr-defined]
        result.max_cov_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
        result.max_se_diff = float(np.max(np.abs(np.sqrt(np.diag(result.covariance)) - np.sqrt(np.diag(ref.covariance)))))  # type: ignore[attr-defined]


def result_payload(case: CaseSpec, results: list[BackendResult]) -> dict:
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
                "params": result.params,
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
    print("alignment:")
    print("  benchmark_mode: full_estimation")
    print(f"  data_source: {payload['source']}")
    print("  initial_values: zeros shared across all estimators")
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
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), required=True)
    parser.add_argument("--n-obs", type=int, default=None)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    case = CASE_BUILDERS[args.case](args.n_obs)
    results = [
        run_torch(case),
        run_torch_choice(case),
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
