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

from benchmark_runtime import (
    configure_single_thread_cpu,
    estimation_covariance_total,
    runtime_policy_metadata,
)

if __name__ == "__main__":
    configure_single_thread_cpu(configure_torch=True)

import numpy as np
import pandas as pd
import torch

from torchdcm import Beta, ChoiceDataset, Nest, NestedLogit, UtilitySpec

import compare_biogeme_public_mnl as public_mnl
import compare_nhts_mnl as nhts_mnl
import run_mlogit_dataset_battery as mlogit_datasets
from compare_mnl_estimators import load_biogeme_swissmetro


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
APOLLO_SCRIPT = ROOT / "experiments" / "apollo" / "R" / "run_nl.R"
NL_LL_ABS_TOL = 0.25
NL_LL_REL_TOL = 1e-5
NL_LL_PER_OBSERVATION_TOL = 0.01


@dataclass(frozen=True)
class NestSpec:
    alternatives: list[str]
    init: float = 0.8
    fixed: bool = False


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


@dataclass
class NestedCase:
    case: str
    data_label: str
    model_label: str
    source: str
    wide_df: pd.DataFrame
    data: ChoiceDataset
    spec: UtilitySpec
    alternatives: list[str]
    beta_names: list[str]
    nests: dict[str, NestSpec]
    initial_values: dict[str, float]

    @property
    def lambda_names(self) -> list[str]:
        return [lambda_name(name) for name, nest in self.nests.items() if not nest.fixed]

    @property
    def parameter_names(self) -> list[str]:
        return [*self.beta_names, *self.lambda_names]


def safe_alt(alt: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in alt)


def lambda_name(nest_name: str) -> str:
    return f"LAMBDA_{nest_name.upper()}"


def raw_lambda_name(nest_name: str) -> str:
    return f"RAW_{lambda_name(nest_name)}"


def nested_case_from_design_long(
    *,
    case: str,
    data_label: str,
    model_label: str,
    source: str,
    long_df: pd.DataFrame,
    alternatives: list[str],
    beta_names: list[str],
    nests: dict[str, NestSpec],
) -> NestedCase:
    long_df = complete_design_long(long_df, alternatives, beta_names)
    chosen_rows = long_df.loc[long_df["choice"].astype(bool), ["obs_id", "alt"]].copy()
    if chosen_rows["obs_id"].duplicated().any():
        raise ValueError(f"More than one chosen alternative per observation in {case}.")
    ordered_obs = long_df[["obs_id"]].drop_duplicates("obs_id").sort_values("obs_id")
    wide_df = ordered_obs.merge(chosen_rows.rename(columns={"alt": "choice"}), on="obs_id", how="left")
    if wide_df["choice"].isna().any():
        raise ValueError(f"Missing chosen alternative for at least one observation in {case}.")
    wide_df = wide_df.reset_index(drop=True)
    feature_columns: dict[str, dict[str, str]] = {name: {} for name in beta_names}
    availability_columns: dict[str, str] = {}
    for alt in alternatives:
        alt_key = safe_alt(alt)
        sub = long_df.loc[long_df["alt"] == alt].sort_values("obs_id").reset_index(drop=True)
        if len(sub) != len(wide_df):
            raise ValueError(f"Alternative {alt} is not present for every observation in {case}.")
        if not np.array_equal(sub["obs_id"].to_numpy(), wide_df["obs_id"].to_numpy()):
            raise ValueError(f"Observation ordering mismatch for alternative {alt} in {case}.")
        for name in beta_names:
            col = f"x__{name}__{alt_key}"
            wide_df[col] = sub[name].to_numpy(dtype=float)
            feature_columns[name][alt] = col
        avail_col = f"avail__{alt_key}"
        wide_df[avail_col] = sub["availability"].astype(bool).to_numpy()
        availability_columns[alt] = avail_col

    data = ChoiceDataset.from_wide(
        wide_df,
        alternatives=alternatives,
        choice="choice",
        variables=feature_columns,
        availability=availability_columns,
        obs_id="obs_id",
    )
    spec = UtilitySpec()
    for alt in alternatives:
        expr = None
        for name in beta_names:
            term = Beta(name) * name
            expr = term if expr is None else expr + term
        spec.utility(alt, expr if expr is not None else 0)

    initial_values = {name: 0.0 for name in beta_names}
    for nest_name, nest in nests.items():
        if not nest.fixed:
            initial_values[lambda_name(nest_name)] = nest.init

    return NestedCase(
        case=case,
        data_label=data_label,
        model_label=model_label,
        source=source,
        wide_df=wide_df,
        data=data,
        spec=spec,
        alternatives=alternatives,
        beta_names=beta_names,
        nests=nests,
        initial_values=initial_values,
    )


def complete_design_long(long_df: pd.DataFrame, alternatives: list[str], beta_names: list[str]) -> pd.DataFrame:
    df = long_df.copy()
    if "availability" not in df.columns:
        df["availability"] = True
    if df[["obs_id", "alt"]].duplicated().any():
        raise ValueError("Duplicate obs_id/alt rows in long-format design.")
    obs_order = list(pd.unique(df["obs_id"]))
    index = pd.MultiIndex.from_product([obs_order, alternatives], names=["obs_id", "alt"])
    df = df.set_index(["obs_id", "alt"]).reindex(index).reset_index()
    df["choice"] = df["choice"].where(df["choice"].notna(), False).astype(bool)
    df["availability"] = df["availability"].where(df["availability"].notna(), False).astype(bool)
    for name in beta_names:
        df[name] = df[name].fillna(0.0).astype(float)
    return df


def nested_case_from_public_mnl(
    *,
    key: str,
    data_label: str,
    base: public_mnl.CaseSpec,
    nests: dict[str, NestSpec],
) -> NestedCase:
    return nested_case_from_design_long(
        case=f"{key}_nested",
        data_label=data_label,
        model_label="Nested logit",
        source=base.source,
        long_df=public_mnl.make_case_design_long(base),
        alternatives=base.alternatives,
        beta_names=base.parameter_names,
        nests=nests,
    )


def make_swissmetro_case(n_obs: int | None) -> NestedCase:
    df, _, _, alternatives = load_biogeme_swissmetro(n_obs)
    utility_columns = {
        "TRAIN": {"ASC_TRAIN": 1.0, "B_TIME": "time_train", "B_COST": "cost_train"},
        "SM": {"B_TIME": "time_sm", "B_COST": "cost_sm"},
        "CAR": {"ASC_CAR": 1.0, "B_TIME": "time_car", "B_COST": "cost_car"},
    }
    long_df = public_mnl.make_design_long(
        df,
        alternatives,
        "choice",
        utility_columns,
        {"TRAIN": "avail_train", "SM": "avail_sm", "CAR": "avail_car"},
        ["ASC_TRAIN", "B_TIME", "B_COST", "ASC_CAR"],
    )
    return nested_case_from_design_long(
        case="swissmetro_nested",
        data_label="Swissmetro",
        model_label="Nested logit",
        source="data/small/biogeme_swissmetro/data.csv",
        long_df=long_df,
        alternatives=alternatives,
        beta_names=["ASC_TRAIN", "B_TIME", "B_COST", "ASC_CAR"],
        nests={
            "PUBLIC": NestSpec(["TRAIN", "SM"], init=0.8),
            "PRIVATE": NestSpec(["CAR"], init=1.0, fixed=True),
        },
    )


def make_airline_case(n_obs: int | None) -> NestedCase:
    base = public_mnl.make_airline(n_obs)
    return nested_case_from_public_mnl(
        key="airline",
        data_label="Airline itinerary",
        base=base,
        nests={
            "ALT12": NestSpec(["ALT1", "ALT2"], init=0.8),
            "ALT3_NEST": NestSpec(["ALT3"], init=1.0, fixed=True),
        },
    )


def make_lpmc_case(_: int | None) -> NestedCase:
    base = public_mnl.make_lpmc(None)
    return nested_case_from_design_long(
        case="lpmc_nested",
        data_label="LPMC London",
        model_label="Nested logit",
        source=base.source,
        long_df=public_mnl.make_case_design_long(base),
        alternatives=base.alternatives,
        beta_names=base.parameter_names,
        nests={
            "ACTIVE": NestSpec(["walk", "cycle"], init=0.8),
            "MOTORIZED": NestSpec(["pt", "drive"], init=0.8),
        },
    )


def make_nhts_case(_: int | None) -> NestedCase:
    base = nhts_mnl.make_nhts_2022_case(None)
    return nested_case_from_design_long(
        case="nhts_2022_nested",
        data_label="NHTS 2022",
        model_label="Nested logit",
        source=base.source,
        long_df=nhts_mnl.make_case_design_long(base),
        alternatives=base.alternatives,
        beta_names=base.parameter_names,
        nests={
            "ACTIVE": NestSpec(["WALK", "BIKE"], init=0.8),
            "MOTORIZED": NestSpec(["AUTO", "TRANSIT", "OTHER"], init=0.8),
        },
    )


def make_parking_case(_: int | None) -> NestedCase:
    base = public_mnl.make_parking(None)
    return nested_case_from_design_long(
        case="parking_nested",
        data_label="Parking Spain",
        model_label="Nested logit",
        source=base.source,
        long_df=public_mnl.make_case_design_long(base),
        alternatives=base.alternatives,
        beta_names=base.parameter_names,
        nests={
            "FACILITY": NestSpec(["FSP", "PSP"], init=0.8),
            "PUP_NEST": NestSpec(["PUP"], init=1.0, fixed=True),
        },
    )


MLOGIT_DATA_LABELS = {
    "catsup": "Catsup",
    "cracker": "Cracker",
    "electricity": "Electricity",
    "fishing": "Fishing",
    "hc": "HC",
    "heating": "Heating",
    "mode": "Mode",
}


MLOGIT_NESTS = {
    "catsup": {
        "HEINZ": NestSpec(["heinz41", "heinz32", "heinz28"], init=0.8),
        "HUNTS": NestSpec(["hunts32"], init=1.0, fixed=True),
    },
    "cracker": {
        "NATIONAL": NestSpec(["sunshine", "kleebler", "nabisco"], init=0.8),
        "PRIVATE": NestSpec(["private"], init=1.0, fixed=True),
    },
    "electricity": {
        "PLAN_12": NestSpec(["1", "2"], init=0.8),
        "PLAN_34": NestSpec(["3", "4"], init=0.8),
    },
    "fishing": {
        "SHORE": NestSpec(["beach", "pier"], init=0.8),
        "BOAT": NestSpec(["boat", "charter"], init=0.8),
    },
    "hc": {
        "CURRENT": NestSpec(["gcc", "ecc", "erc"], init=0.8),
        "NEW": NestSpec(["gc", "ec", "er"], init=0.8),
        "HPC_NEST": NestSpec(["hpc"], init=1.0, fixed=True),
    },
    "heating": {
        "GAS": NestSpec(["gc", "gr"], init=0.8),
        "ELECTRIC": NestSpec(["ec", "er"], init=0.8),
        "HEATPUMP": NestSpec(["hp"], init=1.0, fixed=True),
    },
    "mode": {
        "PRIVATE": NestSpec(["car", "carpool"], init=0.8),
        "TRANSIT": NestSpec(["bus", "rail"], init=0.8),
    },
}


def make_split_nests(alternatives: list[str]) -> dict[str, NestSpec]:
    if len(alternatives) < 3:
        raise ValueError("Nested-logit battery skips binary choice sets.")
    midpoint = len(alternatives) // 2
    return {
        "GROUP_A": NestSpec(alternatives[:midpoint], init=0.8),
        "GROUP_B": NestSpec(alternatives[midpoint:], init=0.8),
    }


def make_mlogit_dataset_case(dataset: str) -> NestedCase:
    df, reference = load_mlogit_dataset(dataset)
    if df is None:
        raise RuntimeError(reference.get("message", f"Unable to load mlogit dataset {dataset}."))
    variables = reference["variables"]
    if isinstance(variables, str):
        variables = [variables]
    long_df, beta_names = mlogit_datasets.make_design_long(df, variables)
    alternatives = [str(alt) for alt in pd.unique(long_df["alt"])]
    long_df["alt"] = long_df["alt"].astype(str)
    nests = MLOGIT_NESTS.get(dataset) or make_split_nests(alternatives)
    missing = sorted({alt for nest in nests.values() for alt in nest.alternatives} - set(alternatives))
    if missing:
        raise ValueError(f"Nest alternatives are not in {dataset}: {missing}")
    return nested_case_from_design_long(
        case=f"mlogit_{dataset}_nested",
        data_label=MLOGIT_DATA_LABELS[dataset],
        model_label="Nested logit",
        source=f"R mlogit::{dataset}",
        long_df=long_df,
        alternatives=alternatives,
        beta_names=beta_names,
        nests=nests,
    )


def make_mlogit_case_builder(dataset: str):
    return lambda _n_obs: make_mlogit_dataset_case(dataset)


def load_mlogit_dataset(dataset: str) -> tuple[pd.DataFrame | None, dict]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None, {"available": False, "message": "Rscript not found."}
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_nested_{dataset}_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        result_path = tmp_path / "result.json"
        env = os.environ.copy()
        r_user_lib = str(Path.home() / "R" / "site-library")
        existing = env.get("R_LIBS_USER")
        env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
        command = [
            rscript,
            str(mlogit_datasets.R_SCRIPT),
            "--dataset",
            dataset,
            "--data-input",
            str(mlogit_datasets.DATA_ROOT / f"mlogit_{dataset}" / "data.csv"),
            "--data-output",
            str(data_path),
            "--result-output",
            str(result_path),
            "--data-only",
        ]
        proc = subprocess.run(command, text=True, capture_output=True, env=env)
        if proc.returncode != 0:
            return None, {"available": False, "message": (proc.stderr or proc.stdout).strip()}
        return pd.read_csv(data_path), json.loads(result_path.read_text(encoding="utf-8"))


CASE_BUILDERS = {
    "swissmetro": make_swissmetro_case,
    "airline": make_airline_case,
    "lpmc": make_lpmc_case,
    "nhts": make_nhts_case,
    "parking": make_parking_case,
    "mlogit_catsup": make_mlogit_case_builder("catsup"),
    "mlogit_cracker": make_mlogit_case_builder("cracker"),
    "mlogit_electricity": make_mlogit_case_builder("electricity"),
    "mlogit_fishing": make_mlogit_case_builder("fishing"),
    "mlogit_hc": make_mlogit_case_builder("hc"),
    "mlogit_heating": make_mlogit_case_builder("heating"),
    "mlogit_mode": make_mlogit_case_builder("mode"),
}


def torch_nests(case: NestedCase) -> dict[str, Nest]:
    return {
        name: Nest(nest.alternatives, init=nest.init, fixed=nest.fixed)
        for name, nest in case.nests.items()
    }


def run_torch(case: NestedCase, max_iter: int) -> BackendResult:
    model = NestedLogit(case.spec, torch_nests(case), max_iter=max_iter)
    data = case.data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    initial = torch.cat(
        [
            compiled.free_initial,
            model._lambda_to_internal(compiled.lambda_initial[~compiled.lambda_is_fixed]),
        ]
    )

    # Exclude one-time tensor-kernel and autograd initialization from timing,
    # following the same runtime policy as the generated MNL benchmark.
    warmup_internal = initial.clone().detach().requires_grad_(True)
    warmup_natural = model._internal_to_natural(warmup_internal, compiled)
    (-model.loglike(warmup_natural, data, compiled)).backward()

    params = initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=model.max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        natural = model._internal_to_natural(params, compiled)
        loss = -model.loglike(natural, data, compiled)
        loss.backward()
        return loss

    estimate_start = time.perf_counter()
    optimizer.step(closure)
    estimate_s = time.perf_counter() - estimate_start
    final_internal = params.detach().clone().requires_grad_(True)
    final_natural = model._internal_to_natural(final_internal, compiled)
    loglike = float(model.loglike(final_natural, data, compiled).detach().cpu())
    covariance_start = time.perf_counter()
    hessian = torch.autograd.functional.hessian(
        lambda p: model.loglike(model._internal_to_natural(p, compiled), data, compiled),
        final_internal,
    )
    information = -hessian.detach()
    covariance = None
    if torch.isfinite(information).all():
        try:
            cov_internal = torch.linalg.pinv(information, hermitian=True)
        except RuntimeError:
            try:
                cov_internal = torch.linalg.pinv(information, hermitian=False)
            except RuntimeError:
                cov_internal = None
        if cov_internal is not None:
            transform = model._natural_jacobian(final_internal.detach(), compiled)
            covariance = (transform @ cov_internal @ transform.T).detach().cpu().numpy()
    covariance_s = time.perf_counter() - covariance_start
    names = compiled.free_names
    probabilities = model.predict_proba(data, final_natural, compiled).detach().cpu().numpy()
    return BackendResult(
        backend="torchdcm",
        available=True,
        total_s=estimate_s + covariance_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=loglike,
        params={name: float(final_natural[i].detach().cpu()) for i, name in enumerate(names)},
        covariance=covariance,
        probabilities=probabilities,
    )


def run_biogeme(case: NestedCase, lambda_min: float) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme import models
        from biogeme.expressions import Beta as BioBeta
        from biogeme.expressions import Variable
        from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult(backend="biogeme", available=False, message=f"Biogeme unavailable: {exc}")

    df = case.wide_df.copy()
    code_by_alt = {alt: i + 1 for i, alt in enumerate(case.alternatives)}
    df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
    df = df.drop(columns=["choice"])
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    database = db.Database(f"torchdcm_{case.case}", df)
    betas = {name: BioBeta(name, case.initial_values.get(name, 0.0), None, None, 0) for name in case.beta_names}
    lambda_betas = {
        lambda_name(nest_name): BioBeta(lambda_name(nest_name), nest.init, lambda_min, 1.0, 0)
        for nest_name, nest in case.nests.items()
        if not nest.fixed
    }
    utility = {}
    availability = {}
    for alt, code in code_by_alt.items():
        alt_key = safe_alt(alt)
        expr = 0
        for name in case.beta_names:
            expr += betas[name] * Variable(f"x__{name}__{alt_key}")
        utility[code] = expr
        availability[code] = Variable(f"avail__{alt_key}")
    biogeme_nests = []
    for nest_name, nest in case.nests.items():
        nest_codes = [code_by_alt[alt] for alt in nest.alternatives]
        nest_parameter = 1.0 if nest.fixed else 1.0 / lambda_betas[lambda_name(nest_name)]
        biogeme_nests.append(OneNestForNestedLogit(nest_parameter, nest_codes, name=nest_name))
    nests = NestsForNestedLogit(choice_set=list(code_by_alt.values()), tuple_of_nests=tuple(biogeme_nests))
    logprob = models.lognested(utility, availability, nests, Variable("choice_code"))
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
    total_s = estimation_covariance_total(estimate_s, covariance_s)
    names = case.parameter_names
    covariance = covariance_to_array(covariance_obj, names)
    beta_values = estimates.get_beta_values()
    return BackendResult(
        backend="biogeme",
        available=True,
        total_s=total_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=float(estimates.final_log_likelihood),
        params={name: float(beta_values[name]) for name in names},
        covariance=covariance,
    )


def run_apollo(case: NestedCase, lambda_min: float) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo", available=False, message="Rscript not found.")
    if not APOLLO_SCRIPT.exists():
        return BackendResult(backend="apollo", available=False, message=f"Missing Apollo script: {APOLLO_SCRIPT}")
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_{case.case}_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        spec_path = tmp_path / "spec.json"
        output_path = tmp_path / "apollo_result.json"
        df, spec = make_apollo_inputs(case, lambda_min)
        df.to_csv(data_path, index=False)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        env = os.environ.copy()
        r_user_lib = str(Path.home() / "R" / "site-library")
        existing = env.get("R_LIBS_USER")
        env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
        command = [rscript, str(APOLLO_SCRIPT), "--data", str(data_path), "--spec", str(spec_path), "--output", str(output_path)]
        total_start = time.perf_counter()
        proc = subprocess.run(command, text=True, capture_output=True, env=env)
        total_s = time.perf_counter() - total_start
        if proc.returncode != 0:
            return BackendResult(backend="apollo", available=False, total_s=total_s, message=(proc.stderr or proc.stdout).strip())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        covariance_names = payload.get("covariance_names") or case.parameter_names
        covariance = reorder_covariance_or_none(payload.get("covariance"), covariance_names, case.parameter_names)
        estimate_s = payload.get("timing", {}).get("estimate_seconds")
        covariance_s = payload.get("timing", {}).get("covariance_seconds")
        convergence = payload.get("convergence") or {}
        convergence_status = convergence.get("status")
        convergence_message = convergence.get("message")
        message = (
            f"apollo_version={payload.get('apollo_version')}; "
            f"convergence_status={convergence_status}; "
            f"convergence_message={convergence_message}"
        )
        if "unfavorable" in str(convergence_message).lower():
            return BackendResult(
                backend="apollo",
                available=False,
                total_s=estimation_covariance_total(estimate_s, covariance_s),
                estimate_s=estimate_s,
                covariance_s=covariance_s,
                message=message,
            )
        return BackendResult(
            backend="apollo",
            available=True,
            total_s=estimation_covariance_total(estimate_s, covariance_s),
            estimate_s=estimate_s,
            covariance_s=covariance_s,
            loglike=float(payload["loglike"]),
            params={name: float(payload["estimates"][name]) for name in case.parameter_names},
            covariance=covariance,
            message=message,
        )


def make_apollo_inputs(case: NestedCase, lambda_min: float) -> tuple[pd.DataFrame, dict]:
    code_by_alt = {alt: i + 1 for i, alt in enumerate(case.alternatives)}
    df = case.wide_df.copy()
    df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
    df = df.drop(columns=["choice"])
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    parameters = {name: case.initial_values.get(name, 0.0) for name in case.beta_names}
    nests = {}
    for nest_name, nest in case.nests.items():
        if nest.fixed:
            nests[nest_name] = {"alternatives": nest.alternatives, "lambda_param": None, "lambda_value": nest.init}
        else:
            lam = np.clip((nest.init - lambda_min) / (1.0 - lambda_min), 1e-12, 1.0 - 1e-12)
            parameters[raw_lambda_name(nest_name)] = float(np.log(lam / (1.0 - lam)))
            nests[nest_name] = {
                "alternatives": nest.alternatives,
                "lambda_param": lambda_name(nest_name),
                "lambda_raw_param": raw_lambda_name(nest_name),
                "lambda_min": lambda_min,
            }
    utility = {}
    for alt in case.alternatives:
        alt_key = safe_alt(alt)
        utility[alt] = {
            "code": code_by_alt[alt],
            "asc": None,
            "availability": f"avail__{alt_key}",
            "variables": {name: f"x__{name}__{alt_key}" for name in case.beta_names},
        }
    spec = {
        "model_name": f"apollo_{case.case}_{case.data.n_obs}",
        "alternatives": case.alternatives,
        "choice_col": "choice_code",
        "parameters": parameters,
        "utility": utility,
        "nests": nests,
    }
    return df, spec


def covariance_to_array(covariance_obj, names: list[str]) -> np.ndarray:
    if hasattr(covariance_obj, "loc"):
        return covariance_obj.loc[names, names].to_numpy(dtype=float)
    return np.asarray(covariance_obj, dtype=float)


def reorder_covariance_or_none(covariance, source_names: list[str], target_names: list[str]) -> np.ndarray | None:
    if covariance is None:
        return None
    matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
    matrix = matrix.loc[target_names, target_names].apply(pd.to_numeric, errors="coerce")
    array = matrix.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        return None
    return array


def predict_probabilities(case: NestedCase, params: dict[str, float]) -> np.ndarray:
    model = NestedLogit(case.spec, torch_nests(case))
    vector = torch.as_tensor([params[name] for name in case.parameter_names], dtype=torch.float64)
    return model.predict_proba(case.data, vector).detach().cpu().numpy()


def compare(results: list[BackendResult], case: NestedCase) -> None:
    ref = next((result for result in results if result.backend == "torchdcm" and result.available), None)
    if ref is None:
        return
    for result in results:
        if result.available and result.probabilities is None:
            result.probabilities = predict_probabilities(case, result.params or {})
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        result.max_param_diff = max(abs(result.params[name] - ref.params[name]) for name in case.parameter_names)  # type: ignore[attr-defined]
        result.max_prob_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))  # type: ignore[attr-defined]
        if result.covariance is not None and ref.covariance is not None:
            result.max_cov_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
            result.max_se_diff = float(
                np.max(np.abs(np.sqrt(np.diag(result.covariance)) - np.sqrt(np.diag(ref.covariance))))
            )  # type: ignore[attr-defined]
        else:
            result.max_cov_diff = None  # type: ignore[attr-defined]
            result.max_se_diff = None  # type: ignore[attr-defined]


def loglike_tolerance(best_loglike: float, n_obs: int) -> float:
    return max(
        NL_LL_ABS_TOL,
        NL_LL_REL_TOL * abs(best_loglike),
        NL_LL_PER_OBSERVATION_TOL * n_obs,
    )


def mark_worse_loglikes(results: list[BackendResult], n_obs: int) -> float | None:
    completed = [
        result
        for result in results
        if result.available and result.loglike is not None and np.isfinite(result.loglike)
    ]
    if not completed:
        return None
    best_loglike = max(float(result.loglike) for result in completed)
    tolerance = loglike_tolerance(best_loglike, n_obs)
    for result in completed:
        result.worse_loglike = bool(result.loglike < best_loglike - tolerance)  # type: ignore[attr-defined]
    return tolerance


def is_consistent(results: list[BackendResult], n_obs: int) -> bool | None:
    comparable = [
        result
        for result in results
        if result.available
        and result.loglike is not None
        and np.isfinite(result.loglike)
        and not getattr(result, "worse_loglike", False)
    ]
    if len(comparable) < 2:
        return None
    best_loglike = max(float(result.loglike) for result in comparable)
    tolerance = loglike_tolerance(best_loglike, n_obs)
    return all(abs(float(result.loglike) - best_loglike) <= tolerance for result in comparable)


def result_payload(case: NestedCase, results: list[BackendResult]) -> dict:
    compare(results, case)
    mark_worse_loglikes(results, case.data.n_obs)
    return {
        "case": case.case,
        "data": case.data_label,
        "model": case.model_label,
        "n_obs": case.data.n_obs,
        "n_alternatives": len(case.alternatives),
        "n_parameters": len(case.parameter_names),
        "source": case.source,
        "nests": {name: {"alternatives": nest.alternatives, "fixed": nest.fixed, "init": nest.init} for name, nest in case.nests.items()},
        "parameters": case.parameter_names,
        "consistent": is_consistent(results, case.data.n_obs),
        "runtime_policy": runtime_policy_metadata(),
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
                "worse_loglike": getattr(result, "worse_loglike", False),
                "message": result.message,
            }
            for result in results
        ],
    }


def run_case(case: NestedCase, max_iter: int, lambda_min: float) -> dict:
    results = [
        safe_run("torchdcm", lambda: run_torch(case, max_iter=max_iter)),
        safe_run("biogeme", lambda: run_biogeme(case, lambda_min=lambda_min)),
        safe_run("apollo", lambda: run_apollo(case, lambda_min=lambda_min)),
    ]
    return result_payload(case, results)


def safe_run(backend: str, fn) -> BackendResult:
    start = time.perf_counter()
    try:
        return fn()
    except Exception as exc:
        return BackendResult(
            backend=backend,
            available=False,
            total_s=time.perf_counter() - start,
            message=f"{type(exc).__name__}: {exc}",
        )


def render_markdown(payloads: list[dict]) -> str:
    lines = [
        "# Real-data Nested Logit Benchmark",
        "",
        "Runtimes report estimation plus covariance on one logical CPU.",
        "A dagger marks a clearly inferior final log likelihood; its runtime is retained but its estimate is excluded from consistency. N.A. means fewer than two comparable estimates remain.",
        "",
        "| Data | Model | N | TorchDCM | Biogeme | Apollo | Consistent? |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for payload in payloads:
        rows = {row["backend"]: row for row in payload["backends"]}
        lines.append(
            "| {data} | {model} | {n_obs} | {torchdcm} | {biogeme} | {apollo} | {consistent} |".format(
                data=payload["data"],
                model=payload["model"],
                n_obs=payload["n_obs"],
                torchdcm=fmt_time(rows.get("torchdcm")),
                biogeme=fmt_time(rows.get("biogeme")),
                apollo=fmt_time(rows.get("apollo")),
                consistent=(
                    "N.A."
                    if payload["consistent"] is None
                    else ("Yes" if payload["consistent"] else "No")
                ),
            )
        )
    lines.extend(["", "## Nest specifications", ""])
    for payload in payloads:
        nest_text = "; ".join(f"{name}={','.join(spec['alternatives'])}" for name, spec in payload["nests"].items())
        lines.append(f"- `{payload['case']}`: {nest_text}")
    return "\n".join(lines) + "\n"


def fmt_time(row: dict | None) -> str:
    if not row:
        return "NA"
    if not row["available"]:
        return "Fail"
    marker = "†" if row.get("worse_loglike") else ""
    return f"{row['total_s']:.3f}{marker}"


def print_payload(payload: dict) -> None:
    print(f"case: {payload['case']}")
    print(f"data: {payload['data']}")
    print(f"model: {payload['model']}")
    print(f"n_obs: {payload['n_obs']}")
    print(f"n_parameters: {payload['n_parameters']}")
    print(f"consistent: {payload['consistent']}")
    print("nests:")
    for name, spec in payload["nests"].items():
        fixed = " fixed" if spec["fixed"] else ""
        print(f"  {name}: {', '.join(spec['alternatives'])}{fixed}")
    print()
    print(
        f"{'backend':<12}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'param_diff':>14}{'prob_diff':>14}{'cov_diff':>14}{'se_diff':>14}"
    )
    for row in payload["backends"]:
        if row["available"]:
            print(
                f"{row['backend']:<12}{str(row['available']):>10}"
                f"{fmt_optional(row['total_s']):>12}{fmt_optional(row['estimate_s']):>12}{fmt_optional(row['covariance_s']):>12}"
                f"{row['loglike']:>18.10f}{row['ll_diff']:>14.3e}{row['max_param_diff']:>14.3e}"
                f"{row['max_prob_diff']:>14.3e}{fmt_optional(row['max_cov_diff']):>14}{fmt_optional(row['max_se_diff']):>14}"
            )
        else:
            print(f"{row['backend']:<12}{str(row['available']):>10}  {row['message']}")


def fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*sorted(CASE_BUILDERS), "all"], default="all")
    parser.add_argument("--n-obs", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--lambda-min", type=float, default=0.0001)
    parser.add_argument("--json-output", type=Path, default=RESULTS / "nested_real_battery_full.json")
    parser.add_argument("--md-output", type=Path, default=RESULTS / "nested_real_battery_full.md")
    args = parser.parse_args()

    selected = list(CASE_BUILDERS) if args.case == "all" else [args.case]
    payloads = []
    for key in selected:
        try:
            case = CASE_BUILDERS[key](args.n_obs)
        except Exception as exc:
            print(f"case: {key}")
            print(f"skipped: {type(exc).__name__}: {exc}")
            print()
            continue
        payload = run_case(case, max_iter=args.max_iter, lambda_min=args.lambda_min)
        payloads.append(payload)
        print_payload(payload)
        print()

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    if args.md_output:
        args.md_output.parent.mkdir(parents=True, exist_ok=True)
        args.md_output.write_text(render_markdown(payloads), encoding="utf-8")


if __name__ == "__main__":
    main()
