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

from torchdcm import (
    Beta,
    ChoiceDataset,
    MixedLogit,
    MultinomialLogit,
    RandomCoefficient,
    UtilitySpec,
)
from compare_biogeme_public_mnl import CASE_BUILDERS as PUBLIC_CASE_BUILDERS
from compare_mnl_estimators import load_biogeme_swissmetro, make_initial_values, spec_with_initials
from run_mlogit_dataset_battery import DEFAULT_DATASETS as MLOGIT_DATASETS
from run_mlogit_dataset_battery import run_r_reference


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
APOLLO_MIXED_SCRIPT = ROOT / "experiments" / "apollo" / "R" / "run_generic_mixed_estimate.R"


@dataclass
class MixedCase:
    case: str
    dataset_id: str
    source: str
    model_name: str
    df: pd.DataFrame
    data: ChoiceDataset
    spec: UtilitySpec
    alternatives: list[str]
    choice_col: str
    utility_terms: dict[str, list[tuple[str, str | None]]]
    availability_columns: dict[str, str]
    parameter_names: list[str]
    random_names: list[str]
    sigma_scale: float


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
class MixedStart:
    means: dict[str, float]
    sigmas: dict[str, float]
    mnl_loglike: float
    mnl_gradient_norm: float
    sigma_scale: float


def public_case_to_mixed(case_key: str, n_obs: int | None, sigma_scale: float) -> MixedCase:
    base = PUBLIC_CASE_BUILDERS[case_key](n_obs)
    random_names = choose_random_parameters(base.parameter_names)
    return MixedCase(
        case=base.case,
        dataset_id=base.dataset_id,
        source=base.source,
        model_name=f"{base.model_name.replace(' MNL', '')} mixed logit",
        df=base.df.copy(),
        data=base.data,
        spec=base.spec,
        alternatives=base.alternatives,
        choice_col="choice",
        utility_terms=utility_terms_from_public_case(base),
        availability_columns=base.availability_columns,
        parameter_names=base.parameter_names,
        random_names=random_names,
        sigma_scale=sigma_scale,
    )


def swissmetro_case(n_obs: int | None, sigma_scale: float) -> MixedCase:
    df, data, base_spec, alternatives = load_biogeme_swissmetro(n_obs or 10719)
    initial_values = make_initial_values(base_spec.parameter_names, mode="zero", seed=20260704, scale=0.1)
    spec = spec_with_initials(base_spec, initial_values)
    utility_terms = {
        "TRAIN": [("ASC_TRAIN", None), ("B_TIME", "time_train"), ("B_COST", "cost_train")],
        "SM": [("B_TIME", "time_sm"), ("B_COST", "cost_sm")],
        "CAR": [("ASC_CAR", None), ("B_TIME", "time_car"), ("B_COST", "cost_car")],
    }
    return MixedCase(
        case="swissmetro",
        dataset_id="biogeme_swissmetro",
        source="data/small/biogeme_swissmetro/data.csv",
        model_name="Swissmetro mixed logit",
        df=df.copy(),
        data=data,
        spec=spec,
        alternatives=alternatives,
        choice_col="choice",
        utility_terms=utility_terms,
        availability_columns={alt: f"avail_{alt.lower()}" for alt in alternatives},
        parameter_names=base_spec.parameter_names,
        random_names=["B_TIME", "B_COST"],
        sigma_scale=sigma_scale,
    )


def mlogit_case_to_mixed(dataset: str, sigma_scale: float) -> MixedCase:
    df, ref = run_r_reference(dataset)
    if df is None or not ref.get("available"):
        raise RuntimeError(ref.get("message", f"Could not export mlogit::{dataset}"))
    variables_raw = ref["variables"]
    variables = [variables_raw] if isinstance(variables_raw, str) else list(variables_raw)
    if len(variables) < 2:
        raise RuntimeError(f"Need at least two observed variables for 2+ random coefficients; found {variables}.")
    alternatives = list(pd.unique(df["alt"]))
    data = ChoiceDataset.from_long(
        df,
        obs_id="obs_id",
        alt_id="alt",
        choice="choice",
        variables=variables,
        availability="availability" if "availability" in df.columns else None,
        alt_order=alternatives,
    )
    spec = UtilitySpec()
    parameter_names = [f"B_{variable.upper()}" for variable in variables]
    for alt in alternatives:
        utility = None
        for variable, parameter in zip(variables, parameter_names):
            term = Beta(parameter) * variable
            utility = term if utility is None else utility + term
        spec.utility(alt, utility)
    wide_df, alt_columns, choice_col = long_to_wide(df, variables)
    utility_terms = {
        str(alt): [(f"B_{variable.upper()}", str(columns[variable])) for variable in variables]
        for alt, columns in alt_columns.items()
    }
    availability_columns = {str(alt): str(columns["availability"]) for alt, columns in alt_columns.items()}
    return MixedCase(
        case=f"mlogit_{dataset}",
        dataset_id=f"mlogit_{dataset}",
        source=f"R mlogit::{dataset}",
        model_name=f"mlogit::{dataset} mixed logit",
        df=wide_df,
        data=data,
        spec=spec,
        alternatives=[str(alt) for alt in alternatives],
        choice_col=choice_col,
        utility_terms=utility_terms,
        availability_columns=availability_columns,
        parameter_names=parameter_names,
        random_names=parameter_names[: min(4, len(parameter_names))],
        sigma_scale=sigma_scale,
    )


def choose_random_parameters(parameter_names: list[str]) -> list[str]:
    observed = [name for name in parameter_names if not name.startswith("ASC_")]
    chosen = observed[:4]
    if len(chosen) < 2:
        chosen.extend([name for name in parameter_names if name not in chosen][: 2 - len(chosen)])
    if len(chosen) < 2:
        raise RuntimeError(f"Need at least two free parameters for 2+ random coefficients: {parameter_names}")
    return chosen[:4]


def utility_terms_from_public_case(base) -> dict[str, list[tuple[str, str | None]]]:
    terms: dict[str, list[tuple[str, str | None]]] = {}
    for alt in base.alternatives:
        alt_terms: list[tuple[str, str | None]] = []
        asc_name = f"ASC_{alt.upper()}"
        if asc_name in base.parameter_names:
            alt_terms.append((asc_name, None))
        for feature, columns in base.feature_columns.items():
            alt_terms.append((param_for_feature(feature), columns[alt]))
        terms[alt] = alt_terms
    return terms


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


def prepare_starting_values(case: MixedCase) -> MixedStart:
    mnl = MultinomialLogit(case.spec, device="cpu", max_iter=300)
    result = mnl.fit(case.data, max_iter=300)
    if not (
        np.isfinite(result.loglike)
        and bool(torch.isfinite(result.params).all())
        and bool(torch.isfinite(result.gradient).all())
    ):
        raise RuntimeError(
            f"MNL warm start is non-finite for {case.case}."
        )
    means = {
        name: float(value)
        for name, value in zip(result.param_names, result.params.detach().cpu())
    }
    compiled = mnl.compile(case.data.to(device=mnl.device, dtype=mnl.dtype))
    free_index = {name: index for index, name in enumerate(compiled.free_names)}
    fixed_index = {name: index for index, name in enumerate(compiled.fixed_names)}
    sigmas = {}
    for name in case.random_names:
        if name in free_index:
            column = compiled.design[:, free_index[name]]
        elif name in fixed_index:
            column = compiled.fixed_design[:, fixed_index[name]]
        else:
            raise RuntimeError(f"Random coefficient {name} is absent from the MNL design.")
        rms = float(torch.sqrt(torch.mean(column.square())).detach().cpu())
        if not np.isfinite(rms) or rms <= 0.0:
            raise RuntimeError(f"Random-coefficient design column {name} has zero or non-finite RMS.")
        sigmas[name] = float(np.clip(case.sigma_scale / rms, 1e-6, 10.0))
    return MixedStart(
        means=means,
        sigmas=sigmas,
        mnl_loglike=result.loglike,
        mnl_gradient_norm=float(torch.linalg.vector_norm(result.gradient).detach().cpu()),
        sigma_scale=case.sigma_scale,
    )


def run_torch(
    case: MixedCase,
    draws: torch.Tensor,
    max_iter: int,
    device: str,
    start: MixedStart,
) -> BackendResult:
    if torch.device(device).type != "cpu":
        raise ValueError("Cross-estimator mixed-logit batteries must use CPU for TorchDCM.")
    model = MixedLogit(
        spec_with_initials(case.spec, start.means),
        [
            RandomCoefficient(name, sigma_init=start.sigmas[name])
            for name in case.random_names
        ],
        draws=draws,
        panel=False,
        device=device,
        max_iter=max_iter,
    )
    data = case.data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    # Exclude one-time tensor-kernel and autograd initialization from timing,
    # following the generated MNL and NL runtime policy.
    internal_initial = torch.cat(
        [
            compiled.free_initial,
            model._sigma_to_internal(compiled.sigma_initial[~compiled.sigma_is_fixed]),
            compiled.chol_offdiag_initial,
        ]
    )
    warmup_internal = internal_initial.clone().detach().requires_grad_(True)
    warmup_natural = model._internal_to_natural(warmup_internal, compiled)
    (-model.loglike(warmup_natural, data, compiled)).backward()
    result = model.fit(data, max_iter=max_iter)
    status = result.convergence_status
    estimate_s = float(status.get("optimization_seconds", 0.0))
    covariance_s = float(status.get("inference_seconds", 0.0))
    success = bool(status.get("success", False))
    message = (
        f"success={success}; {status.get('message')}; "
        f"optimizer_iterations={status.get('optimizer_iterations')}; "
        f"function_evaluations={status.get('function_evaluations')}; "
        f"internal_gradient_norm={status.get('gradient_norm')}; "
        f"scaled_gradient_norm={status.get('scaled_gradient_norm')}; "
        f"nonfinite_evaluations={status.get('nonfinite_evaluations')}"
    )
    return BackendResult(
        backend="torchdcm",
        available=success,
        total_s=estimate_s + covariance_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=result.loglike,
        params={
            name: float(value)
            for name, value in zip(result.param_names, result.params.detach().cpu())
        },
        covariance=result.cov_params().detach().cpu().numpy(),
        probabilities=model.predict_proba(data, result.params, compiled).detach().cpu().numpy(),
        message=message,
    )


def run_biogeme(
    case: MixedCase,
    draws: np.ndarray,
    max_iter: int,
    start: MixedStart,
    *,
    smooth_positive_scales: bool = True,
) -> BackendResult:
    tmp_root = Path(tempfile.gettempdir()) if False else Path(os.environ.get("TMPDIR", "/tmp"))
    os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "torchdcm_matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(tmp_root / "torchdcm_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme.expressions import Beta as BioBeta
        from biogeme.expressions import Variable
        from biogeme.expressions import BinaryMax
        from biogeme.expressions import exp
        from biogeme.expressions import log
        from biogeme import models
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult("biogeme", False, message=f"Biogeme unavailable: {exc}")

    try:
        df = case.df.copy().reset_index(drop=True)
        code_by_alt = {alt: i + 1 for i, alt in enumerate(case.alternatives)}
        if case.choice_col == "choice":
            df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
            choice_col = "choice_code"
            df = df.drop(columns=["choice"])
        else:
            choice_col = case.choice_col
        for column in df.select_dtypes(include=["bool"]).columns:
            df[column] = df[column].astype(int)
        draw_columns = {
            f"DRAW_{name}_{draw_index}": float(draws[draw_index, random_index])
            for draw_index in range(draws.shape[0])
            for random_index, name in enumerate(case.random_names)
        }
        if draw_columns:
            df = pd.concat([df, pd.DataFrame(draw_columns, index=df.index)], axis=1)

        sigma_names = [f"SIGMA_{name}" for name in case.random_names]
        names = [*case.parameter_names, *sigma_names]
        betas = {
            name: BioBeta(name, start.means[name], None, None, 0)
            for name in case.parameter_names
        }
        if smooth_positive_scales:
            phi_names = [f"PHI_{name}" for name in sigma_names]
            optimization_names = [*case.parameter_names, *phi_names]
            for random_name, sigma_name, phi_name in zip(
                case.random_names, sigma_names, phi_names
            ):
                sigma_init = start.sigmas[random_name]
                phi_init = float(sigma_init + np.log(-np.expm1(-sigma_init)))
                phi = BioBeta(phi_name, phi_init, None, None, 0)
                abs_phi = BinaryMax(phi, -phi)
                betas[sigma_name] = BinaryMax(phi, 0) + log(1 + exp(-abs_phi))
        else:
            phi_names = []
            optimization_names = names
            for random_name, sigma_name in zip(case.random_names, sigma_names):
                betas[sigma_name] = BioBeta(
                    sigma_name, start.sigmas[random_name], 0.0, None, 0
                )

        database = db.Database(f"torchdcm_mixed_{case.case}_{len(df)}", df)
        choice = Variable(choice_col)
        availability = {code_by_alt[alt]: Variable(case.availability_columns[alt]) for alt in case.alternatives}
        chosen_probs = []
        for draw_index in range(draws.shape[0]):
            utility = {}
            for alt in case.alternatives:
                expr = 0
                for param, column in case.utility_terms[alt]:
                    coeff = betas[param]
                    if param in case.random_names:
                        coeff = coeff + betas[f"SIGMA_{param}"] * Variable(f"DRAW_{param}_{draw_index}")
                    expr += coeff if column is None else coeff * Variable(column)
                utility[code_by_alt[alt]] = expr
            chosen_probs.append(models.logit(utility, availability, choice))
        average_prob = chosen_probs[0]
        for expression in chosen_probs[1:]:
            average_prob = average_prob + expression
        logprob = log(average_prob / float(len(chosen_probs)))

        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = f"torchdcm_mixed_{case.case}_{len(df)}_{draws.shape[0]}"
        biogeme.biogeme_parameters.set_value("save_iterations", False)
        try:
            biogeme.biogeme_parameters.set_value("max_iterations", max_iter)
        except Exception:
            pass
        total_start = time.perf_counter()
        estimate_start = time.perf_counter()
        estimates = biogeme.estimate()
        estimate_s = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance_obj = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
        covariance_s = time.perf_counter() - covariance_start
        beta_values = estimates.get_beta_values()
        covariance_internal = covariance_to_array(covariance_obj, optimization_names)
        if smooth_positive_scales:
            transform = np.ones(len(names))
            natural_values = {
                name: float(beta_values[name]) for name in case.parameter_names
            }
            for index, (sigma_name, phi_name) in enumerate(
                zip(sigma_names, phi_names), start=len(case.parameter_names)
            ):
                phi_value = float(beta_values[phi_name])
                natural_values[sigma_name] = float(np.logaddexp(0.0, phi_value))
                transform[index] = float(
                    np.exp(-np.logaddexp(0.0, -phi_value))
                )
            covariance = (
                transform[:, None] * covariance_internal * transform[None, :]
            )
        else:
            natural_values = {name: float(beta_values[name]) for name in names}
            covariance = covariance_internal
        return BackendResult(
            backend="biogeme",
            available=True,
            total_s=estimation_covariance_total(estimate_s, covariance_s),
            estimate_s=estimate_s,
            covariance_s=covariance_s,
            loglike=float(estimates.final_log_likelihood),
            params=natural_values,
            covariance=covariance,
        )
    except Exception as exc:
        return BackendResult("biogeme", False, message=f"{type(exc).__name__}: {exc}")


def run_apollo(
    case: MixedCase,
    n_draws: int,
    timeout: float,
    start: MixedStart,
    max_iter: int,
) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult("apollo", False, message="Rscript not found.")
    if not APOLLO_MIXED_SCRIPT.exists():
        return BackendResult("apollo", False, message=f"Missing Apollo script: {APOLLO_MIXED_SCRIPT}")
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_real_mixl_{case.case}_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        spec_path = tmp_path / "spec.json"
        output_path = tmp_path / "apollo_result.json"
        df, spec = apollo_inputs(case, n_draws, start, max_iter)
        df.to_csv(data_path, index=False)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        command = [
            rscript,
            str(APOLLO_MIXED_SCRIPT),
            "--data",
            str(data_path),
            "--spec",
            str(spec_path),
            "--output",
            str(output_path),
        ]
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                text=True,
                capture_output=True,
                env=r_env(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return BackendResult(
                "apollo",
                False,
                total_s=timeout,
                message=f"Apollo exceeded {timeout:.0f}s timeout.",
            )
        wall_s = time.perf_counter() - start
        if proc.returncode != 0 or not output_path.exists():
            return BackendResult(
                "apollo",
                False,
                total_s=wall_s,
                message=(proc.stderr or proc.stdout).strip(),
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        names = [*case.parameter_names, *[f"SIGMA_{name}" for name in case.random_names]]
        estimate_s = result.get("timing", {}).get("estimate_seconds")
        covariance_s = result.get("timing", {}).get("covariance_seconds")
        total_s = estimation_covariance_total(estimate_s, covariance_s)
        convergence = result.get("convergence") or {}
        convergence_message = str(convergence.get("message"))
        max_abs_gradient = convergence.get("max_abs_gradient")
        premature_relative_convergence = (
            "relative function convergence" in convergence_message.lower()
            and isinstance(max_abs_gradient, (int, float))
            and max_abs_gradient > 1e-2
        )
        message = (
            f"apollo_version={result.get('apollo_version')}; "
            f"apollo_halton_draws={n_draws}; "
            f"convergence_status={convergence.get('status')}; "
            f"convergence_message={convergence.get('message')}; "
            f"successful_estimation={convergence.get('successful_estimation')}; "
            f"iterations={convergence.get('iterations')}; "
            f"max_abs_gradient={convergence.get('max_abs_gradient')}"
        )
        if premature_relative_convergence:
            message += "; convergence_warning=large gradient after relative-function convergence"
        if (
            "unfavorable" in convergence_message.lower()
            or convergence.get("successful_estimation") is False
        ):
            return BackendResult(
                "apollo",
                False,
                total_s=total_s,
                estimate_s=estimate_s,
                covariance_s=covariance_s,
                message=message,
            )
        covariance = reorder_covariance(
            result.get("covariance"),
            result.get("covariance_names"),
            names,
        )
        return BackendResult(
            "apollo",
            True,
            total_s=total_s,
            estimate_s=estimate_s,
            covariance_s=covariance_s,
            loglike=float(result["loglike"]),
            params={name: float(result["estimates"][name]) for name in names},
            covariance=covariance,
            message=message,
        )


def apollo_inputs(
    case: MixedCase,
    n_draws: int,
    start: MixedStart,
    max_iter: int,
) -> tuple[pd.DataFrame, dict]:
    df = case.df.copy()
    code_by_alt = {alt: index + 1 for index, alt in enumerate(case.alternatives)}
    if case.choice_col == "choice":
        df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
        df = df.drop(columns=["choice"])
        choice_col = "choice_code"
    else:
        choice_col = case.choice_col
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    parameters = dict(start.means)
    for name in case.random_names:
        sigma = start.sigmas[name]
        parameters[f"PHI_SIGMA_{name}"] = float(
            sigma + np.log(-np.expm1(-sigma))
        )
    utility = {
        alt: {
            "code": code_by_alt[alt],
            "availability": case.availability_columns[alt],
            "terms": [
                {"parameter": parameter, "variable": variable}
                for parameter, variable in case.utility_terms[alt]
            ],
        }
        for alt in case.alternatives
    }
    return df, {
        "model_name": f"apollo_real_mixl_{case.case}_{case.data.n_obs}",
        "alternatives": case.alternatives,
        "choice_col": choice_col,
        "parameters": parameters,
        "utility": utility,
        "random_coefficients": case.random_names,
        "smooth_positive_scales": True,
        "n_draws": n_draws,
        "max_iterations": max_iter,
    }


def reorder_covariance(covariance, source_names, target_names: list[str]) -> np.ndarray | None:
    if covariance is None or source_names is None:
        return None
    try:
        matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
        ordered = matrix.loc[target_names, target_names].to_numpy(dtype=float)
    except Exception:
        return None
    return ordered if np.isfinite(ordered).all() else None


def r_env() -> dict[str, str]:
    env = os.environ.copy()
    r_user_lib = str(Path.home() / "R" / "site-library")
    existing = env.get("R_LIBS_USER")
    env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
    return env


def replay_probabilities(case: MixedCase, params: dict[str, float], draws: torch.Tensor) -> np.ndarray | None:
    try:
        model = MixedLogit(
            case.spec,
            [RandomCoefficient(name, sigma_init=0.1) for name in case.random_names],
            draws=draws,
            panel=False,
            device="cpu",
        )
        compiled = model.compile(case.data)
        vector = torch.as_tensor([params[name] for name in compiled.free_names], dtype=torch.float64)
        return model.predict_proba(case.data, vector, compiled).detach().cpu().numpy()
    except Exception:
        return None


def compare_results(case: MixedCase, results: list[BackendResult]) -> bool | None:
    torch_result = next((result for result in results if result.backend == "torchdcm" and result.available), None)
    completed = [
        result
        for result in results
        if result.available and result.loglike is not None and np.isfinite(result.loglike)
    ]
    if not completed:
        return None
    reference_loglike = max(float(result.loglike) for result in completed)
    likelihood_tolerance = max(
        0.25,
        1e-5 * abs(reference_loglike),
        0.01 * case.data.n_obs,
    )
    for result in results:
        if not result.available:
            continue
        if result.probabilities is None and result.params is not None:
            result.probabilities = replay_probabilities(case, result.params, make_draws_cached)
        result.ll_diff = (  # type: ignore[attr-defined]
            None
            if result.loglike is None or torch_result is None or torch_result.loglike is None
            else result.loglike - torch_result.loglike
        )
        common = sorted(set(result.params or {}) & set(torch_result.params or {})) if torch_result else []
        result.max_param_diff = (  # type: ignore[attr-defined]
            max((abs(result.params[name] - torch_result.params[name]) for name in common), default=None)
            if torch_result
            else None
        )
        if torch_result is not None and result.probabilities is not None and torch_result.probabilities is not None:
            result.max_prob_diff = float(np.max(np.abs(result.probabilities - torch_result.probabilities)))  # type: ignore[attr-defined]
        else:
            result.max_prob_diff = None  # type: ignore[attr-defined]
        result.worse_loglike = (  # type: ignore[attr-defined]
            result.loglike is not None
            and np.isfinite(result.loglike)
            and result.loglike < reference_loglike - likelihood_tolerance
        )
    comparable = [result for result in completed if not getattr(result, "worse_loglike", False)]
    if len(comparable) < 2:
        return None
    return all(
        abs(float(result.loglike) - reference_loglike) <= likelihood_tolerance
        for result in comparable
    )


make_draws_cached = torch.empty(0)


def make_draws(n_draws: int, seed: int, n_random: int) -> torch.Tensor:
    """Create the shared antithetic normal draws used by the MixL comparisons."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    half = (n_draws + 1) // 2
    base = torch.randn((half, n_random), generator=generator, dtype=torch.float64)
    return torch.cat([base, -base], dim=0)[:n_draws]


def build_cases(datasets: list[str], n_obs: int | None, sigma_scale: float) -> list[MixedCase | dict]:
    cases: list[MixedCase | dict] = []
    for name in datasets:
        try:
            if name == "swissmetro":
                cases.append(swissmetro_case(n_obs, sigma_scale))
            elif name in PUBLIC_CASE_BUILDERS:
                cases.append(public_case_to_mixed(name, n_obs, sigma_scale))
            elif name.startswith("mlogit_"):
                cases.append(mlogit_case_to_mixed(name.removeprefix("mlogit_"), sigma_scale))
            else:
                cases.append({"case": name, "status": "skipped", "message": "Unknown dataset key."})
        except Exception as exc:
            cases.append({"case": name, "status": "skipped", "message": f"{type(exc).__name__}: {exc}"})
    return cases


def run_case(case: MixedCase, args) -> dict:
    global make_draws_cached
    draws = make_draws(args.n_draws, args.seed, len(case.random_names))
    make_draws_cached = draws
    results = []
    try:
        start = prepare_starting_values(case)
    except Exception as exc:
        return {
            "case": case.case,
            "status": "skipped",
            "message": f"Starting-value preparation failed: {type(exc).__name__}: {exc}",
        }
    if not args.apollo_only:
        try:
            results.append(
                run_torch(
                    case,
                    draws,
                    args.max_iter,
                    args.torch_device,
                    start,
                )
            )
        except Exception as exc:
            results.append(
                BackendResult(
                    "torchdcm",
                    False,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
    if args.apollo_only:
        results.append(
            run_apollo(
                case,
                args.n_draws,
                args.backend_timeout,
                start,
                args.max_iter,
            )
        )
    elif not args.torch_only:
        results.append(
            run_biogeme(
                case,
                draws.detach().cpu().numpy(),
                args.max_iter,
                start,
            )
        )
        results.append(
            run_apollo(
                case,
                args.n_draws,
                args.backend_timeout,
                start,
                args.max_iter,
            )
        )
    consistent = compare_results(case, results)
    return payload(case, draws, results, consistent, start)


def payload(
    case: MixedCase,
    draws: torch.Tensor,
    results: list[BackendResult],
    consistent: bool | None,
    start: MixedStart,
) -> dict:
    return {
        "case": case.case,
        "dataset_id": case.dataset_id,
        "source": case.source,
        "model_name": case.model_name,
        "n_obs": case.data.n_obs,
        "n_rows": case.data.n_rows,
        "n_alternatives": len(case.alternatives),
        "n_draws": int(draws.shape[0]),
        "device_policy": "CPU for all cross-estimator comparisons; CUDA only for torch-only profiles.",
        "runtime_policy": runtime_policy_metadata(),
        "parameter_names": case.parameter_names,
        "random_coefficients": case.random_names,
        "starting_values": {
            "method": "MNL coefficient warm start with random scales set to sigma_scale / RMS(X_k)",
            "mean_coefficients": start.means,
            "random_scales": start.sigmas,
            "sigma_scale": start.sigma_scale,
            "mnl_loglike": start.mnl_loglike,
            "mnl_gradient_norm": start.mnl_gradient_norm,
        },
        "specification": {
            "alternatives": case.alternatives,
            "availability": case.availability_columns,
            "utilities": {
                alt: [
                    {"parameter": param, "variable": column, "random": param in case.random_names}
                    for param, column in case.utility_terms[alt]
                ]
                for alt in case.alternatives
            },
        },
        "consistent": "N.A." if consistent is None else ("Yes" if consistent else "No"),
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
                "worse_loglike": getattr(result, "worse_loglike", False),
                "params": result.params,
                "message": result.message,
            }
            for result in results
        ],
    }


def long_to_wide(df: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, dict[object, dict[str, str | int]], str]:
    alternatives = list(pd.unique(df["alt"]))
    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    safe_by_alt = {alt: safe_name(str(alt), i) for i, alt in enumerate(alternatives)}
    rows = []
    has_availability = "availability" in df.columns
    for obs_id, group in df.groupby("obs_id", sort=False):
        chosen = group.loc[group["choice"].astype(bool), "alt"]
        if chosen.empty:
            continue
        row = {"obs_id": obs_id, "choice_code": code_by_alt[chosen.iloc[0]]}
        by_alt = {alt: alt_group.iloc[0] for alt, alt_group in group.groupby("alt", sort=False)}
        for alt in alternatives:
            source = by_alt.get(alt)
            safe = safe_by_alt[alt]
            row[f"avail_{safe}"] = int(bool(source["availability"])) if source is not None and has_availability else int(source is not None)
            for variable in variables:
                row[f"{variable}_{safe}"] = float(source[variable]) if source is not None else 0.0
        rows.append(row)
    alt_columns = {
        alt: {
            "code": code_by_alt[alt],
            "availability": f"avail_{safe_by_alt[alt]}",
            **{variable: f"{variable}_{safe_by_alt[alt]}" for variable in variables},
        }
        for alt in alternatives
    }
    return pd.DataFrame(rows), alt_columns, "choice_code"


def safe_name(value: str, index: int) -> str:
    import re

    base = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_").lower() or f"alt_{index + 1}"
    if base[0].isdigit():
        base = f"alt_{base}"
    return f"{base}_{index + 1}"


def covariance_to_array(covariance_obj, names: list[str]) -> np.ndarray:
    if hasattr(covariance_obj, "loc"):
        return covariance_obj.loc[names, names].to_numpy(dtype=float)
    return np.asarray(covariance_obj, dtype=float)


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Real-data Mixed Logit Battery",
        "",
        "All cross-estimator runtimes report estimation plus covariance on one logical CPU. Each runnable model uses 2-4 independent normal random coefficients selected from observed-variable coefficients first, then ASC terms only when needed.",
        "A dagger marks a solver whose final log likelihood is clearly below the row best; its runtime is retained but its estimate is excluded from consistency. N.A. means fewer than two comparable estimates remain.",
        "",
        "| case | N | RC | TorchDCM s | Biogeme s | Apollo s | LL diff | Param diff | Prob diff | Consistent? |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row.get("status") == "skipped":
            lines.append(f"| {row['case']} | -- | skipped | -- | -- | -- | -- | -- | -- | -- |")
            continue
        torch_row = backend(row, "torchdcm")
        biogeme_row = backend(row, "biogeme")
        apollo_row = backend(row, "apollo")
        lines.append(
            "| {case} | {n} | {rc} | {torch_s} | {bio_s} | {apollo_s} | {ll} | {pd} | {prob} | {ok} |".format(
                case=row["case"],
                n=row["n_obs"],
                rc=", ".join(row["random_coefficients"]),
                torch_s=(fmt(torch_row.get("total_s")) + ("†" if torch_row.get("worse_loglike") else "")) if torch_row.get("available") else "Fail",
                bio_s=(fmt(biogeme_row.get("total_s")) + ("†" if biogeme_row.get("worse_loglike") else "")) if biogeme_row.get("available") else "Fail",
                apollo_s=(
                    fmt(apollo_row.get("total_s"))
                    + ("†" if apollo_row.get("worse_loglike") else "")
                    if apollo_row.get("available")
                    else "Fail"
                ),
                ll=sci(biogeme_row.get("ll_diff")),
                pd=sci(biogeme_row.get("max_param_diff")),
                prob=sci(biogeme_row.get("max_prob_diff")),
                ok=row["consistent"],
            )
        )
    lines.extend(["", "## Specifications", ""])
    for row in rows:
        if row.get("status") == "skipped":
            lines.append(f"- `{row['case']}` skipped: {row.get('message')}")
            continue
        lines.append(f"- `{row['case']}`: random coefficients = {', '.join(row['random_coefficients'])}; parameters = {', '.join(row['parameter_names'])}.")
    return "\n".join(lines)


def backend(row: dict, name: str) -> dict:
    return next((item for item in row.get("backends", []) if item["backend"] == name), {"available": False})


def fmt(value) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "NA"


def sci(value) -> str:
    return f"{float(value):.2e}" if isinstance(value, (int, float)) else "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["swissmetro", "airline", "parking", "telephone", "lpmc", *[f"mlogit_{name}" for name in MLOGIT_DATASETS]],
    )
    parser.add_argument("--n-obs", type=int, default=None)
    parser.add_argument("--n-draws", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--sigma-scale",
        "--sigma",
        dest="sigma_scale",
        type=float,
        default=1.0,
        help="Initial random scale multiplier in sigma_scale / RMS(X_k).",
    )
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--backend-timeout", type=float, default=300.0)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--torch-only", action="store_true")
    parser.add_argument("--apollo-only", action="store_true")
    parser.add_argument("--profile", default="full")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--md-output", type=Path)
    args = parser.parse_args()
    if args.torch_only and args.apollo_only:
        raise ValueError("--torch-only and --apollo-only cannot be used together.")
    if not args.torch_only and torch.device(args.torch_device).type != "cpu":
        raise ValueError("Cross-estimator mixed-logit batteries must use --torch-device cpu.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = args.json_output or RESULTS / f"mixed_real_battery_{args.profile}.json"
    md_path = args.md_output or RESULTS / f"mixed_real_battery_{args.profile}.md"
    rows = []
    for case_or_skip in build_cases(args.datasets, args.n_obs, args.sigma_scale):
        if isinstance(case_or_skip, dict):
            rows.append(case_or_skip)
            print(f"{case_or_skip['case']}: skipped {case_or_skip.get('message')}", flush=True)
            json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            md_path.write_text(render_markdown(rows), encoding="utf-8")
            continue
        print(f"{case_or_skip.case}: RC={case_or_skip.random_names}", flush=True)
        row = run_case(case_or_skip, args)
        rows.append(row)
        torch_result = backend(row, "torchdcm")
        biogeme_result = backend(row, "biogeme")
        apollo_result = backend(row, "apollo")
        print(
            f"{row['case']}: "
            f"torch={fmt(torch_result.get('total_s')) if torch_result.get('available') else '--'} "
            f"biogeme={fmt(biogeme_result.get('total_s')) if biogeme_result.get('available') else '--'} "
            f"apollo={fmt(apollo_result.get('total_s')) if apollo_result.get('available') else '--'} "
            f"consistent={row['consistent']}",
            flush=True,
        )
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"json: {json_path}", flush=True)
    print(f"markdown: {md_path}", flush=True)


if __name__ == "__main__":
    main()
