from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import shutil
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from benchmark_runtime import (
    configure_single_thread_cpu,
    estimation_covariance_total,
    runtime_policy_metadata,
)

if __name__ == "__main__":
    configure_single_thread_cpu(configure_torch=True)

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import torch

from torchdcm import Beta, ChoiceDataset, MixedLogit, MultinomialLogit, RandomCoefficient, UtilitySpec

import compare_real_mixed_logit_battery as mixed_real
import compare_real_nested_logit_battery as nested_real
from mnl_generic_backends import (
    make_design_long,
    run_gmnl_generic,
    run_mlogit_generic,
    run_scipy_mle,
    run_xlogit_generic,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"
APOLLO_MNL_SCRIPT = ROOT / "benchmarks" / "apollo" / "R" / "run_generic_mnl.R"
APOLLO_MIXED_SCRIPT = ROOT / "benchmarks" / "apollo" / "R" / "run_generic_mixed_estimate.R"
warnings.filterwarnings("ignore", category=PerformanceWarning)

NL_TRUE_DISSIMILARITY = {"GROUP_A": 0.65, "GROUP_B": 0.80}
LL_ABS_TOL = 0.25
LL_REL_TOL = 1e-5
LL_PER_OBSERVATION_TOL = 0.01
MIXL_SIGMA_MAX = 0.40
MIXL_SIGMA_MIN = 0.20
MIXL_SIGMA_INIT = 0.30


@dataclass(frozen=True)
class GeneratedSpec:
    case: str
    model: str
    n_obs: int
    n_alternatives: int
    n_variables: int
    rho: float
    random_coefficients: int = 0


@dataclass
class MNLCase:
    case: str
    df: pd.DataFrame
    data: ChoiceDataset
    spec: UtilitySpec
    alternatives: list[str]
    feature_columns: dict[str, dict[str, str]]
    availability_columns: dict[str, str]
    parameter_names: list[str]
    initial_values: dict[str, float]
    true_parameters: dict[str, float]
    systematic_utility: np.ndarray
    feature_values: np.ndarray
    meta: GeneratedSpec


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
    closure_evals: int | None = None
    optimizer_iterations: int | None = None
    runtime_repeats_s: list[float | None] | None = None
    worse_loglike: bool = False


def generated_specs(profile: str) -> list[GeneratedSpec]:
    if profile == "paper":
        # The 45 cases reported in the paper: 15 each for MNL, NL, and MixL.
        return [
            *generated_specs("controlled"),
            *generated_specs("stress_additional"),
            generated_specs("stress")[0],
            *generated_specs("table4"),
        ]
    if profile == "smoke":
        return [
            GeneratedSpec("gen_mnl_base", "mnl", 400, 3, 3, 0.0),
            GeneratedSpec("gen_nl_base", "nl", 400, 4, 3, 0.0),
            GeneratedSpec("gen_mixl_base", "mixl", 300, 3, 3, 0.0, random_coefficients=2),
        ]
    if profile == "stress":
        return [
            GeneratedSpec("stress_mnl_large", "mnl", 50000, 35, 20, 0.5),
            GeneratedSpec("stress_nl_NJK", "nl", 50000, 20, 12, 0.5),
            GeneratedSpec("stress_mixl_NJK", "mixl", 40000, 20, 12, 0.5, random_coefficients=6),
        ]
    if profile == "stress_additional":
        return [
            GeneratedSpec("stress_mnl_small", "mnl", 30000, 20, 12, 0.5),
            GeneratedSpec("stress_mnl_medium", "mnl", 40000, 28, 16, 0.5),
        ]
    if profile == "controlled":
        return [
            GeneratedSpec("N_1000", "mnl", 1000, 4, 6, 0.3),
            GeneratedSpec("N_10000", "mnl", 10000, 4, 6, 0.3),
            GeneratedSpec("N_100000", "mnl", 100000, 4, 6, 0.3),
            GeneratedSpec("J_3", "mnl", 20000, 3, 6, 0.3),
            GeneratedSpec("J_10", "mnl", 20000, 10, 6, 0.3),
            GeneratedSpec("J_20", "mnl", 20000, 20, 6, 0.3),
            GeneratedSpec("K_4", "mnl", 20000, 5, 4, 0.3),
            GeneratedSpec("K_12", "mnl", 20000, 5, 12, 0.3),
            GeneratedSpec("K_32", "mnl", 20000, 5, 32, 0.3),
            GeneratedSpec("rho_0p0", "mnl", 20000, 5, 12, 0.0),
            GeneratedSpec("rho_0p5", "mnl", 20000, 5, 12, 0.5),
            GeneratedSpec("rho_0p98", "mnl", 20000, 5, 12, 0.98),
        ]
    if profile == "controlled_additional":
        return [
            GeneratedSpec("J_10", "mnl", 20000, 10, 6, 0.3),
            GeneratedSpec("K_12", "mnl", 20000, 5, 12, 0.3),
            GeneratedSpec("rho_0p5", "mnl", 20000, 5, 12, 0.5),
        ]
    if profile == "table4":
        controlled_grid = [
            ("N_1000", 1000, 4, 6, 0.3),
            ("N_10000", 10000, 4, 6, 0.3),
            ("N_100000", 100000, 4, 6, 0.3),
            ("C_3", 20000, 3, 6, 0.3),
            ("C_10", 20000, 10, 6, 0.3),
            ("C_20", 20000, 20, 6, 0.3),
            ("K_4", 20000, 5, 4, 0.3),
            ("K_12", 20000, 5, 12, 0.3),
            ("K_32", 20000, 5, 32, 0.3),
            ("rho_0p0", 20000, 5, 12, 0.0),
            ("rho_0p5", 20000, 5, 12, 0.5),
            ("rho_0p98", 20000, 5, 12, 0.98),
        ]
        rows = [
            GeneratedSpec(f"nl_{case}", "nl", n_obs, n_alt, n_var, rho)
            for case, n_obs, n_alt, n_var, rho in controlled_grid
        ]
        rows.extend(
            [
                GeneratedSpec("nl_stress_small", "nl", 30000, 12, 8, 0.5),
                GeneratedSpec("nl_stress_medium", "nl", 40000, 16, 10, 0.5),
                GeneratedSpec("stress_nl_NJK", "nl", 50000, 20, 12, 0.5),
            ]
        )
        mixed_random_counts = {
            "N_1000": 3,
            "N_10000": 3,
            "N_100000": 3,
            "C_3": 3,
            "C_10": 3,
            "C_20": 3,
            "K_4": 2,
            "K_12": 6,
            "K_32": 16,
            "rho_0p0": 6,
            "rho_0p5": 6,
            "rho_0p98": 6,
        }
        rows.extend(
            GeneratedSpec(
                f"mixl_{case}",
                "mixl",
                n_obs,
                n_alt,
                n_var,
                rho,
                random_coefficients=mixed_random_counts[case],
            )
            for case, n_obs, n_alt, n_var, rho in controlled_grid
        )
        rows.extend(
            [
                GeneratedSpec("mixl_stress_small", "mixl", 20000, 12, 8, 0.5, random_coefficients=4),
                GeneratedSpec("mixl_stress_medium", "mixl", 30000, 16, 10, 0.5, random_coefficients=5),
                GeneratedSpec("stress_mixl_NJK", "mixl", 40000, 20, 12, 0.5, random_coefficients=6),
            ]
        )
        return rows
    return [
        GeneratedSpec("gen_mnl_base", "mnl", 1000, 3, 4, 0.0),
        GeneratedSpec("gen_mnl_N", "mnl", 10000, 3, 4, 0.0),
        GeneratedSpec("gen_mnl_J", "mnl", 3000, 8, 4, 0.0),
        GeneratedSpec("gen_mnl_K", "mnl", 3000, 4, 12, 0.0),
        GeneratedSpec("gen_mnl_rho", "mnl", 3000, 4, 6, 0.8),
        GeneratedSpec("gen_nl_base", "nl", 1000, 4, 4, 0.0),
        GeneratedSpec("gen_nl_N", "nl", 5000, 4, 4, 0.0),
        GeneratedSpec("gen_nl_J", "nl", 2500, 8, 4, 0.0),
        GeneratedSpec("gen_nl_K", "nl", 2500, 4, 8, 0.0),
        GeneratedSpec("gen_nl_rho", "nl", 2500, 4, 6, 0.8),
        GeneratedSpec("gen_mixl_base", "mixl", 1000, 3, 4, 0.0, random_coefficients=2),
        GeneratedSpec("gen_mixl_N", "mixl", 3000, 3, 4, 0.0, random_coefficients=2),
        GeneratedSpec("gen_mixl_J", "mixl", 1500, 5, 4, 0.0, random_coefficients=2),
        GeneratedSpec("gen_mixl_K", "mixl", 1500, 4, 8, 0.0, random_coefficients=4),
        GeneratedSpec("gen_mixl_rho", "mixl", 1500, 4, 6, 0.8, random_coefficients=3),
    ]


def build_mnl_case(meta: GeneratedSpec, seed: int) -> MNLCase:
    rng = np.random.default_rng(seed + stable_case_offset(meta.case))
    alternatives = [f"ALT{j + 1}" for j in range(meta.n_alternatives)]
    features = [f"x{k + 1}" for k in range(meta.n_variables)]
    beta_names = [f"B_X{k + 1}" for k in range(meta.n_variables)]
    asc_names = [f"ASC_ALT{j + 1}" for j in range(1, meta.n_alternatives)]
    parameter_names = [*beta_names, *asc_names]
    true_beta = -np.linspace(0.25, 1.1, meta.n_variables)
    true_asc = np.linspace(0.35, -0.35, meta.n_alternatives)
    true_asc = true_asc - true_asc[0]
    x = correlated_features(rng, meta.n_obs, meta.n_alternatives, meta.n_variables, meta.rho)
    utility = np.einsum("njk,k->nj", x, true_beta) + true_asc
    probabilities = softmax(utility)
    choices = sample_choices(rng, probabilities)

    columns: dict[str, object] = {"obs_id": np.arange(meta.n_obs), "choice": [alternatives[i] for i in choices]}
    feature_columns: dict[str, dict[str, str]] = {feature: {} for feature in features}
    availability_columns: dict[str, str] = {}
    for j, alt in enumerate(alternatives):
        alt_key = alt.lower()
        columns[f"avail_{alt_key}"] = True
        availability_columns[alt] = f"avail_{alt_key}"
        for k, feature in enumerate(features):
            col = f"{feature}_{alt_key}"
            columns[col] = x[:, j, k]
            feature_columns[feature][alt] = col
    df = pd.DataFrame(columns)

    data = ChoiceDataset.from_wide(
        df,
        alternatives=alternatives,
        choice="choice",
        variables=feature_columns,
        availability=availability_columns,
        obs_id="obs_id",
    )
    spec = UtilitySpec()
    for alt_index, alt in enumerate(alternatives):
        utility_expr = None
        if alt_index > 0:
            utility_expr = Beta(f"ASC_{alt}")
        for feature, beta_name in zip(features, beta_names):
            term = Beta(beta_name) * feature
            utility_expr = term if utility_expr is None else utility_expr + term
        spec.utility(alt, utility_expr if utility_expr is not None else 0)
    true_parameters = {name: float(value) for name, value in zip(beta_names, true_beta)}
    true_parameters.update({f"ASC_{alternatives[j]}": float(true_asc[j]) for j in range(1, meta.n_alternatives)})
    return MNLCase(
        case=meta.case,
        df=df,
        data=data,
        spec=spec,
        alternatives=alternatives,
        feature_columns=feature_columns,
        availability_columns=availability_columns,
        parameter_names=parameter_names,
        initial_values={name: 0.0 for name in parameter_names},
        true_parameters=true_parameters,
        systematic_utility=utility,
        feature_values=x,
        meta=meta,
    )


def correlated_features(rng: np.random.Generator, n_obs: int, n_alt: int, n_var: int, rho: float) -> np.ndarray:
    covariance = np.full((n_var, n_var), rho, dtype=float)
    np.fill_diagonal(covariance, 1.0)
    raw = rng.multivariate_normal(np.zeros(n_var), covariance, size=n_obs * n_alt)
    return raw.reshape(n_obs, n_alt, n_var)


def softmax(utility: np.ndarray) -> np.ndarray:
    shifted = utility - utility.max(axis=1, keepdims=True)
    exp_utility = np.exp(shifted)
    return exp_utility / exp_utility.sum(axis=1, keepdims=True)


def sample_choices(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.cumsum(probabilities, axis=1)
    draws = rng.random(probabilities.shape[0])[:, None]
    return (draws > thresholds).sum(axis=1)


def nested_choice_probabilities(
    utility: np.ndarray,
    nest_indices: list[list[int]],
    dissimilarities: list[float],
) -> np.ndarray:
    """Evaluate two-level nested-logit probabilities with a unit root scale."""
    if len(nest_indices) != len(dissimilarities):
        raise ValueError("Each nest must have one dissimilarity parameter.")
    if sorted(index for nest in nest_indices for index in nest) != list(range(utility.shape[1])):
        raise ValueError("Nested-logit nests must partition all alternatives exactly once.")

    nest_log_sums: list[np.ndarray] = []
    for indices, dissimilarity in zip(nest_indices, dissimilarities):
        if not 0.0 < dissimilarity <= 1.0:
            raise ValueError("Nested-logit dissimilarities must lie in (0, 1].")
        scaled = utility[:, indices] / dissimilarity
        nest_log_sums.append(np.logaddexp.reduce(scaled, axis=1))

    log_nest_weights = np.column_stack(
        [dissimilarity * log_sum for dissimilarity, log_sum in zip(dissimilarities, nest_log_sums)]
    )
    log_denominator = np.logaddexp.reduce(log_nest_weights, axis=1)
    log_probabilities = np.full_like(utility, -np.inf)
    for indices, dissimilarity, log_sum in zip(nest_indices, dissimilarities, nest_log_sums):
        log_probabilities[:, indices] = (
            utility[:, indices] / dissimilarity
            + (dissimilarity - 1.0) * log_sum[:, None]
            - log_denominator[:, None]
        )
    probabilities = np.exp(log_probabilities)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def stable_case_offset(case: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(case))


def run_mnl_torch(case: MNLCase, max_iter: int) -> BackendResult:
    model = MultinomialLogit(case.spec, max_iter=max_iter)
    data = case.data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    initial = torch.as_tensor([case.initial_values[name] for name in compiled.free_names], dtype=torch.float64)

    # Exclude one-time tensor-kernel and autograd initialization from the
    # optimizer timing while preserving the same zero starting values.
    warmup_params = initial.clone().requires_grad_(True)
    (-model.loglike(warmup_params, data, compiled)).backward()

    params = initial.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )
    closure_evals = 0

    def closure():
        nonlocal closure_evals
        closure_evals += 1
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
    optimizer_state = optimizer.state.get(params, {})
    return BackendResult(
        "torchdcm",
        True,
        total_s=estimate_s + covariance_s,
        estimate_s=estimate_s,
        covariance_s=covariance_s,
        loglike=loglike,
        params={name: float(final[i].detach().cpu()) for i, name in enumerate(compiled.free_names)},
        covariance=covariance,
        probabilities=model.predict_proba(data, final, compiled).detach().cpu().numpy(),
        closure_evals=closure_evals,
        optimizer_iterations=int(optimizer_state.get("n_iter", 0)),
    )


def run_mnl_biogeme(case: MNLCase) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme import models
        from biogeme.expressions import Beta as BioBeta
        from biogeme.expressions import Variable
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult("biogeme", False, message=f"Biogeme unavailable: {exc}")

    try:
        df, code_by_alt = wide_for_external(case)
        database = db.Database(f"torchdcm_{case.case}", df)
        betas = {name: BioBeta(name, case.initial_values.get(name, 0.0), None, None, 0) for name in case.parameter_names}
        utility = {}
        availability = {}
        for alt, code in code_by_alt.items():
            expr = 0
            asc = f"ASC_{alt}"
            if asc in betas:
                expr += betas[asc]
            for feature, beta_name in zip(case.feature_columns, beta_names(case)):
                expr += betas[beta_name] * Variable(case.feature_columns[feature][alt])
            utility[code] = expr
            availability[code] = Variable(case.availability_columns[alt])
        logprob = models.loglogit(utility, availability, Variable("choice_code"))
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = f"torchdcm_generated_mnl_{case.case}_{case.data.n_obs}"
        biogeme.biogeme_parameters.set_value("save_iterations", False)
        total_start = time.perf_counter()
        estimate_start = time.perf_counter()
        estimates = biogeme.estimate()
        estimate_s = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance = covariance_to_array(
            estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER),
            case.parameter_names,
        )
        covariance_s = time.perf_counter() - covariance_start
        beta_values = estimates.get_beta_values()
        return BackendResult(
            "biogeme",
            True,
            total_s=estimation_covariance_total(estimate_s, covariance_s),
            estimate_s=estimate_s,
            covariance_s=covariance_s,
            loglike=float(estimates.final_log_likelihood),
            params={name: float(beta_values[name]) for name in case.parameter_names},
            covariance=covariance,
        )
    except Exception as exc:
        return BackendResult("biogeme", False, message=f"{type(exc).__name__}: {exc}")


def run_mnl_apollo(case: MNLCase) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult("apollo", False, message="Rscript not found.")
    if not APOLLO_MNL_SCRIPT.exists():
        return BackendResult("apollo", False, message=f"Missing Apollo script: {APOLLO_MNL_SCRIPT}")
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_generated_mnl_{case.case}_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        spec_path = tmp_path / "spec.json"
        output_path = tmp_path / "apollo_result.json"
        df, spec = apollo_mnl_inputs(case)
        df.to_csv(data_path, index=False)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        command = [rscript, str(APOLLO_MNL_SCRIPT), "--data", str(data_path), "--spec", str(spec_path), "--output", str(output_path)]
        start = time.perf_counter()
        proc = subprocess.run(command, text=True, capture_output=True, env=r_env())
        total_s = time.perf_counter() - start
        if proc.returncode != 0:
            return BackendResult("apollo", False, total_s=total_s, message=(proc.stderr or proc.stdout).strip())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        estimate_s = payload.get("timing", {}).get("estimate_seconds")
        covariance_s = payload.get("timing", {}).get("covariance_seconds")
        return BackendResult(
            "apollo",
            True,
            total_s=estimation_covariance_total(estimate_s, covariance_s),
            estimate_s=estimate_s,
            covariance_s=covariance_s,
            loglike=float(payload["loglike"]),
            params={name: float(payload["estimates"][name]) for name in case.parameter_names},
            covariance=reorder_covariance_or_none(payload.get("covariance"), payload.get("covariance_names"), case.parameter_names),
            message=f"apollo_version={payload.get('apollo_version')}",
        )


def run_mnl_scipy(case: MNLCase) -> BackendResult:
    return ns_to_backend(run_scipy_mle(case.data, case.spec, case.initial_values, target_names=case.parameter_names), "scipy_bfgs")


def run_mnl_mlogit(case: MNLCase) -> BackendResult:
    return ns_to_backend(run_mlogit_generic(mnl_design_long(case), case.parameter_names), "mlogit")


def run_mnl_gmnl(case: MNLCase) -> BackendResult:
    return ns_to_backend(run_gmnl_generic(mnl_design_long(case), case.parameter_names), "gmnl")


def run_mnl_xlogit(case: MNLCase) -> BackendResult:
    return ns_to_backend(run_xlogit_generic(mnl_design_long(case), case.parameter_names), "xlogit")


def mnl_design_long(case: MNLCase) -> pd.DataFrame:
    utility_columns: dict[str, dict[str, str | float]] = {}
    for alt in case.alternatives:
        terms: dict[str, str | float] = {}
        asc = f"ASC_{alt}"
        if asc in case.parameter_names:
            terms[asc] = 1.0
        for feature, beta_name in zip(case.feature_columns, beta_names(case)):
            terms[beta_name] = case.feature_columns[feature][alt]
        utility_columns[alt] = terms
    return make_design_long(
        case.df,
        case.alternatives,
        "choice",
        utility_columns,
        case.availability_columns,
        case.parameter_names,
    )


def build_nested_case(mnl: MNLCase, seed: int) -> nested_real.NestedCase:
    midpoint = max(2, len(mnl.alternatives) // 2)
    group_a = mnl.alternatives[:midpoint]
    group_b = mnl.alternatives[midpoint:]
    nests = {
        "GROUP_A": nested_real.NestSpec(group_a, init=1.0 if len(group_a) == 1 else 0.8, fixed=len(group_a) == 1),
        "GROUP_B": nested_real.NestSpec(group_b, init=1.0 if len(group_b) == 1 else 0.8, fixed=len(group_b) == 1),
    }
    alternative_index = {alternative: index for index, alternative in enumerate(mnl.alternatives)}
    nest_indices = [[alternative_index[alternative] for alternative in nest.alternatives] for nest in nests.values()]
    true_dissimilarities = [
        1.0 if nest.fixed else NL_TRUE_DISSIMILARITY[nest_name]
        for nest_name, nest in nests.items()
    ]
    probabilities = nested_choice_probabilities(mnl.systematic_utility, nest_indices, true_dissimilarities)
    rng = np.random.default_rng(seed + stable_case_offset(mnl.case) + 10_000_019)
    choices = sample_choices(rng, probabilities)
    nested_df = mnl.df.copy()
    nested_df["choice"] = [mnl.alternatives[index] for index in choices]
    nested_mnl = replace(mnl, df=nested_df)
    long_df = mnl_design_long(nested_mnl)
    dgp_text = ", ".join(
        f"{nest_name}={dissimilarity:.2f}"
        for nest_name, dissimilarity in zip(nests, true_dissimilarities)
    )
    return nested_real.nested_case_from_design_long(
        case=mnl.case,
        data_label=mnl.case,
        model_label="Generated nested logit",
        source=(
            f"synthetic NL N={mnl.meta.n_obs} J={mnl.meta.n_alternatives} "
            f"K={mnl.meta.n_variables} rho={mnl.meta.rho}; {dgp_text}"
        ),
        long_df=long_df,
        alternatives=mnl.alternatives,
        beta_names=mnl.parameter_names,
        nests=nests,
    )


def mixed_true_sigmas(random_names: list[str]) -> dict[str, float]:
    values = np.linspace(MIXL_SIGMA_MAX, MIXL_SIGMA_MIN, len(random_names))
    return {name: float(value) for name, value in zip(random_names, values)}


def build_mixed_case(mnl: MNLCase, seed: int) -> mixed_real.MixedCase:
    random_names = beta_names(mnl)[: max(2, min(mnl.meta.random_coefficients, len(beta_names(mnl))))]
    true_sigmas = mixed_true_sigmas(random_names)
    random_indices = [beta_names(mnl).index(name) for name in random_names]
    random_scales = np.asarray([true_sigmas[name] for name in random_names])
    rng = np.random.default_rng(seed + stable_case_offset(mnl.case) + 20_000_033)
    taste_deviations = rng.standard_normal((mnl.meta.n_obs, len(random_names))) * random_scales
    utility = mnl.systematic_utility + np.einsum(
        "njr,nr->nj",
        mnl.feature_values[:, :, random_indices],
        taste_deviations,
    )
    probabilities = softmax(utility)
    choices = sample_choices(rng, probabilities)
    mixed_df = mnl.df.copy()
    mixed_df["choice"] = [mnl.alternatives[index] for index in choices]
    mixed_data = ChoiceDataset.from_wide(
        mixed_df,
        alternatives=mnl.alternatives,
        choice="choice",
        variables=mnl.feature_columns,
        availability=mnl.availability_columns,
        obs_id="obs_id",
    )

    utility_terms: dict[str, list[tuple[str, str | None]]] = {}
    for alt in mnl.alternatives:
        terms: list[tuple[str, str | None]] = []
        asc = f"ASC_{alt}"
        if asc in mnl.parameter_names:
            terms.append((asc, None))
        for feature, beta_name in zip(mnl.feature_columns, beta_names(mnl)):
            terms.append((beta_name, mnl.feature_columns[feature][alt]))
        utility_terms[alt] = terms
    sigma_text = ", ".join(f"SIGMA_{name}={true_sigmas[name]:.3f}" for name in random_names)
    return mixed_real.MixedCase(
        case=mnl.case,
        dataset_id=mnl.case,
        source=(
            f"synthetic MixL N={mnl.meta.n_obs} J={mnl.meta.n_alternatives} "
            f"K={mnl.meta.n_variables} rho={mnl.meta.rho}; {sigma_text}"
        ),
        model_name="Generated mixed logit",
        df=mixed_df,
        data=mixed_data,
        spec=mnl.spec,
        alternatives=mnl.alternatives,
        choice_col="choice",
        utility_terms=utility_terms,
        availability_columns=mnl.availability_columns,
        parameter_names=mnl.parameter_names,
        random_names=random_names,
        sigma_scale=MIXL_SIGMA_INIT,
    )


def run_mixed_apollo(case: mixed_real.MixedCase, n_draws: int) -> mixed_real.BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return mixed_real.BackendResult("apollo", False, message="Rscript not found.")
    if not APOLLO_MIXED_SCRIPT.exists():
        return mixed_real.BackendResult("apollo", False, message=f"Missing Apollo script: {APOLLO_MIXED_SCRIPT}")
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_generated_mixl_{case.case}_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        spec_path = tmp_path / "spec.json"
        output_path = tmp_path / "apollo_result.json"
        df, spec = apollo_mixed_inputs(case, n_draws)
        df.to_csv(data_path, index=False)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        command = [rscript, str(APOLLO_MIXED_SCRIPT), "--data", str(data_path), "--spec", str(spec_path), "--output", str(output_path)]
        start = time.perf_counter()
        proc = subprocess.run(command, text=True, capture_output=True, env=r_env())
        total_s = time.perf_counter() - start
        if proc.returncode != 0:
            return mixed_real.BackendResult("apollo", False, total_s=total_s, message=(proc.stderr or proc.stdout).strip())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        names = [*case.parameter_names, *[f"SIGMA_{name}" for name in case.random_names]]
        covariance = reorder_covariance_or_none(payload.get("covariance"), payload.get("covariance_names"), names)
        estimate_s = payload.get("timing", {}).get("estimate_seconds")
        covariance_s = payload.get("timing", {}).get("covariance_seconds")
        convergence = payload.get("convergence") or {}
        convergence_status = convergence.get("status")
        convergence_message = convergence.get("message")
        message = (
            f"apollo_version={payload.get('apollo_version')}; "
            f"apollo_halton_draws={n_draws}; "
            f"convergence_status={convergence_status}; "
            f"convergence_message={convergence_message}"
        )
        if "unfavorable" in str(convergence_message).lower():
            return mixed_real.BackendResult(
                "apollo",
                False,
                total_s=estimation_covariance_total(estimate_s, covariance_s),
                estimate_s=estimate_s,
                covariance_s=covariance_s,
                message=message,
            )
        return mixed_real.BackendResult(
            "apollo",
            True,
            total_s=estimation_covariance_total(estimate_s, covariance_s),
            estimate_s=estimate_s,
            covariance_s=covariance_s,
            loglike=float(payload["loglike"]),
            params={name: float(payload["estimates"][name]) for name in names},
            covariance=covariance,
            message=message,
        )


def apollo_mnl_inputs(case: MNLCase) -> tuple[pd.DataFrame, dict]:
    df, code_by_alt = wide_for_external(case)
    utility = {}
    for alt, code in code_by_alt.items():
        asc = f"ASC_{alt}"
        utility[alt] = {
            "code": code,
            "asc": asc if asc in case.parameter_names else None,
            "availability": case.availability_columns[alt],
            "variables": {
                beta_name: case.feature_columns[feature][alt]
                for feature, beta_name in zip(case.feature_columns, beta_names(case))
            },
        }
    return df, {
        "model_name": f"apollo_generated_mnl_{case.case}_{case.data.n_obs}",
        "alternatives": case.alternatives,
        "choice_col": "choice_code",
        "parameters": {name: case.initial_values[name] for name in case.parameter_names},
        "utility": utility,
    }


def apollo_mixed_inputs(case: mixed_real.MixedCase, n_draws: int) -> tuple[pd.DataFrame, dict]:
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
    parameters = {name: 0.0 for name in case.parameter_names}
    parameters.update({f"SIGMA_{name}": case.sigma_scale for name in case.random_names})
    utility = {}
    for alt in case.alternatives:
        utility[alt] = {
            "code": code_by_alt[alt],
            "availability": case.availability_columns[alt],
            "terms": [
                {"parameter": param, "variable": column}
                for param, column in case.utility_terms[alt]
            ],
        }
    return df, {
        "model_name": f"apollo_generated_mixl_{case.case}_{case.data.n_obs}",
        "alternatives": case.alternatives,
        "choice_col": choice_col,
        "parameters": parameters,
        "utility": utility,
        "random_coefficients": case.random_names,
        "n_draws": n_draws,
    }


def wide_for_external(case: MNLCase) -> tuple[pd.DataFrame, dict[str, int]]:
    code_by_alt = {alt: index + 1 for index, alt in enumerate(case.alternatives)}
    df = case.df.copy()
    df["choice_code"] = df["choice"].map(code_by_alt).astype(int)
    df = df.drop(columns=["choice"])
    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)
    return df, code_by_alt


def beta_names(case: MNLCase) -> list[str]:
    return [f"B_X{k + 1}" for k in range(case.meta.n_variables)]


def loglike_tolerance(best_loglike: float, n_obs: int) -> float:
    return max(
        LL_ABS_TOL,
        LL_REL_TOL * abs(best_loglike),
        LL_PER_OBSERVATION_TOL * n_obs,
    )


def mark_worse_loglikes(results: list, n_obs: int) -> float | None:
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
        result.worse_loglike = bool(float(result.loglike) < best_loglike - tolerance)
    return tolerance


def loglike_consistency(results: list, n_obs: int) -> bool | None:
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


def compare_mnl(case: MNLCase, results: list[BackendResult]) -> bool | None:
    ref = next((result for result in results if result.backend == "torchdcm" and result.available), None)
    if ref is None:
        mark_worse_loglikes(results, case.data.n_obs)
        return loglike_consistency(results, case.data.n_obs)
    for result in results:
        if result.available and result.probabilities is None and result.params is not None:
            result.probabilities = predict_mnl_probabilities(case, result.params)
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        result.max_param_diff = max(abs(result.params[name] - ref.params[name]) for name in case.parameter_names)  # type: ignore[attr-defined]
        result.max_prob_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))  # type: ignore[attr-defined]
        if result.covariance is not None and ref.covariance is not None:
            result.max_cov_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
            result.max_se_diff = float(np.max(np.abs(np.sqrt(np.diag(result.covariance)) - np.sqrt(np.diag(ref.covariance)))))  # type: ignore[attr-defined]
        else:
            result.max_cov_diff = None  # type: ignore[attr-defined]
            result.max_se_diff = None  # type: ignore[attr-defined]
    mark_worse_loglikes(results, case.data.n_obs)
    return loglike_consistency(results, case.data.n_obs)


def predict_mnl_probabilities(case: MNLCase, params: dict[str, float]) -> np.ndarray:
    model = MultinomialLogit(case.spec)
    vector = torch.as_tensor([params[name] for name in case.parameter_names], dtype=torch.float64)
    return model.predict_proba(case.data, vector).detach().cpu().numpy()


def ns_to_backend(result: SimpleNamespace, name: str) -> BackendResult:
    return BackendResult(
        name,
        bool(result.available),
        total_s=getattr(result, "total_s", getattr(result, "seconds", None)),
        estimate_s=getattr(result, "estimate_s", getattr(result, "estimate_seconds", None)),
        covariance_s=getattr(result, "covariance_s", getattr(result, "covariance_seconds", None)),
        loglike=getattr(result, "loglike", None),
        params=getattr(result, "params", None),
        covariance=getattr(result, "covariance", None),
        probabilities=getattr(result, "probabilities", None),
        message=getattr(result, "message", ""),
    )


def safe_run(backend: str, fn: Callable[[], BackendResult]) -> BackendResult:
    start = time.perf_counter()
    try:
        return fn()
    except Exception as exc:
        return BackendResult(backend, False, total_s=time.perf_counter() - start, message=f"{type(exc).__name__}: {exc}")


def safe_run_mixed(backend: str, fn: Callable[[], mixed_real.BackendResult]) -> mixed_real.BackendResult:
    start = time.perf_counter()
    try:
        return fn()
    except Exception as exc:
        return mixed_real.BackendResult(backend, False, total_s=time.perf_counter() - start, message=f"{type(exc).__name__}: {exc}")


def safe_run_generic(backend: str, fn, result_class):
    start = time.perf_counter()
    try:
        return fn()
    except Exception as exc:
        return result_class(backend, False, total_s=time.perf_counter() - start, message=f"{type(exc).__name__}: {exc}")


def timed_safe_run(backend: str, fn, result_class, timeout_s: int | None):
    if not timeout_s or timeout_s <= 0:
        return safe_run_generic(backend, fn, result_class)
    start = time.perf_counter()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"{backend} exceeded {timeout_s}s timeout")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn()
    except TimeoutError as exc:
        return result_class(backend, False, total_s=time.perf_counter() - start, message=str(exc))
    except Exception as exc:
        return result_class(backend, False, total_s=time.perf_counter() - start, message=f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def isolated_timed_safe_run(backend: str, fn, result_class, timeout_s: int | None):
    context = mp.get_context("fork")
    queue: mp.Queue = context.Queue(maxsize=1)

    def _worker():
        try:
            os.setsid()
            result = fn()
            result.probabilities = None
            queue.put(("ok", result))
        except Exception as exc:
            queue.put(("error", f"{type(exc).__name__}: {exc}"))

    start = time.perf_counter()
    process = context.Process(target=_worker, name=f"torchdcm_{backend}_benchmark")
    process.start()
    timeout = timeout_s if timeout_s and timeout_s > 0 else None
    process.join(timeout)
    elapsed = time.perf_counter() - start
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.terminate()
        process.join(10)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
            process.join()
        return result_class(backend, False, total_s=elapsed, message=f"{backend} exceeded {timeout_s}s timeout")
    if process.exitcode != 0:
        return result_class(backend, False, total_s=elapsed, message=f"{backend} failed in isolated process with exit code {process.exitcode}")
    if queue.empty():
        return result_class(backend, False, total_s=elapsed, message=f"{backend} isolated process returned no result")
    status, payload = queue.get()
    if status == "ok":
        return payload
    return result_class(backend, False, total_s=elapsed, message=payload)


def repeated_isolated_run(backend: str, fn, result_class, repeats: int, timeout_s: int | None):
    runs = [isolated_timed_safe_run(backend, fn, result_class, timeout_s) for _ in range(repeats)]
    completed = [result for result in runs if result.available and result.total_s is not None]
    if not completed:
        result = runs[0]
    else:
        result = sorted(completed, key=lambda item: float(item.total_s))[len(completed) // 2]
    result.runtime_repeats_s = [item.total_s for item in runs]
    return result


def compare_available_nested(results: list, n_obs: int) -> bool | None:
    """Compare final log likelihoods after excluding clearly inferior solutions."""
    return loglike_consistency(results, n_obs)


def run_generated_mnl(meta: GeneratedSpec, args) -> dict:
    case = build_mnl_case(meta, args.seed)
    runners = [
        ("torchdcm", lambda: run_mnl_torch(case, args.max_iter)),
        ("scipy_bfgs", lambda: run_mnl_scipy(case)),
        ("biogeme", lambda: run_mnl_biogeme(case)),
        ("apollo", lambda: run_mnl_apollo(case)),
        ("mlogit", lambda: run_mnl_mlogit(case)),
        ("gmnl", lambda: run_mnl_gmnl(case)),
        ("xlogit", lambda: run_mnl_xlogit(case)),
    ]
    results = [
        repeated_isolated_run(backend, runner, BackendResult, args.repeats, args.backend_timeout)
        for backend, runner in runners
    ]
    consistent = compare_mnl(case, results)
    return payload(
        meta,
        "MNL",
        case.parameter_names,
        results,
        consistent,
        extra={
            "true_parameters": case.true_parameters,
            "runtime_repetitions": args.repeats,
            "runtime_summary": "median" if args.repeats > 1 else "single run",
            "worker_policy": "independent child process per backend repetition",
        },
    )


def run_generated_nl(meta: GeneratedSpec, args) -> dict:
    mnl_case = build_mnl_case(meta, args.seed)
    case = build_nested_case(mnl_case, args.seed)
    results = [
        nested_real.safe_run("torchdcm", lambda: nested_real.run_torch(case, max_iter=args.max_iter)),
        isolated_timed_safe_run("biogeme", lambda: nested_real.run_biogeme(case, lambda_min=args.lambda_min), nested_real.BackendResult, args.backend_timeout)
        if args.profile.startswith(("stress", "table4"))
        else timed_safe_run("biogeme", lambda: nested_real.run_biogeme(case, lambda_min=args.lambda_min), nested_real.BackendResult, args.backend_timeout),
        isolated_timed_safe_run("apollo", lambda: nested_real.run_apollo(case, lambda_min=args.lambda_min), nested_real.BackendResult, args.backend_timeout)
        if args.profile.startswith(("stress", "table4"))
        else timed_safe_run("apollo", lambda: nested_real.run_apollo(case, lambda_min=args.lambda_min), nested_real.BackendResult, args.backend_timeout),
    ]
    result = nested_real.result_payload(case, results)
    comparison = compare_available_nested(results, case.data.n_obs)
    result["consistent"] = "N.A." if comparison is None else ("Yes" if comparison else "No")
    result.update(
        {
            "generated": meta_payload(meta),
            "family": "Nested logit",
            "data_generating_process": "two-level nested logit",
            "true_dissimilarities": {
                name: 1.0 if nest.fixed else NL_TRUE_DISSIMILARITY[name]
                for name, nest in case.nests.items()
            },
        }
    )
    return result


def run_generated_mixl(meta: GeneratedSpec, args) -> dict:
    mnl_case = build_mnl_case(meta, args.seed)
    case = build_mixed_case(mnl_case, args.seed)
    start = mixed_real.MixedStart(
        means={name: 0.0 for name in case.parameter_names},
        sigmas={name: MIXL_SIGMA_INIT for name in case.random_names},
        mnl_loglike=float("nan"),
        mnl_gradient_norm=float("nan"),
        sigma_scale=MIXL_SIGMA_INIT,
    )
    draws = mixed_real.make_draws(args.n_draws, args.seed + stable_case_offset(meta.case), len(case.random_names))
    mixed_real.make_draws_cached = draws
    results = [
        safe_run_mixed("torchdcm", lambda: mixed_real.run_torch(case, draws, args.max_iter, args.torch_device, start)),
        isolated_timed_safe_run("biogeme", lambda: mixed_real.run_biogeme(case, draws.detach().cpu().numpy(), args.max_iter, start, smooth_positive_scales=True), mixed_real.BackendResult, args.backend_timeout)
        if args.profile.startswith(("stress", "table4"))
        else timed_safe_run("biogeme", lambda: mixed_real.run_biogeme(case, draws.detach().cpu().numpy(), args.max_iter, start, smooth_positive_scales=True), mixed_real.BackendResult, args.backend_timeout),
        isolated_timed_safe_run("apollo", lambda: run_mixed_apollo(case, args.n_draws), mixed_real.BackendResult, args.backend_timeout)
        if args.profile.startswith(("stress", "table4"))
        else timed_safe_run("apollo", lambda: run_mixed_apollo(case, args.n_draws), mixed_real.BackendResult, args.backend_timeout),
    ]
    consistent = mixed_real.compare_results(case, results)
    result = mixed_real.payload(case, draws, results, consistent, start)
    result["starting_values"] = {
        "method": "zero mean coefficients with common positive random-coefficient scales",
        "mean_coefficients": start.means,
        "random_scales": start.sigmas,
        "sigma_scale": start.sigma_scale,
    }
    result.update(
        {
            "generated": meta_payload(meta),
            "family": "Mixed logit",
            "data_generating_process": "mixed logit with independent normal coefficients",
            "true_sigmas": {
                f"SIGMA_{name}": value
                for name, value in mixed_true_sigmas(case.random_names).items()
            },
        }
    )
    return result


def payload(meta: GeneratedSpec, family: str, parameters: list[str], results: list[BackendResult], consistent: bool | None, extra: dict | None = None) -> dict:
    row = {
        "case": meta.case,
        "family": family,
        "generated": meta_payload(meta),
        "n_obs": meta.n_obs,
        "n_alternatives": meta.n_alternatives,
        "n_variables": meta.n_variables,
        "rho": meta.rho,
        "n_parameters": len(parameters),
        "parameters": parameters,
        "consistent": "N.A." if consistent is None else ("Yes" if consistent else "No"),
        "runtime_policy": runtime_policy_metadata(),
        "backends": [backend_payload(result) for result in results],
    }
    if extra:
        row.update(extra)
    return row


def backend_payload(result) -> dict:
    return {
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
        "closure_evals": getattr(result, "closure_evals", None),
        "optimizer_iterations": getattr(result, "optimizer_iterations", None),
        "runtime_repeats_s": getattr(result, "runtime_repeats_s", None),
        "worse_loglike": getattr(result, "worse_loglike", False),
        "message": result.message,
    }


def meta_payload(meta: GeneratedSpec) -> dict:
    return {
        "model": meta.model,
        "N": meta.n_obs,
        "J": meta.n_alternatives,
        "K": meta.n_variables,
        "rho": meta.rho,
        "random_coefficients": meta.random_coefficients,
    }


def covariance_to_array(covariance_obj, names: list[str]) -> np.ndarray:
    if hasattr(covariance_obj, "loc"):
        return covariance_obj.loc[names, names].to_numpy(dtype=float)
    return np.asarray(covariance_obj, dtype=float)


def reorder_covariance_or_none(covariance, source_names, target_names: list[str]) -> np.ndarray | None:
    if covariance is None or source_names is None:
        return None
    try:
        matrix = pd.DataFrame(covariance, index=source_names, columns=source_names)
        array = matrix.loc[target_names, target_names].to_numpy(dtype=float)
    except Exception:
        return None
    if not np.isfinite(array).all():
        return None
    return array


def r_env() -> dict[str, str]:
    env = os.environ.copy()
    r_user_lib = str(Path.home() / "R" / "site-library")
    existing = env.get("R_LIBS_USER")
    env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
    return env


def render_markdown(rows: list[dict], profile: str) -> str:
    description = (
        "All cross-estimator runtimes report estimation plus covariance on one logical CPU. Stress rows apply a 300-second worker-wall-clock limit to every external backend; timeout is not treated as numerical disagreement."
        if profile.startswith("stress")
        else "All cross-estimator runtimes report estimation plus covariance on one logical CPU. Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit and mixed-logit rows compare TorchDCM against Biogeme and Apollo."
    )
    repeated = [row for row in rows if int(row.get("runtime_repetitions", 1)) > 1]
    if repeated:
        counts = sorted({int(row["runtime_repetitions"]) for row in repeated})
        count_text = ", ".join(str(count) for count in counts)
        description += (
            f" Repeated rows report the median of {count_text} independent backend workers."
            " TorchDCM performs one untimed likelihood-and-gradient warm-up per worker and records LBFGS closure evaluations."
        )
    description += (
        " A dagger marks a completed solver whose final log likelihood is clearly below the row best;"
        " its runtime is retained but its estimate is excluded from consistency."
    )
    lines = [
        f"# Generated Choice Benchmark Battery ({profile})",
        "",
        description,
        "",
        "| case | family | N | J | K | rho | TorchDCM | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        backends = {item["backend"]: item for item in row.get("backends", [])}
        meta = row.get("generated", {})
        lines.append(
            "| {case} | {family} | {N} | {J} | {K} | {rho} | {torchdcm} | {scipy} | {biogeme} | {apollo} | {mlogit} | {gmnl} | {xlogit} | {consistent} |".format(
                case=row["case"],
                family=row.get("family", row.get("model", "")),
                N=meta.get("N", row.get("n_obs")),
                J=meta.get("J", row.get("n_alternatives")),
                K=meta.get("K", row.get("n_variables")),
                rho=meta.get("rho", row.get("rho")),
                torchdcm=fmt_time(backends.get("torchdcm")),
                scipy=fmt_time(backends.get("scipy_bfgs")),
                biogeme=fmt_time(backends.get("biogeme")),
                apollo=fmt_time(backends.get("apollo")),
                mlogit=fmt_time(backends.get("mlogit")),
                gmnl=fmt_time(backends.get("gmnl")),
                xlogit=fmt_time(backends.get("xlogit")),
                consistent=consistency_label(row.get("consistent", "No")),
            )
        )
    lines.extend(["", "## Objective Diagnostics", ""])
    for row in rows:
        backends = {item["backend"]: item for item in row.get("backends", [])}
        ref = backends.get("torchdcm", {})
        lines.append(f"- `{row['case']}`: reference loglike={sci(ref.get('loglike'))}; " + ", ".join(
            f"{name} ll_diff={sci(backends[name].get('ll_diff'))}"
            for name in ["scipy_bfgs", "biogeme", "apollo", "mlogit", "gmnl", "xlogit"]
            if name in backends
        ))
    return "\n".join(lines) + "\n"


def fmt_time(row: dict | None) -> str:
    if not row:
        return "NA"
    if not row.get("available"):
        if "timeout" in str(row.get("message", "")).lower():
            return "Timeout"
        return "Fail"
    value = row.get("total_s")
    marker = "†" if row.get("worse_loglike") else ""
    return f"{float(value):.3f}{marker}" if isinstance(value, (int, float)) else "NA"


def consistency_label(value) -> str:
    if value is True or value == "Yes":
        return "Yes"
    if value is False or value == "No":
        return "No"
    if value is None or value in {"--", "N.A."}:
        return "N.A."
    return str(value)


def sci(value) -> str:
    return f"{float(value):.2e}" if isinstance(value, (int, float)) and np.isfinite(value) else "NA"


def write_outputs(rows: list[dict], profile: str) -> tuple[Path, Path]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    json_path = GENERATED / f"generated_choice_battery_{profile}.json"
    md_path = GENERATED / f"generated_choice_battery_{profile}.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(rows, profile), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["paper", "smoke", "full", "controlled", "controlled_additional", "stress", "stress_additional", "table4"],
        default="smoke",
    )
    parser.add_argument("--models", nargs="+", choices=["mnl", "nl", "mixl"], default=["mnl", "nl", "mixl"])
    parser.add_argument("--case", help="Run only the named case within the selected profile.")
    parser.add_argument("--output-profile", help="Optional output filename suffix; defaults to --profile.")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--lambda-min", type=float, default=0.0001)
    parser.add_argument("--n-draws", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1, help="Independent backend repetitions; the median runtime run is reported.")
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--backend-timeout", type=int, default=0, help="Seconds before marking non-Torch backends as timed out. Zero disables per-backend timeout.")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if torch.device(args.torch_device).type != "cpu":
        raise ValueError("Generated cross-estimator comparisons must use --torch-device cpu.")

    output_profile = args.output_profile or args.profile
    rows = []
    for meta in generated_specs(args.profile):
        if args.case and meta.case != args.case:
            continue
        if meta.model not in args.models:
            continue
        print(f"[generated] running {meta.case} ({meta.model}, N={meta.n_obs}, J={meta.n_alternatives}, K={meta.n_variables}, rho={meta.rho})", flush=True)
        if meta.model == "mnl":
            row = run_generated_mnl(meta, args)
        elif meta.model == "nl":
            row = run_generated_nl(meta, args)
        else:
            row = run_generated_mixl(meta, args)
        rows.append(row)
        write_outputs(rows, output_profile)
        backends = {item["backend"]: item for item in row.get("backends", [])}
        print(
            f"[generated] {meta.case}: torch={fmt_time(backends.get('torchdcm'))} "
            f"biogeme={fmt_time(backends.get('biogeme'))} apollo={fmt_time(backends.get('apollo'))} "
            f"consistent={row.get('consistent')}",
            flush=True,
        )
    json_path, md_path = write_outputs(rows, output_profile)
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")


if __name__ == "__main__":
    main()
