from __future__ import annotations

import argparse
import csv
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
from scipy.optimize import minimize

from benchmark_runtime import estimation_covariance_total
from torchdcm import Beta, ChoiceDataset, MultinomialLogit, UtilitySpec
from torchdcm.spec.expressions import Expression, Term
from compare_biogeme import build_case, run_biogeme
from mnl_generic_backends import make_design_long, run_gmnl_generic, run_mlogit_generic, run_xlogit_generic


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
    wtp: float | None = None
    wtp_se: float | None = None
    elasticities: dict[str, np.ndarray] | None = None
    message: str = ""


def spec_with_initials(spec: UtilitySpec, initial_values: dict[str, float]) -> UtilitySpec:
    registry: dict[str, Beta] = {}

    def convert_beta(beta: Beta) -> Beta:
        if beta.name not in registry:
            registry[beta.name] = Beta(
                beta.name,
                init=initial_values.get(beta.name, beta.init),
                fixed=beta.fixed,
            )
        return registry[beta.name]

    new_spec = UtilitySpec()
    for alt, expr in spec.utilities.items():
        new_spec.utility(
            alt,
            Expression(
                [
                    Term(
                        parameter=convert_beta(term.parameter),
                        variable=term.variable,
                        multiplier=term.multiplier,
                    )
                    for term in expr.terms
                ]
            ),
        )
    return new_spec


def make_initial_values(names: list[str], mode: str, seed: int, scale: float) -> dict[str, float]:
    if mode == "zero":
        return {name: 0.0 for name in names}
    if mode == "random":
        rng = np.random.default_rng(seed)
        return {name: float(value) for name, value in zip(names, rng.normal(0.0, scale, len(names)))}
    raise ValueError(f"Unknown initialization mode: {mode}")


def load_biogeme_swissmetro(n_obs: int | None = None):
    try:
        import biogeme.data.swissmetro as swissmetro
    except ImportError as exc:
        raise RuntimeError("Biogeme is required to load the official Swissmetro testing data.") from exc

    raw = pd.read_csv(Path(swissmetro.__file__).resolve().parent / "data" / "swissmetro.dat", sep="\t")
    df = raw.loc[raw["CHOICE"] != 0].copy()
    if n_obs is not None:
        df = df.head(n_obs).copy()
    df["obs_id"] = np.arange(len(df))
    df["choice"] = df["CHOICE"].map({1: "TRAIN", 2: "SM", 3: "CAR"})
    df["time_train"] = df["TRAIN_TT"] / 100.0
    df["time_sm"] = df["SM_TT"] / 100.0
    df["time_car"] = df["CAR_TT"] / 100.0
    df["cost_train"] = df["TRAIN_CO"] * (df["GA"] == 0) / 100.0
    df["cost_sm"] = df["SM_CO"] * (df["GA"] == 0) / 100.0
    df["cost_car"] = df["CAR_CO"] / 100.0
    df["avail_train"] = (df["TRAIN_AV"] * (df["SP"] != 0)).astype(bool)
    df["avail_sm"] = df["SM_AV"].astype(bool)
    df["avail_car"] = (df["CAR_AV"] * (df["SP"] != 0)).astype(bool)
    alternatives = ["TRAIN", "SM", "CAR"]
    data = ChoiceDataset.from_wide(
        df,
        alternatives=alternatives,
        choice="choice",
        variables={
            "time": {"TRAIN": "time_train", "SM": "time_sm", "CAR": "time_car"},
            "cost": {"TRAIN": "cost_train", "SM": "cost_sm", "CAR": "cost_car"},
        },
        availability={"TRAIN": "avail_train", "SM": "avail_sm", "CAR": "avail_car"},
        obs_id="obs_id",
        individual_id="ID",
    )
    spec = UtilitySpec()
    spec.utility("TRAIN", Beta("ASC_TRAIN") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("SM", Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("CAR", Beta("ASC_CAR") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    return df, data, spec, alternatives


def run_torchdcm(data, spec) -> BackendResult:
    model = MultinomialLogit(spec)
    data = data.to(device=model.device, dtype=model.dtype)
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
    estimate_seconds = time.perf_counter() - estimate_start
    final_params = params.detach().clone()
    loglike = float(model.loglike(final_params, data, compiled).detach().cpu())
    param_values = dict(zip(compiled.free_names, final_params.detach().cpu().numpy()))
    covariance_start = time.perf_counter()
    covariance = exact_classic_covariance(model, data, compiled, compiled.free_names, param_values)
    covariance_seconds = time.perf_counter() - covariance_start
    return BackendResult(
        backend="torchdcm",
        available=True,
        seconds=estimate_seconds + covariance_seconds,
        estimate_seconds=estimate_seconds,
        covariance_seconds=covariance_seconds,
        loglike=loglike,
        params=param_values,
        covariance=covariance,
    )


def run_scipy_mle(data, spec, initial_values: dict[str, float]) -> BackendResult:
    model = MultinomialLogit(spec)
    compiled = model.compile(data)
    x0 = np.asarray([initial_values[name] for name in compiled.free_names], dtype=float)
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

    start = time.perf_counter()
    opt = minimize(
        fun=lambda beta: value_and_grad(beta)[0],
        x0=x0,
        jac=lambda beta: value_and_grad(beta)[1],
        method="BFGS",
        options={"gtol": 1e-7, "maxiter": 200},
    )
    seconds = time.perf_counter() - start
    if not opt.success:
        message = f"{opt.message}"
    else:
        message = ""
    loglike = -float(opt.fun)
    covariance_start = time.perf_counter()
    covariance = exact_classic_covariance(model, data, compiled, compiled.free_names, dict(zip(compiled.free_names, opt.x)))
    covariance_seconds = time.perf_counter() - covariance_start
    return BackendResult(
        backend="scipy_bfgs",
        available=True,
        seconds=seconds + covariance_seconds,
        estimate_seconds=seconds,
        covariance_seconds=covariance_seconds,
        loglike=loglike,
        params=dict(zip(compiled.free_names, opt.x)),
        covariance=covariance,
        message=message,
    )


def run_biogeme_timed(df, alternatives, names, initial_values) -> BackendResult:
    start = time.perf_counter()
    result = run_biogeme(df, alternatives, names, initial_values=initial_values)
    seconds = time.perf_counter() - start
    return BackendResult(
        backend="biogeme",
        available=True,
        seconds=seconds,
        estimate_seconds=result.get("estimate_seconds"),
        covariance_seconds=result.get("covariance_seconds"),
        loglike=result["loglike"],
        params=result["params"],
        covariance=np.asarray(result["covariance"], dtype=float),
    )


def write_apollo_inputs(df, alternatives: list[str], names: list[str], initial_values: dict[str, float], directory: Path):
    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy()
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    wide_df = wide_df.drop(columns=["choice"])
    for column in wide_df.select_dtypes(include=["bool"]).columns:
        wide_df[column] = wide_df[column].astype(int)

    csv_path = directory / "data.csv"
    spec_path = directory / "spec.json"
    wide_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    spec = {
        "model_name": f"apollo_mnl_{'_'.join(alternatives)}_{len(df)}",
        "alternatives": alternatives,
        "choice_col": "choice_code",
        "parameters": {name: initial_values[name] for name in names},
        "utility": {
            alt: {
                "code": code_by_alt[alt],
                "asc": f"ASC_{alt.upper()}" if f"ASC_{alt.upper()}" in names else None,
                "time": f"time_{alt.lower()}",
                "cost": f"cost_{alt.lower()}",
                "availability": f"avail_{alt.lower()}",
            }
            for alt in alternatives
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return csv_path, spec_path


def run_apollo_timed(df, alternatives, names, initial_values) -> BackendResult:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return BackendResult(backend="apollo", available=False, message="Rscript not found")

    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_mnl.R"
    if not script.exists():
        return BackendResult(backend="apollo", available=False, message=f"Missing Apollo script: {script}")

    with tempfile.TemporaryDirectory(prefix="torchdcm_apollo_") as tmp:
        tmp_path = Path(tmp)
        data_path, spec_path = write_apollo_inputs(df, alternatives, names, initial_values, tmp_path)
        output_path = tmp_path / "apollo_result.json"
        cmd = [
            rscript,
            str(script),
            "--data",
            str(data_path),
            "--spec",
            str(spec_path),
            "--output",
            str(output_path),
        ]
        env = os.environ.copy()
        r_user_lib = str(Path.home() / "R" / "site-library")
        existing_r_lib = env.get("R_LIBS_USER")
        env["R_LIBS_USER"] = r_user_lib if not existing_r_lib else f"{r_user_lib}:{existing_r_lib}"
        start = time.perf_counter()
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        seconds = time.perf_counter() - start
        if proc.returncode != 0:
            return BackendResult(
                backend="apollo",
                available=False,
                seconds=seconds,
                message=(proc.stderr or proc.stdout).strip(),
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        estimate_seconds = payload.get("timing", {}).get("estimate_seconds")
        covariance_seconds = payload.get("timing", {}).get("covariance_seconds")
        return BackendResult(
            backend="apollo",
            available=True,
            seconds=estimation_covariance_total(estimate_seconds, covariance_seconds),
            estimate_seconds=estimate_seconds,
            covariance_seconds=covariance_seconds,
            loglike=float(payload["loglike"]),
            params={key: float(value) for key, value in payload["estimates"].items()},
            covariance=np.asarray(payload["covariance"], dtype=float)
            if "covariance" in payload and payload["covariance"] is not None
            else None,
            message=f"apollo_version={payload.get('apollo_version')}",
        )


def make_swissmetro_design_long(df, alternatives: list[str], names: list[str]) -> pd.DataFrame:
    utility_columns = {
        "TRAIN": {"ASC_TRAIN": 1.0, "B_TIME": "time_train", "B_COST": "cost_train"},
        "SM": {"B_TIME": "time_sm", "B_COST": "cost_sm"},
        "CAR": {"ASC_CAR": 1.0, "B_TIME": "time_car", "B_COST": "cost_car"},
    }
    availability = {alt: f"avail_{alt.lower()}" for alt in alternatives}
    return make_design_long(df, alternatives, "choice", utility_columns, availability, names)


def exact_classic_covariance(model: MultinomialLogit, data, compiled, names: list[str], params: dict[str, float]) -> np.ndarray:
    vector = torch.as_tensor([params[name] for name in names], dtype=torch.float64)
    hessian_ll = torch.autograd.functional.hessian(lambda p: model.loglike(p, data, compiled), vector)
    information = -hessian_ll.detach()
    return torch.linalg.pinv(information, hermitian=True).cpu().numpy()


def params_to_vector(names: list[str], params: dict[str, float]) -> torch.Tensor:
    return torch.as_tensor([params[name] for name in names], dtype=torch.float64)


def predict_probabilities(data, spec, names: list[str], params: dict[str, float]) -> np.ndarray:
    model = MultinomialLogit(spec)
    vector = params_to_vector(names, params)
    return model.predict_proba(data, vector).detach().cpu().numpy()


def wtp_with_delta(names: list[str], params: dict[str, float], covariance: np.ndarray | None) -> tuple[float | None, float | None]:
    if "B_TIME" not in params or "B_COST" not in params:
        return None, None
    wtp = -params["B_TIME"] / params["B_COST"]
    if covariance is None:
        return float(wtp), None
    idx_time = names.index("B_TIME")
    idx_cost = names.index("B_COST")
    grad = np.zeros(len(names), dtype=float)
    beta_time = params["B_TIME"]
    beta_cost = params["B_COST"]
    grad[idx_time] = -1.0 / beta_cost
    grad[idx_cost] = beta_time / (beta_cost**2)
    variance = float(grad @ covariance @ grad)
    return float(wtp), float(np.sqrt(max(variance, 0.0)))


def direct_elasticities(data, probabilities: np.ndarray, params: dict[str, float]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    if "B_TIME" in params and "time" in data.x_alt:
        x = data.x_alt["time"].detach().cpu().numpy()
        values["time"] = params["B_TIME"] * x * (1.0 - probabilities)
    if "B_COST" in params and "cost" in data.x_alt:
        x = data.x_alt["cost"].detach().cpu().numpy()
        values["cost"] = params["B_COST"] * x * (1.0 - probabilities)
    return values


def attach_post_estimation(results: list[BackendResult], data, spec, names: list[str]) -> None:
    for result in results:
        if not result.available:
            continue
        result.probabilities = predict_probabilities(data, spec, names, result.params)
        result.wtp, result.wtp_se = wtp_with_delta(names, result.params, result.covariance)
        result.elasticities = direct_elasticities(data, result.probabilities, result.params)


def compare_to_reference(results: list[BackendResult], reference: str):
    ref = next(result for result in results if result.backend == reference and result.available)
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        shared = sorted(set(result.params) & set(ref.params))
        result.max_abs_param_diff = max(abs(result.params[name] - ref.params[name]) for name in shared)  # type: ignore[attr-defined]
        result.max_abs_probability_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))  # type: ignore[attr-defined]
        if result.covariance is not None and ref.covariance is not None:
            result.max_abs_covariance_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
            result.max_abs_se_diff = float(
                np.max(np.abs(np.sqrt(np.diag(result.covariance)) - np.sqrt(np.diag(ref.covariance))))
            )  # type: ignore[attr-defined]
        else:
            result.max_abs_covariance_diff = None  # type: ignore[attr-defined]
            result.max_abs_se_diff = None  # type: ignore[attr-defined]
        if result.wtp is not None and ref.wtp is not None:
            result.wtp_diff = result.wtp - ref.wtp  # type: ignore[attr-defined]
        else:
            result.wtp_diff = None  # type: ignore[attr-defined]
        if result.wtp_se is not None and ref.wtp_se is not None:
            result.wtp_se_diff = result.wtp_se - ref.wtp_se  # type: ignore[attr-defined]
        else:
            result.wtp_se_diff = None  # type: ignore[attr-defined]
        elasticity_diffs = {}
        for key in sorted(set(result.elasticities or {}) & set(ref.elasticities or {})):
            elasticity_diffs[key] = float(np.max(np.abs(result.elasticities[key] - ref.elasticities[key])))
        result.max_abs_elasticity_diff = elasticity_diffs  # type: ignore[attr-defined]


def run_case(case: str, n_obs: int, data_seed: int, initial: str, init_seed: int, init_scale: float):
    if case == "swissmetro":
        df, data, base_spec, alternatives = load_biogeme_swissmetro(n_obs)
    else:
        raise ValueError(
            "Aligned estimator benchmarks use real external data only. "
            "Add a real-data loader before enabling this case."
        )
    names = base_spec.parameter_names
    initial_values = make_initial_values(names, initial, init_seed, init_scale)
    spec = spec_with_initials(base_spec, initial_values)

    results = [
        run_torchdcm(data, spec),
        run_scipy_mle(data, spec, initial_values),
        run_biogeme_timed(df, alternatives, names, initial_values),
        run_apollo_timed(df, alternatives, names, initial_values),
    ]
    design_long = make_swissmetro_design_long(df, alternatives, names)
    results.extend(
        [
            run_mlogit_generic(design_long, names),
            run_gmnl_generic(design_long, names),
            run_xlogit_generic(design_long, names),
        ]
    )
    attach_post_estimation(results, data, spec, names)
    compare_to_reference(results, reference="torchdcm")
    return initial_values, results, len(df)


def print_results(case: str, n_obs: int, initial: str, initial_values: dict[str, float], results: list[BackendResult]) -> None:
    print(f"case: {case}")
    print(f"n_obs: {n_obs}")
    print("alignment:")
    print("  benchmark_mode: full_estimation")
    print("  data_source: biogeme.data.swissmetro/data/swissmetro.dat")
    print("  model: MNL with TRAIN/SM/CAR, ASC_TRAIN, ASC_CAR, B_TIME, B_COST")
    print("  scaling: time/100, cost/100, GA discount and official availability filters")
    print("  initial_values: shared across TorchDCM, SciPy, Biogeme, Apollo")
    print("  covariance: classic inverse observed information where available")
    print("  reference: torchdcm")
    print(f"initial: {initial}")
    print("initial_values:")
    for name, value in initial_values.items():
        print(f"  {name}: {value:.12g}")
    print()
    print(
        f"{'backend':<12}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'max_param_diff':>18}"
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
                f"{getattr(result, 'max_abs_param_diff'):>18.3e}"
            )
        else:
            print(
                f"{result.backend:<12}{str(result.available):>10}"
                f"{'':>12}{'':>12}{'':>12}{'':>18}{'':>14}{'':>18}  {result.message}"
            )
    print()
    print(
        f"{'backend':<12}{'prob_diff':>14}{'cov_diff':>14}{'se_diff':>14}"
        f"{'wtp':>14}{'wtp_diff':>14}{'wtp_se':>14}{'wtp_se_diff':>14}"
        f"{'elas_time':>14}{'elas_cost':>14}"
    )
    for result in results:
        if not result.available:
            continue
        elas = getattr(result, "max_abs_elasticity_diff")
        print(
            f"{result.backend:<12}"
            f"{getattr(result, 'max_abs_probability_diff'):>14.3e}"
            f"{_fmt_optional(getattr(result, 'max_abs_covariance_diff')):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_se_diff')):>14}"
            f"{_fmt_optional(result.wtp):>14}"
            f"{_fmt_optional(getattr(result, 'wtp_diff')):>14}"
            f"{_fmt_optional(result.wtp_se):>14}"
            f"{_fmt_optional(getattr(result, 'wtp_se_diff')):>14}"
            f"{_fmt_optional(elas.get('time')):>14}"
            f"{_fmt_optional(elas.get('cost')):>14}"
        )


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3e}"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["swissmetro"], default="swissmetro")
    parser.add_argument("--n-obs", type=int, default=500)
    parser.add_argument("--data-seed", type=int, default=3)
    parser.add_argument("--initial", choices=["zero", "random"], default="zero")
    parser.add_argument("--init-seed", type=int, default=20260704)
    parser.add_argument("--init-scale", type=float, default=0.1)
    args = parser.parse_args()

    initial_values, results, actual_n_obs = run_case(
        case=args.case,
        n_obs=args.n_obs,
        data_seed=args.data_seed,
        initial=args.initial,
        init_seed=args.init_seed,
        init_scale=args.init_scale,
    )
    print_results(args.case, actual_n_obs, args.initial, initial_values, results)


if __name__ == "__main__":
    main()
