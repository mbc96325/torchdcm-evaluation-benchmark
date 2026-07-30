from __future__ import annotations

import argparse
import json
import os
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

from torchdcm import Beta, OrderedChoiceDataset, OrderedLogit, OrderedProbit


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


CATEGORIES = [1, 2, 3, 4, 5, 6]
VARIABLES = ["male", "highEducation", "haveGA", "ScaledIncome"]
WEIGHT_COLUMN = "normalized_weight"


def load_ordered_optima(indicator: str, n_obs: int | None):
    try:
        from biogeme.data.optima import read_data
    except ImportError as exc:
        raise RuntimeError("Biogeme is required for the aligned Optima ordered benchmark.") from exc

    database = read_data()
    df = database.dataframe.copy().reset_index(drop=True)
    required = [indicator, *VARIABLES, WEIGHT_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Biogeme Optima dataframe is missing required columns: {missing}")
    df = df[df[indicator].isin(CATEGORIES)].copy().reset_index(drop=True)
    if n_obs is not None:
        df = df.head(n_obs).copy().reset_index(drop=True)
    data = OrderedChoiceDataset.from_dataframe(
        df,
        outcome=indicator,
        variables=VARIABLES,
        categories=CATEGORIES,
        weight=WEIGHT_COLUMN,
    )
    return df.reset_index(drop=True), data


def make_model(kind: str):
    latent = (
        Beta("B_MALE") * "male"
        + Beta("B_HIGH_EDUCATION") * "highEducation"
        + Beta("B_GA") * "haveGA"
        + Beta("B_INCOME") * "ScaledIncome"
    )
    thresholds = [
        Beta("TH_1", init=-1.5),
        Beta("TH_2", init=-0.5),
        Beta("TH_3", init=0.5),
        Beta("TH_4", init=1.5),
        Beta("TH_5", init=2.5),
    ]
    if kind == "logit":
        return OrderedLogit(latent, thresholds)
    if kind == "probit":
        return OrderedProbit(latent, thresholds)
    raise ValueError(f"Unknown ordered model: {kind}")


def default_params() -> dict[str, float]:
    return {
        "B_MALE": 0.2,
        "B_HIGH_EDUCATION": 0.15,
        "B_GA": -0.3,
        "B_INCOME": 0.05,
        "TH_1": -1.5,
        "TH_2": -0.5,
        "TH_3": 0.5,
        "TH_4": 1.5,
        "TH_5": 2.5,
    }


def ordered_probabilities(
    df: pd.DataFrame,
    kind: str,
    params: dict[str, float],
) -> np.ndarray:
    utility = sum(params[name] * df[variable].to_numpy(dtype=float) for name, variable in zip(
        ["B_MALE", "B_HIGH_EDUCATION", "B_GA", "B_INCOME"],
        VARIABLES,
    ))
    thresholds = np.asarray(
        [params[f"TH_{index}"] for index in range(1, len(CATEGORIES))],
        dtype=float,
    )
    bounds = np.concatenate(([-np.inf], thresholds, [np.inf]))
    differences = torch.as_tensor(
        bounds[None, :] - utility[:, None], dtype=torch.float64
    )
    if kind == "logit":
        cumulative = torch.sigmoid(differences)
    else:
        cumulative = torch.special.ndtr(differences)
    return (cumulative[:, 1:] - cumulative[:, :-1]).numpy()


def run_torch_fixed(data, kind: str, params: dict[str, float]) -> BackendResult:
    model = make_model(kind)
    compiled = model.compile(data)
    vector = torch.as_tensor([params[name] for name in compiled.free_names], dtype=torch.float64)
    start = time.perf_counter()
    ll = model.loglike(vector, data, compiled)
    probabilities = model.predict_proba(data, vector)
    seconds = time.perf_counter() - start
    return BackendResult(
        backend="torchdcm_fixed",
        available=True,
        seconds=seconds,
        loglike=float(ll.detach().cpu()),
        params={name: params[name] for name in compiled.free_names},
        probabilities=probabilities.detach().cpu().numpy(),
    )


def run_torch_fit(data, kind: str, max_iter: int) -> BackendResult:
    model = make_model(kind)
    data = data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    internal_initial = torch.cat(
        [
            compiled.free_initial[: len(compiled.beta_names)],
            model._threshold_to_internal(compiled.threshold_initial),
        ]
    )
    internal_params = internal_initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [internal_params],
        max_iter=max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        natural = model._internal_to_natural(internal_params, compiled)
        loss = -model.loglike(natural, data, compiled)
        loss.backward()
        return loss

    estimate_start = time.perf_counter()
    optimizer.step(closure)
    estimate_seconds = time.perf_counter() - estimate_start
    final_internal = internal_params.detach().clone().requires_grad_(True)
    final_natural = model._internal_to_natural(final_internal, compiled)
    ll = model.loglike(final_natural, data, compiled)
    covariance_start = time.perf_counter()
    hessian_internal = torch.autograd.functional.hessian(
        lambda p: model.loglike(model._internal_to_natural(p, compiled), data, compiled),
        final_internal,
    )
    cov_internal = torch.linalg.pinv(-hessian_internal.detach(), hermitian=True)
    transform_jac = model._natural_jacobian(final_internal.detach(), compiled)
    covariance = transform_jac @ cov_internal @ transform_jac.T
    covariance_seconds = time.perf_counter() - covariance_start
    return BackendResult(
        backend="torchdcm_fit",
        available=True,
        seconds=estimate_seconds + covariance_seconds,
        estimate_seconds=estimate_seconds,
        covariance_seconds=covariance_seconds,
        loglike=float(ll.detach().cpu()),
        params=dict(zip(compiled.free_names, final_natural.detach().cpu().numpy())),
        covariance=covariance.detach().cpu().numpy(),
        probabilities=model.predict_proba(data, final_natural.detach()).detach().cpu().numpy(),
    )


def run_biogeme_fixed(df: pd.DataFrame, kind: str, params: dict[str, float]) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        import biogeme.models as models
        from biogeme.expressions import Variable
    except ImportError as exc:
        return BackendResult(backend="biogeme_fixed", available=False, message=f"Biogeme not found: {exc}")

    try:
        database = db.Database("torchdcm_ordered_fixed", df)
        continuous = (
            params["B_MALE"] * Variable("male")
            + params["B_HIGH_EDUCATION"] * Variable("highEducation")
            + params["B_GA"] * Variable("haveGA")
            + params["B_INCOME"] * Variable("ScaledIncome")
        )
        thresholds = [params[f"TH_{i}"] for i in range(1, len(CATEGORIES))]
        if kind == "logit":
            formulas_by_category = models.ordered_logit_from_thresholds(continuous, 1.0, CATEGORIES, thresholds)
        else:
            formulas_by_category = models.ordered_probit_from_thresholds(continuous, 1.0, CATEGORIES, thresholds)
        formulas = {f"prob_{category}": formulas_by_category[category] for category in CATEGORIES}
        start = time.perf_counter()
        biogeme = bio.BIOGEME(database, formulas)
        biogeme.model_name = f"torchdcm_ordered_{kind}_fixed"
        simulated = biogeme.simulate({})
        seconds = time.perf_counter() - start
    except Exception as exc:
        return BackendResult(backend="biogeme_fixed", available=False, message=f"{type(exc).__name__}: {exc}")

    probabilities = np.column_stack([simulated[f"prob_{category}"].to_numpy(dtype=float) for category in CATEGORIES])
    indicator = df.attrs.get("indicator", "Envir01")
    y_index = np.asarray([CATEGORIES.index(int(value)) for value in df[indicator]], dtype=int)
    chosen = probabilities[np.arange(len(df)), y_index]
    weights = df[WEIGHT_COLUMN].to_numpy(dtype=float, copy=True)
    return BackendResult(
        backend="biogeme_fixed",
        available=True,
        seconds=seconds,
        loglike=float((weights * np.log(np.clip(chosen, np.finfo(float).tiny, None))).sum()),
        params=params,
        probabilities=probabilities,
    )


def run_biogeme_estimate(df: pd.DataFrame, kind: str, initial: dict[str, float]) -> BackendResult:
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        import biogeme.models as models
        from biogeme.expressions import Beta, Elem, Variable, log
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance
    except ImportError as exc:
        return BackendResult(backend="biogeme", available=False, message=f"Biogeme not found: {exc}")

    names = list(default_params())
    beta_names = names[: len(VARIABLES)]
    threshold_names = names[len(VARIABLES) :]
    internal_threshold_names = [threshold_names[0]] + [
        f"DTH_{index}" for index in range(2, len(CATEGORIES))
    ]
    internal_names = [*beta_names, *internal_threshold_names]
    indicator = df.attrs.get("indicator", "Envir01")
    try:
        database = db.Database(f"torchdcm_ordered_{kind}_estimate", df)
        continuous = (
            Beta("B_MALE", initial["B_MALE"], None, None, 0) * Variable("male")
            + Beta("B_HIGH_EDUCATION", initial["B_HIGH_EDUCATION"], None, None, 0) * Variable("highEducation")
            + Beta("B_GA", initial["B_GA"], None, None, 0) * Variable("haveGA")
            + Beta("B_INCOME", initial["B_INCOME"], None, None, 0) * Variable("ScaledIncome")
        )
        first_threshold = Beta(
            threshold_names[0], initial[threshold_names[0]], None, None, 0
        )
        thresholds = [first_threshold]
        for index, threshold_name in enumerate(threshold_names[1:], start=2):
            previous_name = threshold_names[index - 2]
            increment = Beta(
                f"DTH_{index}",
                initial[threshold_name] - initial[previous_name],
                1e-8,
                None,
                0,
            )
            thresholds.append(thresholds[-1] + increment)
        if kind == "logit":
            formulas_by_category = models.ordered_logit_from_thresholds(continuous, 1.0, CATEGORIES, thresholds)
        else:
            formulas_by_category = models.ordered_probit_from_thresholds(continuous, 1.0, CATEGORIES, thresholds)
        logprob = Variable(WEIGHT_COLUMN) * log(Elem(formulas_by_category, Variable(indicator)))
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.model_name = f"torchdcm_ordered_{kind}_estimate"
        biogeme.biogeme_parameters.set_value("save_iterations", False)
        estimate_start = time.perf_counter()
        estimates = biogeme.estimate()
        estimate_seconds = time.perf_counter() - estimate_start
        covariance_start = time.perf_counter()
        covariance = None
        covariance_message = ""
        try:
            covariance = estimates.get_variance_covariance_matrix(EstimateVarianceCovariance.RAO_CRAMER)
            if hasattr(covariance, "loc"):
                covariance = covariance.loc[internal_names, internal_names]
            covariance = np.asarray(covariance, dtype=float)
            transform = np.zeros((len(names), len(internal_names)), dtype=float)
            transform[: len(beta_names), : len(beta_names)] = np.eye(
                len(beta_names)
            )
            threshold_offset = len(beta_names)
            for row in range(len(threshold_names)):
                transform[threshold_offset + row, threshold_offset : threshold_offset + row + 1] = 1.0
            covariance = transform @ covariance @ transform.T
            if not np.isfinite(covariance).all():
                covariance_message = "covariance unavailable: non-finite values"
                covariance = None
        except Exception as exc:
            covariance_message = f"covariance unavailable: {type(exc).__name__}: {exc}"
        covariance_seconds = time.perf_counter() - covariance_start
        internal_values = estimates.get_beta_values()
        params = {name: float(internal_values[name]) for name in beta_names}
        current_threshold = float(internal_values[threshold_names[0]])
        params[threshold_names[0]] = current_threshold
        for index, threshold_name in enumerate(threshold_names[1:], start=2):
            current_threshold += float(internal_values[f"DTH_{index}"])
            params[threshold_name] = current_threshold
        fixed = run_biogeme_fixed(df, kind, params)
        return BackendResult(
            backend="biogeme",
            available=True,
            seconds=estimation_covariance_total(estimate_seconds, covariance_seconds),
            estimate_seconds=estimate_seconds,
            covariance_seconds=covariance_seconds,
            loglike=float(estimates.final_log_likelihood),
            params=params,
            covariance=covariance,
            probabilities=fixed.probabilities if fixed.available else None,
            message=covariance_message,
        )
    except Exception as exc:
        return BackendResult(backend="biogeme", available=False, message=f"{type(exc).__name__}: {exc}")


def run_apollo_estimate(
    df: pd.DataFrame,
    kind: str,
    initial: dict[str, float],
    max_iter: int,
    timeout: int,
) -> BackendResult:
    runner = Path(__file__).with_name("apollo") / "R" / "run_ordered.R"
    rscript = os.environ.get("TORCHDCM_RSCRIPT", "Rscript")
    indicator = df.attrs.get("indicator", "Envir01")
    names = list(default_params())
    try:
        with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_ordered_{kind}_") as temp:
            temp_dir = Path(temp)
            data_path = temp_dir / "data.csv"
            spec_path = temp_dir / "spec.json"
            output_path = temp_dir / "result.json"
            apollo_frame = df.copy()
            apollo_frame.insert(0, "obs_id", np.arange(1, len(apollo_frame) + 1))
            apollo_frame.to_csv(data_path, index=False)
            spec = {
                "model_name": f"torchdcm_ordered_{kind}_{indicator.lower()}",
                "kind": kind,
                "id_col": "obs_id",
                "outcome_col": indicator,
                "weight_col": WEIGHT_COLUMN,
                "categories": CATEGORIES,
                "variables": dict(zip(names[: len(VARIABLES)], VARIABLES)),
                "thresholds": names[len(VARIABLES) :],
                "parameters": initial,
                "max_iter": max_iter,
            }
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            start = time.perf_counter()
            completed = subprocess.run(
                [
                    rscript,
                    str(runner),
                    "--data",
                    str(data_path),
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            wall_seconds = time.perf_counter() - start
            if completed.returncode != 0 or not output_path.exists():
                message = completed.stderr.strip() or completed.stdout.strip()
                return BackendResult(
                    backend="apollo",
                    available=False,
                    message=message or f"Apollo exited with code {completed.returncode}",
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        return BackendResult(
            backend="apollo",
            available=False,
            message=f"timeout after {timeout} seconds",
        )
    except Exception as exc:
        return BackendResult(
            backend="apollo",
            available=False,
            message=f"{type(exc).__name__}: {exc}",
        )

    params = {name: float(payload["estimates"][name]) for name in names}
    covariance = np.asarray(payload.get("covariance"), dtype=float)
    covariance_names = payload.get("covariance_names", names)
    if covariance.ndim == 2 and covariance_names != names:
        order = [covariance_names.index(name) for name in names]
        covariance = covariance[np.ix_(order, order)]
    timing = payload.get("timing", {})
    estimate_seconds = timing.get("estimate_seconds")
    covariance_seconds = timing.get("covariance_seconds")
    total_seconds = (
        estimation_covariance_total(estimate_seconds, covariance_seconds)
        if estimate_seconds is not None and covariance_seconds is not None
        else wall_seconds
    )
    convergence = payload.get("convergence", {})
    return BackendResult(
        backend="apollo",
        available=True,
        seconds=float(total_seconds),
        estimate_seconds=float(estimate_seconds) if estimate_seconds is not None else None,
        covariance_seconds=float(covariance_seconds) if covariance_seconds is not None else None,
        loglike=float(payload["loglike"]),
        params=params,
        covariance=covariance if covariance.ndim == 2 else None,
        probabilities=ordered_probabilities(df, kind, params),
        message=str(convergence.get("message", "")),
    )


def compare_to_reference(results: list[BackendResult], reference: str) -> None:
    ref = next(result for result in results if result.backend == reference and result.available)
    names = list(ref.params or {})
    for result in results:
        if not result.available:
            continue
        result.ll_diff = result.loglike - ref.loglike  # type: ignore[attr-defined]
        result.max_abs_param_diff = max(abs(result.params[name] - ref.params[name]) for name in names)  # type: ignore[attr-defined]
        if result.probabilities is not None and ref.probabilities is not None:
            result.max_abs_probability_diff = float(np.max(np.abs(result.probabilities - ref.probabilities)))  # type: ignore[attr-defined]
        else:
            result.max_abs_probability_diff = None  # type: ignore[attr-defined]
        if result.covariance is not None and ref.covariance is not None:
            result.max_abs_covariance_diff = float(np.max(np.abs(result.covariance - ref.covariance)))  # type: ignore[attr-defined]
            se = np.sqrt(np.diag(result.covariance))
            ref_se = np.sqrt(np.diag(ref.covariance))
            result.max_abs_se_diff = float(np.max(np.abs(se - ref_se)))  # type: ignore[attr-defined]
            beta = np.asarray([result.params[name] for name in names], dtype=float)
            ref_beta = np.asarray([ref.params[name] for name in names], dtype=float)
            result.max_abs_t_diff = float(np.max(np.abs(beta / se - ref_beta / ref_se)))  # type: ignore[attr-defined]
        else:
            result.max_abs_covariance_diff = None  # type: ignore[attr-defined]
            result.max_abs_se_diff = None  # type: ignore[attr-defined]
            result.max_abs_t_diff = None  # type: ignore[attr-defined]


def print_results(
    results: list[BackendResult],
    reference: str,
    n_obs: int,
    kind: str,
    mode: str,
    indicator: str,
):
    ref = next(result for result in results if result.backend == reference and result.available)
    print(f"case: biogeme_optima_ordered_{indicator.lower()}_{kind}")
    print(f"mode: {mode}")
    print(f"n_obs: {n_obs}")
    print(f"runtime_policy: {runtime_policy_metadata()}")
    print("alignment:")
    if mode == "fixed":
        print("  benchmark_mode: fixed_likelihood_replay")
        print("  estimated_backend: none")
    elif mode == "full-estimation":
        print("  benchmark_mode: full_estimation")
        print("  estimated_backend: each backend estimates independently")
    else:
        print("  benchmark_mode: torchdcm_full_estimation_then_fixed_replay")
        print("  estimated_backend: torchdcm")
    print("  data_source: biogeme.data.optima/data/optima.dat")
    print("  source_loader: biogeme.data.optima.read_data")
    print(f"  outcome: {indicator} Likert ordered categories [1, 2, 3, 4, 5, 6]")
    print("  latent_index: B_MALE*male + B_HIGH_EDUCATION*highEducation + B_GA*haveGA + B_INCOME*ScaledIncome")
    print("  thresholds: shared explicit thresholds")
    print("  weights: normalized_weight")
    print("  biogeme_model_function: ordered_logit_from_thresholds / ordered_probit_from_thresholds")
    print("  example_alignment: Biogeme Optima case-study data and Biogeme ordered model API")
    print(f"  reference: {reference}")
    print()
    print(
        f"{'backend':<18}{'available':>10}{'total_s':>12}{'estimate_s':>12}{'cov_s':>12}"
        f"{'loglike':>18}{'ll_diff':>14}{'param_diff':>14}{'prob_diff':>14}{'cov_diff':>14}{'se_diff':>14}{'t_diff':>14}"
    )
    for result in results:
        if not result.available:
            print(
                f"{result.backend:<18}{str(result.available):>10}"
                f"{'':>12}{'':>12}{'':>12}{'':>18}{'':>14}{'':>14}{'':>14}{'':>14}{'':>14}{'':>14}  {result.message}"
            )
            continue
        print(
            f"{result.backend:<18}{str(result.available):>10}"
            f"{_fmt_seconds(result.seconds):>12}"
            f"{_fmt_seconds(result.estimate_seconds):>12}"
            f"{_fmt_seconds(result.covariance_seconds):>12}"
            f"{result.loglike:>18.10f}"
            f"{getattr(result, 'll_diff', result.loglike - ref.loglike):>14.3e}"
            f"{_fmt_optional(getattr(result, 'max_abs_param_diff', None)):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_probability_diff', None)):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_covariance_diff', None)):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_se_diff', None)):>14}"
            f"{_fmt_optional(getattr(result, 'max_abs_t_diff', None)):>14}"
        )
    print()
    for result in results:
        if result.available and result.params is not None:
            print(f"{result.backend} params:")
            for name, value in result.params.items():
                print(f"  {name}: {value:.12g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["logit", "probit"], default="logit")
    parser.add_argument("--indicator", default="Envir01")
    parser.add_argument("--n-obs", type=int, default=None)
    parser.add_argument("--mode", choices=["fixed", "fit-replay", "full-estimation"], default="fixed")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--apollo-timeout", type=int, default=300)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    df, data = load_ordered_optima(args.indicator, args.n_obs)
    df.attrs["indicator"] = args.indicator
    if args.mode == "fixed":
        params = default_params()
        torch_result = run_torch_fixed(data, args.kind, params)
        biogeme_result = run_biogeme_fixed(df, args.kind, params)
        results = [torch_result, biogeme_result]
        reference = "torchdcm_fixed"
    else:
        if args.mode == "full-estimation":
            torch_result = run_torch_fit(data, args.kind, args.max_iter)
            biogeme_result = run_biogeme_estimate(df, args.kind, default_params())
            apollo_result = run_apollo_estimate(
                df,
                args.kind,
                default_params(),
                args.max_iter,
                args.apollo_timeout,
            )
            results = [torch_result, biogeme_result, apollo_result]
            reference = "torchdcm_fit"
        else:
            torch_result = run_torch_fit(data, args.kind, args.max_iter)
            params = torch_result.params
            biogeme_result = run_biogeme_fixed(df, args.kind, params)
            results = [torch_result, biogeme_result]
            reference = "torchdcm_fit"
    compare_to_reference(results, reference)
    print_results(results, reference, data.n_obs, args.kind, args.mode, args.indicator)
    if args.json_output:
        payload = {
            "case": f"biogeme_optima_ordered_{args.indicator.lower()}_{args.kind}",
            "mode": args.mode,
            "n_obs": data.n_obs,
            "runtime_policy": runtime_policy_metadata(),
            "reference": reference,
            "backends": [serialize_result(result) for result in results],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3e}"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def serialize_result(result: BackendResult) -> dict:
    return {
        "backend": result.backend,
        "available": result.available,
        "total_s": result.seconds,
        "estimate_s": result.estimate_seconds,
        "covariance_s": result.covariance_seconds,
        "loglike": result.loglike,
        "ll_diff": getattr(result, "ll_diff", None),
        "max_param_diff": getattr(result, "max_abs_param_diff", None),
        "max_probability_diff": getattr(result, "max_abs_probability_diff", None),
        "max_covariance_diff": getattr(result, "max_abs_covariance_diff", None),
        "max_se_diff": getattr(result, "max_abs_se_diff", None),
        "max_t_diff": getattr(result, "max_abs_t_diff", None),
        "params": result.params,
        "message": result.message,
    }


if __name__ == "__main__":
    main()
