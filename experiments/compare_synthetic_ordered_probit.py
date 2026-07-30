from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark_runtime import (
    configure_single_thread_cpu,
    estimation_covariance_total,
    runtime_policy_metadata,
)


configure_single_thread_cpu(configure_torch=True)

import numpy as np
import pandas as pd
import torch

from torchdcm import Beta, OrderedChoiceDataset, OrderedLogit, OrderedProbit


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    n_obs: int
    n_categories: int
    n_variables: int
    rho: float
    seed: int


CASES = [
    SyntheticCase("Sample small", 500, 6, 4, 0.0, 20260721),
    SyntheticCase("Baseline", 2_000, 6, 4, 0.0, 20260722),
    SyntheticCase("Sample large", 10_000, 6, 4, 0.0, 20260723),
    SyntheticCase("Levels low", 2_000, 3, 4, 0.0, 20260724),
    SyntheticCase("Levels high", 2_000, 10, 4, 0.0, 20260725),
    SyntheticCase("Variables low", 2_000, 6, 2, 0.0, 20260726),
    SyntheticCase("Variables high", 2_000, 6, 12, 0.0, 20260727),
    SyntheticCase("Correlation medium", 2_000, 6, 4, 0.5, 20260728),
    SyntheticCase("Correlation high", 2_000, 6, 4, 0.9, 20260729),
    SyntheticCase("Joint large", 20_000, 10, 12, 0.5, 20260730),
]


def variable_names(case: SyntheticCase) -> list[str]:
    return [f"x_{index}" for index in range(1, case.n_variables + 1)]


def beta_names(case: SyntheticCase) -> list[str]:
    return [f"B_X_{index}" for index in range(1, case.n_variables + 1)]


def threshold_names(case: SyntheticCase) -> list[str]:
    return [f"TH_{index}" for index in range(1, case.n_categories)]


def generate_case(
    case: SyntheticCase,
    kind: str,
) -> tuple[pd.DataFrame, OrderedChoiceDataset, dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(case.seed)
    covariance = np.full((case.n_variables, case.n_variables), case.rho)
    np.fill_diagonal(covariance, 1.0)
    design = rng.multivariate_normal(
        np.zeros(case.n_variables), covariance, size=case.n_obs
    )

    raw_beta = np.linspace(0.4, 1.0, case.n_variables)
    raw_beta[1::2] *= -1.0
    beta = 0.9 * raw_beta / np.linalg.norm(raw_beta)
    quantile_levels = np.arange(1, case.n_categories) / case.n_categories
    calibration_size = 100_000
    calibration_design = rng.multivariate_normal(
        np.zeros(case.n_variables), covariance, size=calibration_size
    )
    if kind == "logit":
        calibration_error = rng.logistic(size=calibration_size)
        error = rng.logistic(size=case.n_obs)
    else:
        calibration_error = rng.standard_normal(calibration_size)
        error = rng.standard_normal(case.n_obs)
    calibration_latent = calibration_design @ beta + calibration_error
    thresholds = np.quantile(calibration_latent, quantile_levels)
    latent = design @ beta + error
    outcome = np.searchsorted(thresholds, latent, side="right") + 1

    variables = variable_names(case)
    frame = pd.DataFrame(design, columns=variables)
    frame["outcome"] = outcome
    frame["weight"] = 1.0
    categories = list(range(1, case.n_categories + 1))
    data = OrderedChoiceDataset.from_dataframe(
        frame,
        outcome="outcome",
        variables=variables,
        categories=categories,
        weight="weight",
    )

    counts = np.bincount(outcome, minlength=case.n_categories + 1)[1:]
    cumulative = np.cumsum(counts[:-1]) / counts.sum()
    cumulative = np.clip(cumulative, 1e-4, 1.0 - 1e-4)
    cumulative_tensor = torch.as_tensor(cumulative, dtype=torch.float64)
    if kind == "logit":
        initial_thresholds = torch.logit(cumulative_tensor).numpy()
    else:
        initial_thresholds = torch.distributions.Normal(0.0, 1.0).icdf(
            cumulative_tensor
        ).numpy()
    initial = {
        **{name: 0.0 for name in beta_names(case)},
        **{
            name: float(value)
            for name, value in zip(threshold_names(case), initial_thresholds)
        },
    }
    truth = {
        **{name: float(value) for name, value in zip(beta_names(case), beta)},
        **{
            name: float(value)
            for name, value in zip(threshold_names(case), thresholds)
        },
    }
    return frame, data, initial, truth


def make_torch_model(
    case: SyntheticCase,
    initial: dict[str, float],
    kind: str,
) -> OrderedLogit | OrderedProbit:
    variables = variable_names(case)
    parameters = beta_names(case)
    latent = Beta(parameters[0], init=initial[parameters[0]]) * variables[0]
    for parameter, variable in zip(parameters[1:], variables[1:]):
        latent = latent + Beta(parameter, init=initial[parameter]) * variable
    thresholds = [
        Beta(name, init=initial[name]) for name in threshold_names(case)
    ]
    model_class = OrderedLogit if kind == "logit" else OrderedProbit
    return model_class(latent, thresholds, max_iter=300)


def run_torch(
    case: SyntheticCase,
    data: OrderedChoiceDataset,
    initial: dict[str, float],
    kind: str,
    max_iter: int,
) -> dict:
    model = make_torch_model(case, initial, kind)
    data = data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    internal = torch.cat(
        [
            compiled.free_initial[: len(compiled.beta_names)],
            model._threshold_to_internal(compiled.threshold_initial),
        ]
    ).detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [internal],
        max_iter=max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )
    closure_evaluations = 0

    def closure():
        nonlocal closure_evaluations
        optimizer.zero_grad(set_to_none=True)
        natural = model._internal_to_natural(internal, compiled)
        loss = -model.loglike(natural, data, compiled)
        loss.backward()
        closure_evaluations += 1
        return loss

    estimate_start = time.perf_counter()
    optimizer.step(closure)
    estimate_seconds = time.perf_counter() - estimate_start
    final_internal = internal.detach().clone().requires_grad_(True)
    final_natural = model._internal_to_natural(final_internal, compiled)
    loglike = model.loglike(final_natural, data, compiled)
    internal_gradient = torch.autograd.grad(loglike, final_internal)[0]

    covariance_start = time.perf_counter()
    hessian = torch.autograd.functional.hessian(
        lambda point: model.loglike(
            model._internal_to_natural(point, compiled), data, compiled
        ),
        final_internal,
    )
    covariance_internal = torch.linalg.pinv(-hessian.detach(), hermitian=True)
    transform = model._natural_jacobian(final_internal.detach(), compiled)
    covariance = transform @ covariance_internal @ transform.T
    covariance_seconds = time.perf_counter() - covariance_start
    probabilities = model.predict_proba(data, final_natural.detach())
    return {
        "backend": "torchdcm",
        "available": True,
        "runtime": estimate_seconds + covariance_seconds,
        "estimate_seconds": estimate_seconds,
        "covariance_seconds": covariance_seconds,
        "loglike": float(loglike.detach()),
        "params": {
            name: float(value)
            for name, value in zip(compiled.free_names, final_natural.detach())
        },
        "covariance": covariance.detach().cpu().numpy(),
        "probabilities": probabilities.detach().cpu().numpy(),
        "closure_evaluations": closure_evaluations,
        "gradient_norm": float(torch.linalg.vector_norm(internal_gradient.detach())),
    }


def biogeme_probability_replay(
    case: SyntheticCase,
    frame: pd.DataFrame,
    params: dict[str, float],
    kind: str,
) -> np.ndarray:
    import biogeme.biogeme as bio
    import biogeme.database as db
    import biogeme.models as models
    from biogeme.expressions import Variable

    continuous = sum(
        params[parameter] * Variable(variable)
        for parameter, variable in zip(beta_names(case), variable_names(case))
    )
    categories = list(range(1, case.n_categories + 1))
    thresholds = [params[name] for name in threshold_names(case)]
    if kind == "logit":
        formulas = models.ordered_logit_from_thresholds(
            continuous, 1.0, categories, thresholds
        )
    else:
        formulas = models.ordered_probit_from_thresholds(
            continuous, 1.0, categories, thresholds
        )
    simulator = bio.BIOGEME(
        db.Database(f"ordered_probit_replay_{case.seed}", frame),
        {f"prob_{category}": formulas[category] for category in categories},
    )
    simulated = simulator.simulate({})
    return np.column_stack(
        [simulated[f"prob_{category}"].to_numpy(dtype=float) for category in categories]
    )


def run_biogeme(
    case: SyntheticCase,
    frame: pd.DataFrame,
    initial: dict[str, float],
    kind: str,
) -> dict:
    import biogeme.biogeme as bio
    import biogeme.database as db
    import biogeme.models as models
    from biogeme.expressions import Beta as BiogemeBeta
    from biogeme.expressions import Elem, Variable, log
    from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance

    continuous = sum(
        BiogemeBeta(parameter, initial[parameter], None, None, 0)
        * Variable(variable)
        for parameter, variable in zip(beta_names(case), variable_names(case))
    )
    categories = list(range(1, case.n_categories + 1))
    natural_threshold_names = threshold_names(case)
    internal_threshold_names = [natural_threshold_names[0]] + [
        f"DTH_{index}" for index in range(2, case.n_categories)
    ]
    first_threshold = BiogemeBeta(
        natural_threshold_names[0],
        initial[natural_threshold_names[0]],
        None,
        None,
        0,
    )
    thresholds = [first_threshold]
    for index, threshold_name in enumerate(
        natural_threshold_names[1:], start=2
    ):
        previous_name = natural_threshold_names[index - 2]
        increment = BiogemeBeta(
            f"DTH_{index}",
            initial[threshold_name] - initial[previous_name],
            1e-8,
            None,
            0,
        )
        thresholds.append(thresholds[-1] + increment)
    if kind == "logit":
        formulas = models.ordered_logit_from_thresholds(
            continuous, 1.0, categories, thresholds
        )
    else:
        formulas = models.ordered_probit_from_thresholds(
            continuous, 1.0, categories, thresholds
        )
    log_probability = log(Elem(formulas, Variable("outcome")))
    estimator = bio.BIOGEME(
        db.Database(f"ordered_{kind}_{case.seed}", frame), log_probability
    )
    estimator.model_name = f"ordered_{kind}_{case.seed}"
    estimator.biogeme_parameters.set_value("save_iterations", False)
    estimate_start = time.perf_counter()
    estimates = estimator.estimate()
    estimate_seconds = time.perf_counter() - estimate_start

    natural_names = [*beta_names(case), *natural_threshold_names]
    internal_names = [*beta_names(case), *internal_threshold_names]
    covariance_start = time.perf_counter()
    covariance_object = estimates.get_variance_covariance_matrix(
        EstimateVarianceCovariance.RAO_CRAMER
    )
    if hasattr(covariance_object, "loc"):
        covariance_object = covariance_object.loc[internal_names, internal_names]
    covariance_internal = np.asarray(covariance_object, dtype=float)
    transform = np.zeros(
        (len(natural_names), len(internal_names)), dtype=float
    )
    n_beta = len(beta_names(case))
    transform[:n_beta, :n_beta] = np.eye(n_beta)
    for row in range(len(natural_threshold_names)):
        transform[n_beta + row, n_beta : n_beta + row + 1] = 1.0
    covariance = transform @ covariance_internal @ transform.T
    covariance_seconds = time.perf_counter() - covariance_start
    internal_values = estimates.get_beta_values()
    params = {
        name: float(internal_values[name]) for name in beta_names(case)
    }
    current_threshold = float(internal_values[natural_threshold_names[0]])
    params[natural_threshold_names[0]] = current_threshold
    for index, threshold_name in enumerate(
        natural_threshold_names[1:], start=2
    ):
        current_threshold += float(internal_values[f"DTH_{index}"])
        params[threshold_name] = current_threshold
    return {
        "backend": "biogeme",
        "available": True,
        "runtime": estimation_covariance_total(
            estimate_seconds, covariance_seconds
        ),
        "estimate_seconds": estimate_seconds,
        "covariance_seconds": covariance_seconds,
        "loglike": float(estimates.final_log_likelihood),
        "params": params,
        "covariance": covariance,
        "probabilities": biogeme_probability_replay(case, frame, params, kind),
    }


def ordered_probability_replay(
    case: SyntheticCase,
    frame: pd.DataFrame,
    params: dict[str, float],
    kind: str,
) -> np.ndarray:
    design = frame[variable_names(case)].to_numpy(dtype=float)
    coefficients = np.asarray(
        [params[name] for name in beta_names(case)], dtype=float
    )
    thresholds = np.asarray(
        [params[name] for name in threshold_names(case)], dtype=float
    )
    bounds = np.concatenate(([-np.inf], thresholds, [np.inf]))
    differences = torch.as_tensor(
        bounds[None, :] - (design @ coefficients)[:, None],
        dtype=torch.float64,
    )
    if kind == "logit":
        cumulative = torch.sigmoid(differences)
    else:
        cumulative = torch.special.ndtr(differences)
    return (cumulative[:, 1:] - cumulative[:, :-1]).numpy()


def run_apollo(
    case: SyntheticCase,
    frame: pd.DataFrame,
    initial: dict[str, float],
    kind: str,
    max_iter: int,
    timeout: int,
) -> dict:
    runner = Path(__file__).with_name("apollo") / "R" / "run_ordered.R"
    rscript = os.environ.get("TORCHDCM_RSCRIPT", "Rscript")
    names = [*beta_names(case), *threshold_names(case)]
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_apollo_ordered_{kind}_") as temp:
        temp_dir = Path(temp)
        data_path = temp_dir / "data.csv"
        spec_path = temp_dir / "spec.json"
        output_path = temp_dir / "result.json"
        apollo_frame = frame.copy()
        apollo_frame.insert(0, "obs_id", np.arange(1, len(apollo_frame) + 1))
        apollo_frame.to_csv(data_path, index=False)
        spec = {
            "model_name": f"torchdcm_ordered_{kind}_{case.seed}",
            "kind": kind,
            "id_col": "obs_id",
            "outcome_col": "outcome",
            "weight_col": "weight",
            "categories": list(range(1, case.n_categories + 1)),
            "variables": dict(zip(beta_names(case), variable_names(case))),
            "thresholds": threshold_names(case),
            "parameters": initial,
            "max_iter": max_iter,
        }
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
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
        if completed.returncode != 0 or not output_path.exists():
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                message or f"Apollo exited with code {completed.returncode}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))

    params = {name: float(payload["estimates"][name]) for name in names}
    covariance = np.asarray(payload["covariance"], dtype=float)
    covariance_names = payload.get("covariance_names", names)
    if covariance_names != names:
        order = [covariance_names.index(name) for name in names]
        covariance = covariance[np.ix_(order, order)]
    timing = payload["timing"]
    estimate_seconds = float(timing["estimate_seconds"])
    covariance_seconds = float(timing["covariance_seconds"])
    return {
        "backend": "apollo",
        "available": True,
        "runtime": estimation_covariance_total(
            estimate_seconds, covariance_seconds
        ),
        "estimate_seconds": estimate_seconds,
        "covariance_seconds": covariance_seconds,
        "loglike": float(payload["loglike"]),
        "params": params,
        "covariance": covariance,
        "probabilities": ordered_probability_replay(
            case, frame, params, kind
        ),
        "message": str(payload.get("convergence", {}).get("message", "")),
    }


def rmse(params: dict[str, float], truth: dict[str, float], names: list[str]) -> float:
    return math.sqrt(
        sum((params[name] - truth[name]) ** 2 for name in names) / len(names)
    )


def run_case(
    case: SyntheticCase,
    kind: str,
    max_iter: int,
    apollo_timeout: int,
) -> dict:
    frame, data, initial, truth = generate_case(case, kind)
    torch_result = run_torch(case, data, initial, kind, max_iter)
    biogeme_result = run_biogeme(case, frame, initial, kind)
    apollo_result = run_apollo(
        case, frame, initial, kind, max_iter, apollo_timeout
    )
    names = [*beta_names(case), *threshold_names(case)]
    comparisons = {}
    for result in (biogeme_result, apollo_result):
        backend = result["backend"]
        loglike_difference = abs(
            torch_result["loglike"] - result["loglike"]
        )
        parameter_difference = max(
            abs(torch_result["params"][name] - result["params"][name])
            for name in names
        )
        probability_difference = float(
            np.max(
                np.abs(
                    torch_result["probabilities"] - result["probabilities"]
                )
            )
        )
        covariance_difference = float(
            np.max(
                np.abs(torch_result["covariance"] - result["covariance"])
            )
        )
        comparisons[backend] = {
            "abs_loglike_diff": loglike_difference,
            "max_param_diff": parameter_difference,
            "max_probability_diff": probability_difference,
            "max_covariance_diff": covariance_difference,
            "consistent": bool(
                loglike_difference
                <= max(1e-5, 1e-8 * abs(torch_result["loglike"]))
                and parameter_difference <= 1e-3
                and probability_difference <= 1e-4
            ),
        }
    consistent = all(item["consistent"] for item in comparisons.values())
    category_counts = np.bincount(
        data.y.detach().cpu().numpy(), minlength=case.n_categories
    )
    return {
        "case": case.name,
        "n_obs": case.n_obs,
        "n_categories": case.n_categories,
        "n_variables": case.n_variables,
        "rho": case.rho,
        "seed": case.seed,
        "category_counts": category_counts.tolist(),
        "truth": truth,
        "initial": initial,
        "torchdcm": serialize_backend(torch_result),
        "biogeme": serialize_backend(biogeme_result),
        "apollo": serialize_backend(apollo_result),
        "comparisons_to_torchdcm": comparisons,
        "abs_loglike_diff": max(
            item["abs_loglike_diff"] for item in comparisons.values()
        ),
        "max_param_diff": max(
            item["max_param_diff"] for item in comparisons.values()
        ),
        "max_probability_diff": max(
            item["max_probability_diff"] for item in comparisons.values()
        ),
        "max_covariance_diff": max(
            item["max_covariance_diff"] for item in comparisons.values()
        ),
        "beta_rmse_to_truth": rmse(torch_result["params"], truth, beta_names(case)),
        "threshold_rmse_to_truth": rmse(
            torch_result["params"], truth, threshold_names(case)
        ),
        "consistent": bool(consistent),
    }


def serialize_backend(result: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
        if key not in {"probabilities", "covariance"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--kind", choices=["logit", "probit"], default="probit")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--apollo-timeout", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.case:
        case = next(item for item in CASES if item.name == args.case)
        result = run_case(
            case, args.kind, args.max_iter, args.apollo_timeout
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    rows = []
    with tempfile.TemporaryDirectory(prefix="torchdcm_ordered_synthetic_") as temp:
        temp_dir = Path(temp)
        for index, case in enumerate(CASES):
            print(f"running {case.name}", flush=True)
            output = temp_dir / f"case_{index}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case",
                case.name,
                "--kind",
                args.kind,
                "--max-iter",
                str(args.max_iter),
                "--apollo-timeout",
                str(args.apollo_timeout),
                "--output",
                str(output),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=temp_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired:
                rows.append({"case": case.name, "status": "timeout"})
                continue
            if completed.returncode != 0 or not output.exists():
                rows.append(
                    {
                        "case": case.name,
                        "status": "failed",
                        "message": completed.stderr.strip()
                        or completed.stdout.strip(),
                    }
                )
                continue
            row = json.loads(output.read_text(encoding="utf-8"))
            row["status"] = "completed"
            rows.append(row)

    payload = {
        "benchmark": f"synthetic_ordered_{args.kind}",
        "model": f"Ordered {args.kind}",
        "runtime_policy": runtime_policy_metadata(),
        "tolerances": {
            "loglike": "max(1e-5, 1e-8 * abs(torchdcm_loglike))",
            "parameter": 1e-3,
            "probability": 1e-4,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
