from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark_runtime import configure_single_thread_cpu, runtime_policy_metadata

if __name__ == "__main__":
    configure_single_thread_cpu(configure_torch=False)

import numpy as np
import pandas as pd
import torch
from advanced_cases import (
    STARTS_HYBRID,
    STARTS_LATENT,
    STARTS_PANEL,
    make_hybrid_actual,
    make_hybrid_synthetic,
    make_latent_class_actual,
    make_latent_class_synthetic,
    make_panel_actual,
    make_panel_synthetic,
)


@dataclass
class EstimationResult:
    backend: str
    available: bool
    seconds: float | None = None
    loglike: float | None = None
    params: dict[str, float] | None = None
    covariance_available: bool | None = None
    convergence: dict | None = None
    worse_loglike: bool | None = None
    message: str = ""


def _finite_params(values: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in values.values())


def run_torch_full(model, data, max_iter: int) -> EstimationResult:
    print(f"[torchdcm] starting {type(model).__name__}", flush=True)
    try:
        started = time.perf_counter()
        result = model.fit(data, cov_type="classic", max_iter=max_iter)
        seconds = time.perf_counter() - started
        params = {
            name: float(value)
            for name, value in zip(result.param_names, result.values)
        }
        covariance = result.cov_params()
        covariance_available = bool(
            covariance is not None and torch.isfinite(covariance).all()
        )
        result_payload = EstimationResult(
            backend="torchdcm",
            available=math.isfinite(result.loglike) and _finite_params(params),
            seconds=seconds,
            loglike=float(result.loglike),
            params=params,
            covariance_available=covariance_available,
            convergence=result.convergence_status,
        )
        print(
            f"[torchdcm] finished in {seconds:.3f}s; LL={result.loglike:.6f}",
            flush=True,
        )
        return result_payload
    except Exception as exc:
        print(f"[torchdcm] failed: {type(exc).__name__}: {exc}", flush=True)
        return EstimationResult(
            backend="torchdcm",
            available=False,
            message=f"{type(exc).__name__}: {exc}",
        )


def _biogeme_imports():
    import biogeme.biogeme as bio
    import biogeme.database as db
    import biogeme.models as models
    from biogeme.expressions import Beta, PanelLikelihoodTrajectory, Variable, exp, log
    from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance

    return bio, db, models, Beta, PanelLikelihoodTrajectory, Variable, exp, log, EstimateVarianceCovariance


def _estimate_biogeme(
    database,
    log_probability,
    model_name: str,
    parameter_names: list[str],
    max_iter: int,
) -> EstimationResult:
    print(f"[biogeme] starting {model_name}", flush=True)
    try:
        import biogeme.biogeme as bio
        from biogeme.results_processing.variance_covariance import EstimateVarianceCovariance

        estimator = bio.BIOGEME(database, log_probability)
        estimator.model_name = model_name
        estimator.biogeme_parameters.set_value("save_iterations", False)
        estimator.biogeme_parameters.set_value("max_iterations", max_iter)
        started = time.perf_counter()
        estimates = estimator.estimate()
        covariance_available = False
        covariance_message = ""
        try:
            covariance = estimates.get_variance_covariance_matrix(
                EstimateVarianceCovariance.RAO_CRAMER
            )
            covariance_array = (
                covariance.loc[parameter_names, parameter_names].to_numpy(dtype=float)
                if hasattr(covariance, "loc")
                else np.asarray(covariance, dtype=float)
            )
            covariance_available = bool(np.isfinite(covariance_array).all())
        except Exception as exc:
            covariance_message = f"covariance: {type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - started
        values = estimates.get_beta_values()
        params = {name: float(values[name]) for name in parameter_names}
        result_payload = EstimationResult(
            backend="biogeme",
            available=math.isfinite(float(estimates.final_log_likelihood))
            and _finite_params(params),
            seconds=seconds,
            loglike=float(estimates.final_log_likelihood),
            params=params,
            covariance_available=covariance_available,
            convergence={"status": "completed"},
            message=covariance_message,
        )
        print(
            f"[biogeme] finished in {seconds:.3f}s; "
            f"LL={float(estimates.final_log_likelihood):.6f}",
            flush=True,
        )
        return result_payload
    except Exception as exc:
        print(f"[biogeme] failed: {type(exc).__name__}: {exc}", flush=True)
        return EstimationResult(
            backend="biogeme",
            available=False,
            message=f"{type(exc).__name__}: {exc}",
        )


def run_biogeme_latent(
    frame: pd.DataFrame,
    max_iter: int,
) -> EstimationResult:
    try:
        _, db, models, Beta, _, Variable, exp, log, _ = _biogeme_imports()
        database = db.Database(
            "torchdcm_advanced_latent_full", frame.drop(columns=["choice"])
        )
        choice = Variable("choice_code")
        availability = {
            1: Variable("av_A"),
            2: Variable("av_B"),
            3: Variable("av_C"),
        }
        betas = {
            name: Beta(name, value, None, None, 0)
            for name, value in STARTS_LATENT.items()
        }
        class_2_exp = exp(
            betas["CLASS_2"] + betas["CLASS_2_Z"] * Variable("z")
        )
        class_probabilities = [
            1.0 / (1.0 + class_2_exp),
            class_2_exp / (1.0 + class_2_exp),
        ]
        chosen_probability = 0
        for class_index, suffix in enumerate(("C1", "C2")):
            utilities = {
                1: betas[f"B_X_{suffix}"] * Variable("x_A"),
                2: betas[f"ASC_B_{suffix}"]
                + betas[f"B_X_{suffix}"] * Variable("x_B"),
                3: betas[f"ASC_C_{suffix}"]
                + betas[f"B_X_{suffix}"] * Variable("x_C"),
            }
            chosen_probability += class_probabilities[class_index] * models.logit(
                utilities, availability, choice
            )
        return _estimate_biogeme(
            database,
            log(chosen_probability),
            f"torchdcm_advanced_latent_full_{len(frame)}",
            list(STARTS_LATENT),
            max_iter,
        )
    except Exception as exc:
        return EstimationResult(
            backend="biogeme",
            available=False,
            message=f"{type(exc).__name__}: {exc}",
        )


def run_biogeme_hybrid(
    frame: pd.DataFrame,
    draws: np.ndarray,
    max_iter: int,
) -> EstimationResult:
    try:
        _, db, models, Beta, _, Variable, exp, log, _ = _biogeme_imports()
        database = db.Database(
            "torchdcm_advanced_hybrid_full", frame.drop(columns=["choice"])
        )
        choice = Variable("choice_code")
        positive = {"SIGMA_LV", "SIGMA_Y1", "SIGMA_Y2"}
        betas = {
            name: Beta(
                name,
                value,
                1e-5 if name in positive else None,
                None,
                0,
            )
            for name, value in STARTS_HYBRID.items()
        }
        integrated = 0
        for draw in draws:
            latent = (
                betas["G_Q"] * Variable("q")
                + betas["SIGMA_LV"] * float(draw)
            )
            utilities = {
                1: 0,
                2: betas["ASC_B"]
                + betas["B_X"] * Variable("x")
                + betas["B_ATT"] * latent,
            }
            choice_probability = models.logit(utilities, None, choice)
            density_1 = exp(
                -0.5
                * ((Variable("y1") - latent) / betas["SIGMA_Y1"]) ** 2
            ) / (betas["SIGMA_Y1"] * math.sqrt(2.0 * math.pi))
            density_2 = exp(
                -0.5
                * (
                    (
                        Variable("y2")
                        - betas["A2"]
                        - betas["L2"] * latent
                    )
                    / betas["SIGMA_Y2"]
                )
                ** 2
            ) / (betas["SIGMA_Y2"] * math.sqrt(2.0 * math.pi))
            integrated += choice_probability * density_1 * density_2
        return _estimate_biogeme(
            database,
            log(integrated / float(len(draws))),
            f"torchdcm_advanced_hybrid_full_{len(frame)}_{len(draws)}",
            list(STARTS_HYBRID),
            max_iter,
        )
    except Exception as exc:
        return EstimationResult(
            backend="biogeme",
            available=False,
            message=f"{type(exc).__name__}: {exc}",
        )


def run_biogeme_panel(
    frame: pd.DataFrame,
    draws: np.ndarray,
    max_iter: int,
) -> EstimationResult:
    try:
        _, db, models, Beta, PanelLikelihoodTrajectory, Variable, _, log, _ = (
            _biogeme_imports()
        )
        database = db.Database(
            "torchdcm_advanced_panel_full", frame.drop(columns=["choice"])
        )
        database.panel("person_id")
        choice = Variable("choice_code")
        betas = {
            name: Beta(
                name,
                value,
                1e-5 if name == "SIGMA_B_X" else None,
                None,
                0,
            )
            for name, value in STARTS_PANEL.items()
        }
        integrated = 0
        for draw in draws:
            random_beta = betas["B_X"] + betas["SIGMA_B_X"] * float(draw)
            utilities = {
                1: random_beta * Variable("x_A"),
                2: betas["ASC_B"] + random_beta * Variable("x_B"),
                3: betas["ASC_C"] + random_beta * Variable("x_C"),
            }
            if "x_D" in frame.columns:
                utilities[4] = betas["ASC_D"] + random_beta * Variable("x_D")
            integrated += PanelLikelihoodTrajectory(
                models.logit(utilities, None, choice)
            )
        return _estimate_biogeme(
            database,
            log(integrated / float(len(draws))),
            f"torchdcm_advanced_panel_full_{frame['person_id'].nunique()}_{len(draws)}",
            list(STARTS_PANEL),
            max_iter,
        )
    except Exception as exc:
        return EstimationResult(
            backend="biogeme",
            available=False,
            message=f"{type(exc).__name__}: {exc}",
        )


def run_apollo_full(
    kind: str,
    frame: pd.DataFrame,
    starts: dict[str, float],
    positive_parameters: list[str],
    draws: np.ndarray,
    max_iter: int,
    timeout: int,
) -> EstimationResult:
    print(f"[apollo] starting {kind} with {len(frame)} observations", flush=True)
    rscript = os.environ.get("TORCHDCM_RSCRIPT") or shutil.which("Rscript")
    if not rscript:
        return EstimationResult(
            backend="apollo", available=False, message="Rscript not found"
        )
    script = Path(__file__).resolve().parent / "apollo" / "R" / "run_advanced_estimate.R"
    try:
        with tempfile.TemporaryDirectory(prefix="torchdcm_advanced_estimate_") as tmp:
            directory = Path(tmp)
            data_path = directory / "data.csv"
            spec_path = directory / "spec.json"
            output_path = directory / "result.json"
            frame.drop(columns=["choice"], errors="ignore").to_csv(
                data_path, index=False
            )
            payload = {
                "kind": kind,
                "model_name": f"torchdcm_advanced_{kind}_{len(frame)}",
                "id_col": "person_id" if kind == "panel_likelihood" else "id",
                "parameters": starts,
                "positive_parameters": positive_parameters,
                "draws": draws.tolist(),
                "max_iter": max_iter,
            }
            spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            completed = subprocess.run(
                [
                    rscript,
                    str(script),
                    "--data",
                    str(data_path),
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(output_path),
                ],
                text=True,
                capture_output=True,
                timeout=timeout,
                env=os.environ.copy(),
            )
            if completed.returncode != 0 or not output_path.exists():
                return EstimationResult(
                    backend="apollo",
                    available=False,
                    message=(completed.stderr or completed.stdout).strip(),
                )
            result = json.loads(output_path.read_text(encoding="utf-8"))
            params = {
                name: float(value) for name, value in result["estimates"].items()
            }
            result_payload = EstimationResult(
                backend="apollo",
                available=math.isfinite(float(result["loglike"]))
                and _finite_params(params),
                seconds=float(result["timing"]["total_seconds"]),
                loglike=float(result["loglike"]),
                params=params,
                covariance_available=bool(result.get("covariance_available")),
                convergence=result.get("convergence"),
            )
            print(
                f"[apollo] finished in {result_payload.seconds:.3f}s; "
                f"LL={result_payload.loglike:.6f}",
                flush=True,
            )
            return result_payload
    except subprocess.TimeoutExpired:
        print(f"[apollo] timed out after {timeout} seconds", flush=True)
        return EstimationResult(
            backend="apollo",
            available=False,
            message=f"timeout after {timeout} seconds",
        )
    except Exception as exc:
        print(f"[apollo] failed: {type(exc).__name__}: {exc}", flush=True)
        return EstimationResult(
            backend="apollo",
            available=False,
            message=f"{type(exc).__name__}: {exc}",
        )


def summarize_case(
    name: str,
    kind: str,
    data_type: str,
    data_source: str,
    n_obs: int,
    n_units: int,
    n_draws: int,
    results: list[EstimationResult],
    extra: dict,
) -> dict:
    available = [
        result
        for result in results
        if result.available and result.loglike is not None
    ]
    best_loglike = max(
        (float(result.loglike) for result in available), default=None
    )
    tolerance = (
        max(0.25, 1e-5 * abs(best_loglike), 0.01 * n_obs)
        if best_loglike is not None
        else None
    )
    for result in available:
        result.worse_loglike = bool(
            best_loglike - float(result.loglike) > tolerance
        )
    comparable = [result for result in available if not result.worse_loglike]
    return {
        "case": name,
        "kind": kind,
        "data_type": data_type,
        "data_source": data_source,
        "n_obs": n_obs,
        "n_units": n_units,
        "n_draws": n_draws,
        "extra": extra,
        "results": {result.backend: asdict(result) for result in results},
        "best_loglike": best_loglike,
        "loglike_tolerance": tolerance,
        "consistent": None
        if len(comparable) < 2
        else all(
            best_loglike - float(result.loglike) <= tolerance
            for result in comparable
        ),
    }


def _run_selected(
    backends: set[str],
    runners: list[tuple[str, object]],
) -> list[EstimationResult]:
    results = []
    for backend, runner in runners:
        if backend in backends:
            results.append(runner())
    return results


def _include_case(case_name: str, case_filters: set[str] | None) -> bool:
    return case_filters is None or case_name in case_filters


def _warm_torch_backend() -> None:
    """Remove one-time optimizer/autograd setup from timed estimations."""
    try:
        _, data, model, _ = make_latent_class_synthetic(128, seed=19)
        model.fit(data, cov_type="classic", max_iter=2)
    except Exception:
        pass


def _case_grid(profile: str):
    if profile == "smoke":
        return {
            "latent_synthetic": [(500,)],
            "latent_actual": [(500,)],
            "hybrid_synthetic": [(500, 32)],
            "hybrid_actual": [(500, 32)],
            "panel_synthetic": [(250, 2, 32)],
            "panel_actual": [(100, 32)],
        }
    return {
        "latent_synthetic": [(2_000,), (5_000,), (10_000,)],
        "latent_actual": [(2_000,), (3_500,), (5_000,)],
        "hybrid_synthetic": [(500, 32), (2_000, 64), (10_000, 128)],
        "hybrid_actual": [(500, 32), (1_000, 64), (1_298, 128)],
        "panel_synthetic": [(250, 2, 32), (500, 4, 64), (1_250, 8, 128)],
        "panel_actual": [(100, 32), (250, 64), (348, 128)],
    }


def run_suite(
    profile: str,
    max_iter: int,
    timeout: int,
    kinds: set[str] | None,
    backends: set[str] | None,
    case_filters: set[str] | None,
) -> dict:
    kinds = kinds or {"latent_class", "hybrid_choice", "panel_likelihood"}
    backends = backends or {"torchdcm", "biogeme", "apollo"}
    grid = _case_grid(profile)
    cases = []
    if "torchdcm" in backends:
        _warm_torch_backend()

    if "latent_class" in kinds:
        for (n_obs,) in grid["latent_synthetic"]:
            case_name = f"Synthetic {n_obs:,}"
            if not _include_case(case_name, case_filters):
                continue
            frame, data, model, _ = make_latent_class_synthetic(
                n_obs, seed=100 + n_obs
            )
            results = _run_selected(
                backends,
                [
                    ("torchdcm", lambda: run_torch_full(model, data, max_iter)),
                    ("biogeme", lambda: run_biogeme_latent(frame, max_iter)),
                    (
                        "apollo",
                        lambda: run_apollo_full(
                            "latent_class",
                            frame,
                            STARTS_LATENT,
                            [],
                            np.asarray([], dtype=float),
                            max_iter,
                            timeout,
                        ),
                    ),
                ],
            )
            cases.append(
                summarize_case(
                    case_name,
                    "latent_class",
                    "Synthetic",
                    "Controlled latent-class DGP",
                    len(frame),
                    len(frame),
                    0,
                    results,
                    {"classes": 2, "membership_covariates": 1},
                )
            )
        for (n_obs,) in grid["latent_actual"]:
            case_name = f"Swissmetro {n_obs:,}"
            if not _include_case(case_name, case_filters):
                continue
            frame, data, model, _ = make_latent_class_actual(n_obs)
            case_name = f"Swissmetro {len(frame):,}"
            results = _run_selected(
                backends,
                [
                    ("torchdcm", lambda: run_torch_full(model, data, max_iter)),
                    ("biogeme", lambda: run_biogeme_latent(frame, max_iter)),
                    (
                        "apollo",
                        lambda: run_apollo_full(
                            "latent_class",
                            frame,
                            STARTS_LATENT,
                            [],
                            np.asarray([], dtype=float),
                            max_iter,
                            timeout,
                        ),
                    ),
                ],
            )
            cases.append(
                summarize_case(
                    case_name,
                    "latent_class",
                    "Actual",
                    "Swissmetro",
                    len(frame),
                    len(frame),
                    0,
                    results,
                    {"classes": 2, "membership_covariates": 1},
                )
            )

    if "hybrid_choice" in kinds:
        for n_obs, n_draws in grid["hybrid_synthetic"]:
            case_name = f"Synthetic {n_obs:,}"
            if not _include_case(case_name, case_filters):
                continue
            frame, data, model, _, draws = make_hybrid_synthetic(
                n_obs, n_draws, seed=200 + n_obs
            )
            results = _run_selected(
                backends,
                [
                    ("torchdcm", lambda: run_torch_full(model, data, max_iter)),
                    (
                        "biogeme",
                        lambda: run_biogeme_hybrid(frame, draws, max_iter),
                    ),
                    (
                        "apollo",
                        lambda: run_apollo_full(
                            "hybrid_choice",
                            frame,
                            STARTS_HYBRID,
                            ["SIGMA_LV", "SIGMA_Y1", "SIGMA_Y2"],
                            draws,
                            max_iter,
                            timeout,
                        ),
                    ),
                ],
            )
            cases.append(
                summarize_case(
                    case_name,
                    "hybrid_choice",
                    "Synthetic",
                    "Controlled hybrid-choice DGP",
                    len(frame),
                    len(frame),
                    n_draws,
                    results,
                    {"latent_variables": 1, "continuous_indicators": 2},
                )
            )
        for n_obs, n_draws in grid["hybrid_actual"]:
            case_name = f"Optima {n_obs:,}"
            if not _include_case(case_name, case_filters):
                continue
            frame, data, model, _, draws = make_hybrid_actual(
                n_obs, n_draws, seed=400 + n_obs
            )
            case_name = f"Optima {len(frame):,}"
            results = _run_selected(
                backends,
                [
                    ("torchdcm", lambda: run_torch_full(model, data, max_iter)),
                    (
                        "biogeme",
                        lambda: run_biogeme_hybrid(frame, draws, max_iter),
                    ),
                    (
                        "apollo",
                        lambda: run_apollo_full(
                            "hybrid_choice",
                            frame,
                            STARTS_HYBRID,
                            ["SIGMA_LV", "SIGMA_Y1", "SIGMA_Y2"],
                            draws,
                            max_iter,
                            timeout,
                        ),
                    ),
                ],
            )
            cases.append(
                summarize_case(
                    case_name,
                    "hybrid_choice",
                    "Actual",
                    "Optima",
                    len(frame),
                    len(frame),
                    n_draws,
                    results,
                    {"latent_variables": 1, "continuous_indicators": 2},
                )
            )

    if "panel_likelihood" in kinds:
        for n_units, choices_per_unit, n_draws in grid["panel_synthetic"]:
            case_name = f"Synthetic {n_units:,}x{choices_per_unit}"
            if not _include_case(case_name, case_filters):
                continue
            frame, data, model, _, draws = make_panel_synthetic(
                n_units,
                choices_per_unit,
                n_draws,
                seed=500 + n_units + choices_per_unit,
            )
            results = _run_selected(
                backends,
                [
                    ("torchdcm", lambda: run_torch_full(model, data, max_iter)),
                    ("biogeme", lambda: run_biogeme_panel(frame, draws, max_iter)),
                    (
                        "apollo",
                        lambda: run_apollo_full(
                            "panel_likelihood",
                            frame,
                            STARTS_PANEL,
                            ["SIGMA_B_X"],
                            draws,
                            max_iter,
                            timeout,
                        ),
                    ),
                ],
            )
            cases.append(
                summarize_case(
                    case_name,
                    "panel_likelihood",
                    "Synthetic",
                    "Controlled panel DGP",
                    len(frame),
                    n_units,
                    n_draws,
                    results,
                    {
                        "alternatives": 4,
                        "choices_per_unit": choices_per_unit,
                        "random_coefficients": 1,
                    },
                )
            )
        for n_units, n_draws in grid["panel_actual"]:
            case_name = f"Electricity {n_units:,}"
            if not _include_case(case_name, case_filters):
                continue
            frame, data, model, _, draws = make_panel_actual(
                n_units, n_draws, seed=800 + n_units
            )
            results = _run_selected(
                backends,
                [
                    ("torchdcm", lambda: run_torch_full(model, data, max_iter)),
                    ("biogeme", lambda: run_biogeme_panel(frame, draws, max_iter)),
                    (
                        "apollo",
                        lambda: run_apollo_full(
                            "panel_likelihood",
                            frame,
                            STARTS_PANEL,
                            ["SIGMA_B_X"],
                            draws,
                            max_iter,
                            timeout,
                        ),
                    ),
                ],
            )
            cases.append(
                summarize_case(
                    case_name,
                    "panel_likelihood",
                    "Actual",
                    "Electricity",
                    len(frame),
                    n_units,
                    n_draws,
                    results,
                    {
                        "alternatives": 4,
                        "choices_per_unit": 12,
                        "random_coefficients": 1,
                    },
                )
            )

    return {
        "benchmark": "advanced_full_estimation_validation",
        "timing_scope": "full optimization plus classic covariance construction",
        "runtime_policy": runtime_policy_metadata(),
        "profile": profile,
        "max_iter": max_iter,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=["latent_class", "hybrid_choice", "panel_likelihood"],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["torchdcm", "biogeme", "apollo"],
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Exact displayed case names, such as 'Synthetic 2,000'.",
    )
    args = parser.parse_args()
    configure_single_thread_cpu(configure_torch=True)
    payload = run_suite(
        args.profile,
        args.max_iter,
        args.timeout,
        None if args.kinds is None else set(args.kinds),
        None if args.backends is None else set(args.backends),
        None if args.cases is None else set(args.cases),
    )
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
