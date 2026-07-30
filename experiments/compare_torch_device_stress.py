from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from torchdcm import ChoiceDataset, MixedLogit, MultinomialLogit, NestedLogit, RandomCoefficient

import compare_generated_choice_battery as generated
import compare_real_mixed_logit_battery as mixed_real
import compare_real_nested_logit_battery as nested_real


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class DeviceStressSpec:
    case: str
    model: str
    n_obs: int
    n_alternatives: int
    n_variables: int
    rho: float
    random_coefficients: int
    n_draws: int
    max_iter: int


def profile_specs(profile: str) -> list[DeviceStressSpec]:
    if profile == "smoke":
        return [
            DeviceStressSpec("torch_device_smoke_mnl", "mnl", 1000, 4, 4, 0.2, 0, 0, 20),
            DeviceStressSpec("torch_device_smoke_nl", "nl", 1000, 4, 4, 0.2, 0, 0, 20),
            DeviceStressSpec("torch_device_smoke_mixl", "mixl", 1000, 4, 4, 0.2, 3, 32, 20),
        ]
    if profile == "calibration":
        return [
            DeviceStressSpec("torch_device_calib_mixl_50k", "mixl", 50000, 12, 12, 0.5, 8, 256, 80),
            DeviceStressSpec("torch_device_calib_mixl_100k", "mixl", 100000, 12, 12, 0.5, 8, 512, 80),
        ]
    if profile == "battery":
        return [
            DeviceStressSpec("torch_device_mnl_250k", "mnl", 250000, 12, 12, 0.5, 0, 0, 80),
            DeviceStressSpec("torch_device_mnl_500k", "mnl", 500000, 12, 12, 0.5, 0, 0, 80),
            DeviceStressSpec("torch_device_mnl_1m", "mnl", 1000000, 12, 12, 0.5, 0, 0, 80),
            DeviceStressSpec("torch_device_nl_100k", "nl", 100000, 12, 12, 0.5, 0, 0, 80),
            DeviceStressSpec("torch_device_nl_250k", "nl", 250000, 12, 12, 0.5, 0, 0, 80),
            DeviceStressSpec("torch_device_nl_500k", "nl", 500000, 12, 12, 0.5, 0, 0, 80),
            DeviceStressSpec("torch_device_mixl_5k", "mixl", 5000, 12, 12, 0.5, 8, 256, 80),
            DeviceStressSpec("torch_device_mixl_10k", "mixl", 10000, 12, 12, 0.5, 8, 256, 80),
            DeviceStressSpec("torch_device_mixl_25k", "mixl", 25000, 12, 12, 0.5, 8, 256, 80),
        ]
    return [
        DeviceStressSpec("torch_device_stress_mixl", "mixl", 100000, 12, 12, 0.5, 8, 512, 80),
    ]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def build_device_nested_case(mnl_case, seed: int):
    """Generate NL choices without the external-software long/wide round trip."""
    midpoint = max(2, len(mnl_case.alternatives) // 2)
    groups = [mnl_case.alternatives[:midpoint], mnl_case.alternatives[midpoint:]]
    nests = {
        "GROUP_A": nested_real.NestSpec(groups[0], init=0.8, fixed=False),
        "GROUP_B": nested_real.NestSpec(groups[1], init=0.8, fixed=False),
    }
    probabilities = generated.nested_choice_probabilities(
        mnl_case.systematic_utility,
        [list(range(0, midpoint)), list(range(midpoint, len(mnl_case.alternatives)))],
        [generated.NL_TRUE_DISSIMILARITY["GROUP_A"], generated.NL_TRUE_DISSIMILARITY["GROUP_B"]],
    )
    rng = np.random.default_rng(
        seed + generated.stable_case_offset(mnl_case.case) + 10_000_019
    )
    choices = generated.sample_choices(rng, probabilities)
    frame = mnl_case.df.copy()
    frame["choice"] = [mnl_case.alternatives[index] for index in choices]
    data = ChoiceDataset.from_wide(
        frame,
        alternatives=mnl_case.alternatives,
        choice="choice",
        variables=mnl_case.feature_columns,
        availability=mnl_case.availability_columns,
        obs_id="obs_id",
    )
    return SimpleNamespace(spec=mnl_case.spec, data=data, nests=nests)


def run_model_path(spec: DeviceStressSpec, case, draws, device: torch.device) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    setup_start = time.perf_counter()
    if spec.model == "mnl":
        model = MultinomialLogit(
            case.spec,
            device=device,
            max_iter=spec.max_iter,
            dtype=torch.float64,
        )
    elif spec.model == "nl":
        model = NestedLogit(
            case.spec,
            nested_real.torch_nests(case),
            device=device,
            max_iter=spec.max_iter,
            dtype=torch.float64,
        )
    else:
        model = MixedLogit(
            case.spec,
            [RandomCoefficient(name, sigma_init=case.sigma_init) for name in case.random_names],
            draws=draws,
            panel=False,
            device=device,
            max_iter=spec.max_iter,
            dtype=torch.float64,
        )
    data = case.data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    if spec.model == "mnl":
        internal_initial = compiled.free_initial
    elif spec.model == "nl":
        internal_initial = torch.cat(
            [
                compiled.free_initial,
                model._lambda_to_internal(compiled.lambda_initial[~compiled.lambda_is_fixed]),
            ]
        )
    else:
        internal_initial = torch.cat(
            [
                compiled.free_initial,
                model._sigma_to_internal(compiled.sigma_initial[~compiled.sigma_is_fixed]),
                compiled.chol_offdiag_initial,
            ]
        )
    internal_params = internal_initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [internal_params],
        max_iter=spec.max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )
    sync_device(device)
    setup_s = time.perf_counter() - setup_start

    closures = {"count": 0}

    def closure():
        optimizer.zero_grad(set_to_none=True)
        natural = (
            internal_params
            if spec.model == "mnl"
            else model._internal_to_natural(internal_params, compiled)
        )
        loss = -model.loglike(natural, data, compiled)
        loss.backward()
        closures["count"] += 1
        return loss

    sync_device(device)
    estimate_start = time.perf_counter()
    optimizer.step(closure)
    sync_device(device)
    estimate_s = time.perf_counter() - estimate_start

    final_internal = internal_params.detach().clone()
    final_natural = (
        final_internal
        if spec.model == "mnl"
        else model._internal_to_natural(final_internal, compiled)
    )
    sync_device(device)
    loglike_start = time.perf_counter()
    loglike = float(model.loglike(final_natural, data, compiled).detach().cpu())
    sync_device(device)
    loglike_s = time.perf_counter() - loglike_start

    optimizer_state = optimizer.state.get(internal_params, {})
    params = {
        name: float(value)
        for name, value in zip(compiled.free_names, final_natural.detach().cpu().numpy())
    }
    payload = {
        "available": True,
        "status": "Completed",
        "case": spec.case,
        "model": spec.model,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "n_obs": spec.n_obs,
        "n_alternatives": spec.n_alternatives,
        "n_variables": spec.n_variables,
        "n_rows": data.n_rows,
        "rho": spec.rho,
        "random_coefficients": len(case.random_names) if spec.model == "mixl" else 0,
        "n_draws": spec.n_draws,
        "max_iter": spec.max_iter,
        "n_parameters": len(compiled.free_names),
        "parameter_names": compiled.free_names,
        "setup_s": setup_s,
        "estimate_s": estimate_s,
        "loglike_s": loglike_s,
        "total_model_s": setup_s + estimate_s + loglike_s,
        "closure_evaluations": closures["count"],
        "optimizer_iterations": int(optimizer_state.get("n_iter", 0)),
        "loglike": loglike,
        "params": params,
        "peak_cuda_memory_mb": peak_memory_mb(device),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    return payload


def run_worker(args: argparse.Namespace) -> None:
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(args.cpu_threads)
        except RuntimeError:
            pass
    device = torch.device(args.worker_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"available": False, "status": "Unavailable", "message": "CUDA is not available."}))
        return

    spec = DeviceStressSpec(
        case=args.case,
        model=args.model,
        n_obs=args.n_obs,
        n_alternatives=args.n_alternatives,
        n_variables=args.n_variables,
        rho=args.rho,
        random_coefficients=args.random_coefficients,
        n_draws=args.n_draws,
        max_iter=args.max_iter,
    )
    build_start = time.perf_counter()
    meta = generated.GeneratedSpec(
        spec.case,
        spec.model,
        spec.n_obs,
        spec.n_alternatives,
        spec.n_variables,
        spec.rho,
        spec.random_coefficients,
    )
    mnl_case = generated.build_mnl_case(meta, args.seed)
    if spec.model == "mnl":
        case = mnl_case
        draws = None
    elif spec.model == "nl":
        case = build_device_nested_case(mnl_case, args.seed)
        draws = None
    elif spec.model == "mixl":
        case = generated.build_mixed_case(mnl_case, args.seed)
        draws = mixed_real.make_draws(
            spec.n_draws,
            args.seed + generated.stable_case_offset(spec.case),
            len(case.random_names),
        )
    else:
        raise ValueError(f"Unsupported model: {spec.model}")
    build_s = time.perf_counter() - build_start

    runs = [run_model_path(spec, case, draws, device) for _ in range(args.worker_repeats)]
    representative = sorted(runs, key=lambda result: float(result["total_model_s"]))[
        len(runs) // 2
    ].copy()
    representative["build_s"] = build_s
    representative["total_worker_s"] = build_s + float(representative["total_model_s"])
    representative["repetitions"] = args.worker_repeats
    representative["completed_repetitions"] = args.worker_repeats
    representative["runtime_repeats_s"] = [result["total_model_s"] for result in runs]
    representative["estimate_repeats_s"] = [result["estimate_s"] for result in runs]
    representative["closure_repeats"] = [result["closure_evaluations"] for result in runs]
    representative["iteration_repeats"] = [result["optimizer_iterations"] for result in runs]
    representative["peak_cuda_memory_repeats_mb"] = [result["peak_cuda_memory_mb"] for result in runs]
    print(json.dumps(representative, allow_nan=False))


def run_child(spec: DeviceStressSpec, args: argparse.Namespace, device: str, timeout_s: int) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-device",
        device,
        "--case",
        spec.case,
        "--model",
        spec.model,
        "--n-obs",
        str(spec.n_obs),
        "--n-alternatives",
        str(spec.n_alternatives),
        "--n-variables",
        str(spec.n_variables),
        "--rho",
        str(spec.rho),
        "--random-coefficients",
        str(spec.random_coefficients),
        "--n-draws",
        str(spec.n_draws),
        "--max-iter",
        str(spec.max_iter),
        "--seed",
        str(args.seed),
        "--worker-repeats",
        str(args.repeats),
    ]
    if args.cpu_threads:
        command.extend(["--cpu-threads", str(args.cpu_threads)])
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_s * args.repeats + 300,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": False,
            "status": "Timeout",
            "device": device,
            "timeout_s": timeout_s,
            "total_worker_s": time.perf_counter() - start,
            "message": f"Timed out after {timeout_s} seconds.",
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "status": "Failed",
            "device": device,
            "total_worker_s": time.perf_counter() - start,
            "message": (completed.stderr or completed.stdout).strip()[-4000:],
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "available": False,
            "status": "Failed",
            "device": device,
            "total_worker_s": time.perf_counter() - start,
            "message": f"Could not parse worker JSON: {exc}; stdout={completed.stdout[-2000:]} stderr={completed.stderr[-2000:]}",
        }
    payload["supervisor_wall_s"] = time.perf_counter() - start
    return payload


def run_repeated(
    spec: DeviceStressSpec,
    args: argparse.Namespace,
    device: str,
    timeout_s: int,
) -> dict:
    return run_child(spec, args, device, timeout_s)


def compare_cpu_gpu(results: dict[str, dict]) -> dict:
    cpu = results.get("cpu", {})
    gpu = results.get("cuda", {})
    comparison: dict[str, object] = {}
    if cpu.get("available") and gpu.get("available"):
        comparison["objective_diff"] = abs(float(cpu["loglike"]) - float(gpu["loglike"]))
        comparison["relative_objective_diff"] = comparison["objective_diff"] / max(
            1.0, abs(float(cpu["loglike"]))
        )
        cpu_params = cpu.get("params", {})
        gpu_params = gpu.get("params", {})
        common = sorted(set(cpu_params) & set(gpu_params))
        comparison["max_param_diff"] = max((abs(float(cpu_params[name]) - float(gpu_params[name])) for name in common), default=None)
        comparison["speedup_total_model"] = float(cpu["total_model_s"]) / float(gpu["total_model_s"])
        comparison["speedup_estimate"] = float(cpu["estimate_s"]) / float(gpu["estimate_s"])
        comparison["consistent"] = bool(
            comparison["objective_diff"]
            <= max(1e-4, 1e-8 * abs(float(gpu["loglike"])))
            and (comparison["max_param_diff"] is None or comparison["max_param_diff"] <= 1e-4)
        )
    elif cpu.get("status") == "Timeout" and gpu.get("available"):
        comparison["consistent"] = "CPU timeout; GPU completed"
        comparison["speedup_total_model"] = f">{cpu.get('timeout_s', 0) / float(gpu['total_model_s']):.1f}x"
    else:
        comparison["consistent"] = "Not evaluated"
    return comparison


def fmt_time(result: dict | None, key: str = "total_model_s") -> str:
    if not result:
        return "NA"
    if result.get("status") == "Timeout":
        return "Timeout"
    if not result.get("available"):
        return "Fail"
    value = result.get(key)
    return f"{float(value):.3f}" if isinstance(value, (int, float)) and np.isfinite(value) else "NA"


def render_markdown(rows: list[dict], profile: str) -> str:
    lines = [
        f"# TorchDCM CPU/GPU Device Stress ({profile})",
        "",
        "Rows use pure synthetic data with identical model specifications, initialization, and data on CPU and CUDA; MixL also uses identical antithetic normal draws. Data generation is excluded, and times are medians over repeated model setup, optimization, and final likelihood evaluation within each worker.",
        "",
        "| model | case | N | J | K | rho | RC | draws | CPU s | GPU s | speedup | GPU memory MB | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        cpu = row["devices"].get("cpu")
        gpu = row["devices"].get("cuda")
        comparison = row["comparison"]
        result = comparison.get("consistent")
        memory = gpu.get("peak_cuda_memory_mb") if gpu else None
        lines.append(
            "| {model} | {case} | {N} | {J} | {K} | {rho} | {rc} | {draws} | {cpu_total} | {gpu_total} | {speedup} | {memory} | {result} |".format(
                model=row["model"].upper() if row["model"] != "mixl" else "MixL",
                case=row["case"],
                N=row["n_obs"],
                J=row["n_alternatives"],
                K=row["n_variables"],
                rho=row["rho"],
                rc=row["random_coefficients"],
                draws=row["n_draws"],
                cpu_total=fmt_time(cpu),
                gpu_total=fmt_time(gpu),
                speedup=(
                    f"{float(comparison['speedup_total_model']):.1f}x"
                    if isinstance(comparison.get("speedup_total_model"), (int, float))
                    else comparison.get("speedup_total_model", "NA")
                ),
                memory=f"{float(memory):.0f}" if isinstance(memory, (int, float)) else "NA",
                result=result,
            )
        )
    lines.extend(["", "## Diagnostics", ""])
    for row in rows:
        lines.append(f"- `{row['case']}` comparison: {row['comparison']}")
        for device, result in row["devices"].items():
            lines.append(
                f"  - {device}: status={result.get('status')}, total_model_s={fmt_time(result)}, "
                f"estimate_s={fmt_time(result, 'estimate_s')}, loglike={result.get('loglike')}"
            )
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict], profile: str) -> tuple[Path, Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / f"torch_device_stress_{profile}.json"
    md_path = RESULTS / f"torch_device_stress_{profile}.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(rows, profile), encoding="utf-8")
    return json_path, md_path


def run_supervisor(args: argparse.Namespace) -> None:
    output_profile = args.output_profile or args.profile
    rows: list[dict] = []
    for spec in profile_specs(args.profile):
        if args.case and spec.case != args.case:
            continue
        print(
            f"[device-stress] model={spec.model} case={spec.case} N={spec.n_obs} J={spec.n_alternatives} K={spec.n_variables} "
            f"RC={spec.random_coefficients} draws={spec.n_draws}",
            flush=True,
        )
        devices = {
            "cpu": run_repeated(spec, args, "cpu", args.cpu_timeout),
            "cuda": run_repeated(spec, args, "cuda", args.gpu_timeout),
        }
        row = {
            "case": spec.case,
            "model": spec.model,
            "n_obs": spec.n_obs,
            "n_alternatives": spec.n_alternatives,
            "n_variables": spec.n_variables,
            "rho": spec.rho,
            "random_coefficients": spec.random_coefficients,
            "n_draws": spec.n_draws,
            "max_iter": spec.max_iter,
            "devices": devices,
            "comparison": compare_cpu_gpu(devices),
            "timeout_policy": {"cpu_timeout_s": args.cpu_timeout, "gpu_timeout_s": args.gpu_timeout},
            "repetitions": args.repeats,
        }
        rows.append(row)
        write_outputs(rows, output_profile)
        print(
            f"[device-stress] {spec.case}: cpu={fmt_time(devices['cpu'])} "
            f"gpu={fmt_time(devices['cuda'])} result={row['comparison'].get('consistent')}",
            flush=True,
        )
    json_path, md_path = write_outputs(rows, output_profile)
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "calibration", "battery", "stress"], default="stress")
    parser.add_argument("--output-profile", help="Optional output filename suffix; defaults to --profile.")
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--cpu-timeout", type=int, default=300)
    parser.add_argument("--gpu-timeout", type=int, default=900)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker-repeats", type=int, default=1)
    parser.add_argument("--worker-device", choices=["cpu", "cuda"])
    parser.add_argument("--model", choices=["mnl", "nl", "mixl"])
    parser.add_argument("--n-obs", type=int)
    parser.add_argument("--n-alternatives", type=int)
    parser.add_argument("--n-variables", type=int)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--random-coefficients", type=int)
    parser.add_argument("--n-draws", type=int)
    parser.add_argument("--max-iter", type=int)
    if "--worker-device" in sys.argv:
        args = parser.parse_args()
        required = ["case", "model", "n_obs", "n_alternatives", "n_variables", "rho", "random_coefficients", "n_draws", "max_iter"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(f"Missing worker arguments: {missing}")
        run_worker(args)
        return
    args = parser.parse_args()
    run_supervisor(args)


if __name__ == "__main__":
    main()
