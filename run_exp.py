"""Select and run a paper experiment from the repository root."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Experiment:
    description: str
    command: tuple[str, ...]
    archived_result: str


EXPERIMENTS = {
    "synthetic-mnl": Experiment(
        "Synthetic controlled MNL estimation",
        (
            "experiments/compare_generated_choice_battery.py",
            "--profile",
            "paper",
            "--models",
            "mnl",
            "--backend-timeout",
            "300",
            "--output-profile",
            "synthetic_mnl_reproduction",
        ),
        "results/synthetic_mnl_single_core.json",
    ),
    "synthetic-nl": Experiment(
        "Synthetic controlled NL estimation",
        (
            "experiments/compare_generated_choice_battery.py",
            "--profile",
            "paper",
            "--models",
            "nl",
            "--backend-timeout",
            "300",
            "--output-profile",
            "synthetic_nl_reproduction",
        ),
        "results/generated_choice_battery_table4_nl.json",
    ),
    "synthetic-mixl": Experiment(
        "Synthetic controlled MixL estimation",
        (
            "experiments/compare_generated_choice_battery.py",
            "--profile",
            "paper",
            "--models",
            "mixl",
            "--backend-timeout",
            "300",
            "--output-profile",
            "synthetic_mixl_reproduction",
        ),
        "results/generated_choice_battery_table4_mixl.json",
    ),
    "real-mnl": Experiment(
        "Real-data MNL estimation",
        (
            "experiments/run_solver_attempt_matrix.py",
            "--profile",
            "mnl_reproduction",
        ),
        "results/solver_attempt_matrix_mnl_single_core.json",
    ),
    "real-nl": Experiment(
        "Real-data NL estimation",
        (
            "experiments/run_real_nested_logit_isolated.py",
            "--json-output",
            "results/nested_real_battery_reproduction.json",
            "--md-output",
            "results/nested_real_battery_reproduction.md",
        ),
        "results/nested_real_battery_single_core.json",
    ),
    "real-mixl": Experiment(
        "Real-data MixL estimation",
        (
            "experiments/run_real_mixed_logit_isolated.py",
            "--json-output",
            "results/mixed_real_battery_reproduction.json",
            "--md-output",
            "results/mixed_real_battery_reproduction.md",
        ),
        "results/mixed_real_battery_apollo.json",
    ),
    "cpu-cuda": Experiment(
        "TorchDCM CPU--CUDA scaling",
        (
            "experiments/compare_torch_device_stress.py",
            "--profile",
            "battery",
            "--repeats",
            "3",
            "--output-profile",
            "device_reproduction",
        ),
        "results/torch_device_stress_battery.json",
    ),
    "ordered-synthetic-logit": Experiment(
        "Synthetic ordered-logit estimation",
        (
            "experiments/compare_synthetic_ordered_probit.py",
            "--kind",
            "logit",
            "--output",
            "results/ordered_logit_synthetic_reproduction.json",
        ),
        "results/ordered_logit_synthetic_threeway_single_core.json",
    ),
    "ordered-synthetic-probit": Experiment(
        "Synthetic ordered-probit estimation",
        (
            "experiments/compare_synthetic_ordered_probit.py",
            "--kind",
            "probit",
            "--output",
            "results/ordered_probit_synthetic_reproduction.json",
        ),
        "results/ordered_probit_synthetic_threeway_single_core.json",
    ),
    "ordered-real-logit": Experiment(
        "Real-data ordered-logit estimation",
        (
            "experiments/run_real_ordered_probit_battery.py",
            "--kind",
            "logit",
            "--output",
            "results/ordered_logit_real_reproduction.json",
        ),
        "results/ordered_logit_real_threeway_single_core.json",
    ),
    "ordered-real-probit": Experiment(
        "Real-data ordered-probit estimation",
        (
            "experiments/run_real_ordered_probit_battery.py",
            "--kind",
            "probit",
            "--output",
            "results/ordered_probit_real_reproduction.json",
        ),
        "results/ordered_probit_real_threeway_single_core.json",
    ),
    "advanced": Experiment(
        "Latent-class, hybrid-choice, and panel estimation",
        (
            "experiments/run_advanced_full_suite.py",
            "--output",
            "results/advanced_full_estimation_reproduction.json",
            "--log-dir",
            "results/advanced_full_logs_reproduction",
        ),
        "results/advanced_full_estimation.json",
    ),
}


def print_experiments() -> None:
    """Print stable experiment identifiers and their archived outputs."""
    width = max(len(name) for name in EXPERIMENTS)
    for name, experiment in EXPERIMENTS.items():
        print(
            f"{name:<{width}}  {experiment.description}\n"
            f"{'':<{width}}  archived: {experiment.archived_result}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one TorchDCM paper experiment from the repository root."
    )
    parser.add_argument("experiment", nargs="?", choices=EXPERIMENTS)
    parser.add_argument(
        "--list", action="store_true", help="List experiment identifiers and outputs."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved command only."
    )
    args, extra = parser.parse_known_args()

    if args.list or args.experiment is None:
        print_experiments()
        return 0

    experiment = EXPERIMENTS[args.experiment]
    command = [sys.executable, *experiment.command, *extra]
    print(f"Running: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
