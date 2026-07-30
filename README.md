# TorchDCM Evaluation Benchmark

This repository reproduces the validation and runtime experiments in the
TorchDCM software paper. The installable package is maintained in the separate
[TorchDCM repository](https://github.com/mbc96325/torchdcm) and released on
[PyPI](https://pypi.org/project/torchdcm/).

The repository intentionally contains only paper-facing experiment code,
aligned external-software wrappers, benchmark datasets, and the committed
outputs used to prepare the manuscript tables.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[bench]"
```

The Python environment provides TorchDCM 0.1.1, PyTorch, Biogeme, SciPy, and
`xlogit`. The R comparisons additionally require Apollo, `mlogit`, `gmnl`, and
`jsonlite`:

```r
install.packages(c("apollo", "mlogit", "gmnl", "jsonlite"))
```

The committed results were generated with TorchDCM 0.1.1, PyTorch 2.12.1,
Biogeme 3.3.3, `xlogit` 0.2.7, Apollo 0.3.8, `mlogit` 2.0.0, and `gmnl`
1.1.3.2.

Small public datasets are committed under `datasets/small`. Before running the
LPMC cases, materialize the official source data:

```bash
python scripts/process_lpmc_london.py
```

The NHTS runner downloads and processes the official 2022 trip file when its
local cache is absent.

## Reproducing the paper results

Each cross-estimator runner enforces the paper's single-logical-CPU policy.
Commands below write new outputs without overwriting the committed Office
results.

| Paper experiment | Reproduction command | Committed result |
| --- | --- | --- |
| Synthetic MNL | `python benchmarks/compare_generated_choice_battery.py --profile paper --models mnl --backend-timeout 300 --output-profile synthetic_mnl_reproduction` | [`synthetic_mnl_single_core_office.json`](generated/synthetic_mnl_single_core_office.json) |
| Synthetic NL | `python benchmarks/compare_generated_choice_battery.py --profile paper --models nl --backend-timeout 300 --output-profile synthetic_nl_reproduction` | [`generated_choice_battery_table4_nl_office.json`](generated/generated_choice_battery_table4_nl_office.json) |
| Synthetic MixL | `python benchmarks/compare_generated_choice_battery.py --profile paper --models mixl --backend-timeout 300 --output-profile synthetic_mixl_reproduction` | [`generated_choice_battery_table4_mixl_office.json`](generated/generated_choice_battery_table4_mixl_office.json) |
| Real-data MNL | `python benchmarks/run_solver_attempt_matrix.py --profile mnl_reproduction` | [`solver_attempt_matrix_mnl_single_core_office.json`](generated/solver_attempt_matrix_mnl_single_core_office.json) |
| Real-data NL | `python benchmarks/run_real_nested_logit_isolated.py --json-output generated/nested_real_battery_reproduction.json --md-output generated/nested_real_battery_reproduction.md` | [`nested_real_battery_single_core_office.json`](generated/nested_real_battery_single_core_office.json) |
| Real-data MixL | `python benchmarks/run_real_mixed_logit_isolated.py --json-output generated/mixed_real_battery_reproduction.json --md-output generated/mixed_real_battery_reproduction.md` | [`mixed_real_battery_apollo_office.json`](generated/mixed_real_battery_apollo_office.json) |
| CPU--CUDA scaling | `python benchmarks/compare_torch_device_stress.py --profile battery --repeats 3 --output-profile device_reproduction` | [`torch_device_stress_battery.json`](generated/torch_device_stress_battery.json) |
| Synthetic ordered logit | `python benchmarks/compare_synthetic_ordered_probit.py --kind logit --output generated/ordered_logit_synthetic_reproduction.json` | [`ordered_logit_synthetic_threeway_single_core_office.json`](generated/ordered_logit_synthetic_threeway_single_core_office.json) |
| Synthetic ordered probit | `python benchmarks/compare_synthetic_ordered_probit.py --kind probit --output generated/ordered_probit_synthetic_reproduction.json` | [`ordered_probit_synthetic_threeway_single_core_office.json`](generated/ordered_probit_synthetic_threeway_single_core_office.json) |
| Real ordered logit | `python benchmarks/run_real_ordered_probit_battery.py --kind logit --output generated/ordered_logit_real_reproduction.json` | [`ordered_logit_real_threeway_single_core_office.json`](generated/ordered_logit_real_threeway_single_core_office.json) |
| Real ordered probit | `python benchmarks/run_real_ordered_probit_battery.py --kind probit --output generated/ordered_probit_real_reproduction.json` | [`ordered_probit_real_threeway_single_core_office.json`](generated/ordered_probit_real_threeway_single_core_office.json) |
| Latent class, hybrid choice, and panel | `python benchmarks/run_advanced_full_suite.py --output generated/advanced_full_estimation_reproduction.json` | [`advanced_full_estimation_office.json`](generated/advanced_full_estimation_office.json) |

Run `python benchmarks/summarize_ordered_results.py` to render the ordered-model
rows used in the electronic companion. Runtime values will vary by machine.
Case definitions, final likelihoods, parameter diagnostics, and comparison
status are retained in the JSON outputs.

## Repository layout

| Path | Purpose |
| --- | --- |
| `benchmarks/` | Paper experiment runners and external-software wrappers. |
| `datasets/` | Included datasets, provenance, and large-data preparation notes. |
| `generated/` | The authoritative outputs used by the paper. |
| `scripts/` | LPMC preparation utility. |
| `tests/` | Numerical parity checks for benchmark infrastructure. |
| `BENCHMARK_SYSTEM.md` | Runtime, alignment, and consistency conventions. |
