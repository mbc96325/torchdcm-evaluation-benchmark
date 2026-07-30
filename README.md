# TorchDCM Evaluation Benchmark

This repository contains the reproducible benchmark and validation system used
for the TorchDCM software paper. The installable package is maintained in the
separate [TorchDCM repository](https://github.com/mbc96325/torchdcm) and is
available from [PyPI](https://pypi.org/project/torchdcm/).

## Repository layout

| Path | Purpose |
| --- | --- |
| `benchmarks/` | Synthetic and real-data comparisons against external estimators. |
| `datasets/` | Dataset registry, provenance, canonical data, and download/export tools. |
| `generated/` | Committed benchmark outputs used by the paper. |
| `scripts/` | Dataset preparation and result-rendering utilities. |
| `tests/` | Tests for benchmark loaders, wrappers, and result processing. |
| `BENCHMARK_SYSTEM.md` | Benchmark conventions and extension guide. |

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torchdcm
python -m pip install -e ".[bench]"
```

Biogeme is installed through the Python benchmark dependencies. Apollo,
`mlogit`, and `gmnl` require R; `xlogit` uses the Python benchmark environment.

## Benchmark matrix

| Benchmark | Runner | Paper-facing output |
| --- | --- | --- |
| Controlled synthetic MNL, NL, and MixL | [`compare_generated_choice_battery.py`](benchmarks/compare_generated_choice_battery.py) | [MNL](generated/generated_choice_battery_controlled_office.md), [NL](generated/generated_choice_battery_table4_nl_office.md), [MixL](generated/generated_choice_battery_table4_mixl_office.md) |
| Real-data MNL | [`run_solver_attempt_matrix.py`](benchmarks/run_solver_attempt_matrix.py) | [`solver_attempt_matrix_mnl_single_core_office.md`](generated/solver_attempt_matrix_mnl_single_core_office.md) |
| Real-data NL | [`compare_real_nested_logit_battery.py`](benchmarks/compare_real_nested_logit_battery.py) | [`nested_real_battery_single_core_office.md`](generated/nested_real_battery_single_core_office.md) |
| Real-data MixL | [`compare_real_mixed_logit_battery.py`](benchmarks/compare_real_mixed_logit_battery.py) | [`mixed_real_battery_apollo_office.md`](generated/mixed_real_battery_apollo_office.md) |
| CPU--GPU scaling | [`compare_torch_device_stress.py`](benchmarks/compare_torch_device_stress.py) | [`torch_device_stress_battery.md`](generated/torch_device_stress_battery.md) |
| Ordered logit and probit | [`compare_ordered_estimators.py`](benchmarks/compare_ordered_estimators.py) | [Ordered logit](generated/ordered_logit_single_core_office.json), [ordered probit](generated/ordered_probit_single_core_office.json) |
| Synthetic ordered probit | [`compare_synthetic_ordered_probit.py`](benchmarks/compare_synthetic_ordered_probit.py) | [`ordered_probit_synthetic_single_core_office.json`](generated/ordered_probit_synthetic_single_core_office.json) |
| Real-data ordered probit | [`run_real_ordered_probit_battery.py`](benchmarks/run_real_ordered_probit_battery.py) | [`ordered_probit_real_battery_single_core_office.json`](generated/ordered_probit_real_battery_single_core_office.json) |
| Latent class, hybrid choice, and panel full estimation | [`run_advanced_full_suite.py`](benchmarks/run_advanced_full_suite.py) | [`advanced_full_estimation_office.json`](generated/advanced_full_estimation_office.json) |

See the [benchmark guide](BENCHMARK_SYSTEM.md), [dataset catalog](datasets/dataset_index.csv),
and [validation dataset notes](datasets/VALIDATION.md) for setup and provenance.
