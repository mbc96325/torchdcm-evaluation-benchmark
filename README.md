# TorchDCM Evaluation Benchmark

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository is the standalone computational archive for the TorchDCM
software paper. It contains the TorchDCM source code used in the study,
experiment drivers for seven estimation packages, all input datasets, the
intermediate solver outputs retained by the benchmark, and the machine-readable
results used to prepare the paper's tables.

The `torchdcm/` directory is a clean snapshot of TorchDCM 0.1.1 at Git commit
`b34ab6924523017aca39f5529c940c2cdd817bde`. The snapshot is installed directly
from this repository. Reproducing the paper therefore does not require a
second Git checkout or a TorchDCM download from PyPI. Later package development
continues in the [TorchDCM repository](https://github.com/mbc96325/torchdcm).
See [TORCHDCM_SNAPSHOT.md](TORCHDCM_SNAPSHOT.md) for provenance.

## Scope

The repository reproduces the paper's:

- synthetic MNL, NL, and MixL full-estimation experiments;
- real-data MNL, NL, and MixL comparisons;
- TorchDCM CPU--CUDA experiments;
- ordered-logit and ordered-probit validation; and
- latent-class, hybrid-choice, and panel validation.

The manuscript source is intentionally not included. All 18 empirical datasets
used by the paper, including the original LPMC files and the NHTS public-use
CSV archive, are committed to ordinary Git. No Git LFS checkout or external
data download is required after cloning.

## Repository layout

```text
.
├── torchdcm/                 # TorchDCM 0.1.1 source snapshot
├── experiments/              # Python experiment drivers
│   ├── apollo/R/             # Apollo adapters
│   └── mlogit/R/             # mlogit and gmnl adapters
├── data/
│   ├── small/                # Included public benchmark datasets
│   ├── raw/                  # Archived LPMC and NHTS source files
│   ├── large/                # Optional processed-data instructions
│   └── dataset_index.csv     # Dataset-to-storage index
├── results/
│   ├── intermediate/         # Retained subordinate outputs and solver logs
│   └── *.json                # Authoritative paper-table results
├── scripts/                  # Large-data preparation utilities
├── tests/                    # Artifact and numerical parity checks
├── run_exp.py                # Root experiment selector
├── requirements.txt          # Tested Python versions
├── pyproject.toml            # Standalone local installation
├── LICENSE                   # MIT License for original software
├── AUTHORS                   # Contribution authors
├── THIRD_PARTY_NOTICES.md    # External software and data attribution
└── BENCHMARK_SYSTEM.md       # Alignment and timing protocol
```

## System requirements

Python 3.10 or later is required. The committed results were produced with:

| Component | Version |
| --- | --- |
| TorchDCM source snapshot | 0.1.1 |
| PyTorch | 2.12.1+cu130 |
| NumPy | 2.4.6 |
| pandas | 2.3.3 |
| SciPy | 1.18.0 |
| Biogeme | 3.3.3 |
| xlogit | 0.2.7 |
| Apollo | 0.3.8 |
| mlogit | 2.0.0 |
| gmnl | 1.1.3.2 |
| jsonlite | 2.0.0 |

Python-only checks run on Linux, macOS, or Windows. The complete
cross-software replication also requires an R installation and the four R
packages listed above. CUDA experiments require a CUDA-capable NVIDIA GPU and
a compatible PyTorch build.

The paper's CPU experiments ran on an AMD Ryzen 9 9950X3D. Each estimator and
its child processes used one logical CPU. Device experiments used one logical
CPU or an NVIDIA GeForce RTX 5090 with 32 GB of memory.

## Quick start

Clone the repository and install its local TorchDCM snapshot together with the
benchmark dependencies:

```bash
git clone https://github.com/mbc96325/torchdcm-evaluation-benchmark.git
cd torchdcm-evaluation-benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Windows, activate the environment with
`.venv\Scripts\activate`. The exact PyTorch command may differ by CPU/CUDA
platform. If so, install the appropriate PyTorch build first, then install the
remaining dependencies and this repository.

Install the external R estimators when their comparison rows are needed:

```r
install.packages(c("apollo", "mlogit", "gmnl", "jsonlite"))
```

Confirm that the vendored source and archived artifacts are available:

```bash
python -c "import torchdcm; print(torchdcm.__version__, torchdcm.__file__)"
pytest tests/test_paper_artifacts.py
python run_exp.py --list
```

The printed package path should point into this repository.

## Reproduction workflow

Run an experiment by its stable identifier:

```bash
python run_exp.py synthetic-mnl
```

Use `python run_exp.py --list` to display all identifiers and corresponding
archived outputs. Use `--dry-run` to inspect a resolved command without
starting estimation:

```bash
python run_exp.py real-mixl --dry-run
```

Arguments not consumed by `run_exp.py` are passed to the selected experiment
driver. Direct execution of the scripts in `experiments/` remains available
for custom profiles.

The standard workflow is:

1. install Python and, where needed, R dependencies;
2. select an experiment with `run_exp.py`;
3. retain the newly generated `*_reproduction` output; and
4. compare it with the corresponding archived JSON file.

Runtime values depend on hardware and software versions. Final log likelihoods,
case dimensions, backend status, and consistency diagnostics are stored in
each JSON output.

## Experiment-to-output mapping

| Identifier | Paper experiment | Paper table or figure | Archived result |
| --- | --- | --- | --- |
| `synthetic-mnl` | Synthetic controlled MNL | Table 3 | [`synthetic_mnl_single_core.json`](results/synthetic_mnl_single_core.json) |
| `synthetic-nl` | Synthetic controlled NL | Table 4 | [`generated_choice_battery_table4_nl.json`](results/generated_choice_battery_table4_nl.json) |
| `synthetic-mixl` | Synthetic controlled MixL | Table 5 | [`generated_choice_battery_table4_mixl.json`](results/generated_choice_battery_table4_mixl.json) |
| `real-mnl` | Real-data MNL | Table 7 | [`solver_attempt_matrix_mnl_single_core.json`](results/solver_attempt_matrix_mnl_single_core.json) |
| `real-nl` | Real-data NL | Table 8 | [`nested_real_battery_single_core.json`](results/nested_real_battery_single_core.json) |
| `real-mixl` | Real-data MixL | Table 9 | [`mixed_real_battery_apollo.json`](results/mixed_real_battery_apollo.json) |
| `cpu-cuda` | TorchDCM CPU--CUDA scaling | Table 6 | [`torch_device_stress_battery.json`](results/torch_device_stress_battery.json) |
| `ordered-synthetic-logit` | Synthetic ordered logit | Table EC.1 | [`ordered_logit_synthetic_threeway_single_core.json`](results/ordered_logit_synthetic_threeway_single_core.json) |
| `ordered-synthetic-probit` | Synthetic ordered probit | Table EC.1 | [`ordered_probit_synthetic_threeway_single_core.json`](results/ordered_probit_synthetic_threeway_single_core.json) |
| `ordered-real-logit` | Real-data ordered logit | Tables EC.2--EC.7 | [`ordered_logit_real_threeway_single_core.json`](results/ordered_logit_real_threeway_single_core.json) |
| `ordered-real-probit` | Real-data ordered probit | Tables EC.2--EC.7 | [`ordered_probit_real_threeway_single_core.json`](results/ordered_probit_real_threeway_single_core.json) |
| `advanced` | Latent class, hybrid choice, and panel | Tables EC.9--EC.11 | [`advanced_full_estimation.json`](results/advanced_full_estimation.json) |

Run the ordered-model summarizer after reproducing those cases:

```bash
python experiments/summarize_ordered_results.py
```

## Archived data

All paper inputs are versioned in this repository. Sixteen datasets are under
`data/small`. The LPMC source CSV and original DAT file and the NHTS 2022
public-use CSV archive are under `data/raw`. Their dimensions, checksums, and
upstream sources are recorded in `data/dataset_index.csv` and the accompanying
metadata.

The benchmark runners read these archived inputs directly. The following
optional command creates long- and wide-format LPMC derivatives for inspection:

```bash
python scripts/process_lpmc_london.py
```

It does not contact the network. NHTS rows are read directly from the committed
`data/raw/nhts_2022/csv.zip`.

## Timing and numerical protocol

All cross-estimator CPU runners enforce one logical CPU by controlling process
affinity and the OpenMP, BLAS, NumExpr, and PyTorch thread counts. Apollo uses
`nCores=1`. Reported runtime covers full parameter estimation and covariance
construction. It excludes data loading or generation, aligned-design
construction, process startup, and file input/output. Stress workers have a
300-second limit.

Within a case, estimators receive aligned data, utility specifications,
parameter scales, and starting values. Simulated models use common draws when
the external interface accepts them. The complete rules for runtime scope,
failed runs, and final-log-likelihood consistency are in
[BENCHMARK_SYSTEM.md](BENCHMARK_SYSTEM.md).

## Tests

Run the lightweight archive checks:

```bash
pytest tests/test_paper_artifacts.py
```

Run the Biogeme numerical parity test when Biogeme is installed:

```bash
pytest tests/test_biogeme_parity.py
```

The complete experiment suite is intentionally not a unit-test target because
the stress and simulation-based cases can require many hours.

## Cross-machine reproducibility

Numerical agreement should be assessed with the final-log-likelihood rule in
`BENCHMARK_SYSTEM.md`, not by expecting byte-identical JSON files. Runtime
comparisons should use the same thread policy and avoid concurrent heavy jobs.
CPU model, GPU model, BLAS implementation, package versions, and compiler
state can change wall-clock time.

## License

The TorchDCM source snapshot, original benchmark software, documentation, and
generated results are released under the [MIT License](LICENSE), with
copyright held by Baichuan Mo. Third-party datasets and external dependencies
retain their upstream terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Support

Use the issue trackers for the corresponding scope:

- [benchmark and reproduction issues](https://github.com/mbc96325/torchdcm-evaluation-benchmark/issues);
- [TorchDCM package issues](https://github.com/mbc96325/torchdcm/issues).
