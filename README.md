# TorchDCM Evaluation Benchmark

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository is the standalone computational archive for the TorchDCM
software paper, *TorchDCM: A Unified PyTorch-Native Package for Discrete Choice
Modeling*. The paper is prepared for the *INFORMS Journal on Computing*
software track.

This archive contains the TorchDCM source code used in the study, experiment
drivers for seven estimation packages, all input datasets, the intermediate
solver outputs retained by the benchmark, and the machine-readable results
used to prepare the paper's tables.

The `torchdcm/` directory is a clean snapshot of TorchDCM 0.1.2 at Git commit
`86bd009795b86f4589403c34c988161d02eb94cc`. The snapshot is installed directly
from this repository. Reproducing the paper therefore does not require a
second Git checkout or a TorchDCM download from PyPI. Later package development
continues in the [TorchDCM repository](https://github.com/mbc96325/torchdcm).
See [TORCHDCM_SNAPSHOT.md](TORCHDCM_SNAPSHOT.md) for provenance.

## Citation

The journal DOI and computational-archive DOI have not yet been assigned.
Until they are available, cite the manuscript and this fixed computational
archive as follows:

```bibtex
@misc{mo2026torchdcm,
  author = {Mo, Baichuan and You, Zhengzhong Ricky and Chen, Xiqun Michael and Li, Ruimin},
  title = {{TorchDCM: A Unified PyTorch-Native Package for Discrete Choice Modeling}},
  year = {2026},
  note = {Software paper manuscript and computational archive},
  url = {https://github.com/mbc96325/torchdcm-evaluation-benchmark}
}
```

For an exact replication record, also report the Git commit used. The active
TorchDCM package and this archived evaluation repository serve different
purposes and should be linked separately.

## Contents

1. [Scope](#scope)
2. [Repository layout](#repository-layout)
3. [System requirements](#system-requirements)
4. [Quick start](#quick-start)
5. [Reproduction workflow](#reproduction-workflow)
6. [Experiment-to-output mapping](#experiment-to-output-mapping)
7. [Archived data](#archived-data)
8. [Generating and checking paper outputs](#generating-and-checking-paper-outputs)
9. [Timing and numerical protocol](#timing-and-numerical-protocol)
10. [Approximate wall-clock times](#approximate-wall-clock-times)
11. [Tests](#tests)
12. [Cross-machine reproducibility](#cross-machine-reproducibility)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)
15. [Ongoing development](#ongoing-development)
16. [Support](#support)

## Scope

### What is reproduced

The repository reproduces the numerical experiments reported in the paper:

- synthetic MNL, NL, and MixL full-estimation experiments;
- real-data MNL, NL, and MixL comparisons;
- TorchDCM CPU--CUDA experiments;
- ordered-logit and ordered-probit validation; and
- latent-class, hybrid-choice, and panel validation.

These experiments produce the results behind Tables 3--9 in the main paper
and Tables EC.1--EC.11 in the electronic companion. All 18 empirical datasets,
including the original LPMC files and the NHTS public-use CSV archive, are
committed to ordinary Git. No Git LFS checkout or external data download is
required after cloning.

### What is not reproduced directly

- Table 1 summarizes documented package capabilities rather than an executed
  numerical experiment.
- Table 2 reports dataset dimensions from the committed data and metadata.
- Figure 1, the code listings, and the estimation-report illustration explain
  the package architecture and public interface rather than benchmark outputs.
- The formatted manuscript and electronic companion are maintained in the
  separate manuscript repository. This archive retains the JSON evidence from
  which their numerical tables were prepared.

## Repository layout

```text
.
├── torchdcm/                 # TorchDCM 0.1.2 source snapshot
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
| TorchDCM source snapshot | 0.1.2 |
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

## Generating and checking paper outputs

Each root command writes a new result with a `_reproduction` suffix and leaves
the archived paper result unchanged. Compare the new JSON with the file named
in the mapping table. The principal checks are:

1. the same case dimensions and model specification;
2. successful completion or the same documented timeout status;
3. final log likelihood within the rule in
   [BENCHMARK_SYSTEM.md](BENCHMARK_SYSTEM.md); and
4. similar runtime ordering, allowing for hardware and software differences.

The top-level JSON files in `results/` are the direct sources for the numerical
tables. `results/intermediate/` contains the subordinate package outputs and
logs used to assemble them. The manuscript repository applies the journal's
table formatting to these values and is not required to rerun estimation.

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

## Approximate wall-clock times

The following planning values summarize the archived run on the paper
hardware. They include sequential estimator work and recorded timeouts, rounded
to practical intervals. Environment setup and scheduler overhead can add time.

| Identifier | Approximate time |
| --- | ---: |
| `synthetic-mnl` | 60 minutes |
| `synthetic-nl` | 45 minutes |
| `synthetic-mixl` | 2 hours |
| `real-mnl` | 10 minutes |
| `real-nl` | 5 minutes |
| `real-mixl` | 25 minutes |
| `cpu-cuda` | 30 minutes |
| each synthetic ordered experiment | 3 minutes |
| each real-data ordered experiment | 7 minutes |
| `advanced` | 15 minutes |

A complete sequential rerun should therefore reserve roughly six hours on
comparable hardware. Users interested only in checking a reported table can run
the corresponding identifier or inspect the committed result without rerunning
the remaining experiments.

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

## Troubleshooting

- **An R estimator is unavailable.** Install Apollo, `mlogit`, `gmnl`, and
  `jsonlite` with the command in [Quick start](#quick-start), then rerun the
  affected identifier.
- **CUDA is not detected.** Install a CUDA-enabled PyTorch build compatible
  with the local driver. All non-device experiments can still run on CPU.
- **A runtime differs from the paper.** Confirm single-thread execution and
  avoid concurrent heavy jobs. Compare final log likelihoods before interpreting
  timing differences.
- **A stress case reaches 300 seconds.** This is an expected benchmark outcome,
  not an archive failure. The JSON records `Timeout` explicitly.
- **A result file already exists.** Reproduction commands use a
  `_reproduction` suffix so that the committed paper evidence is not
  overwritten.

## License

The TorchDCM source snapshot, original benchmark software, documentation, and
generated results are released under the [MIT License](LICENSE), with
copyright held by Baichuan Mo. Third-party datasets and external dependencies
retain their upstream terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Ongoing development

This repository is a fixed paper-evaluation archive. New TorchDCM features,
examples, releases, and bug fixes are developed in the
[main TorchDCM repository](https://github.com/mbc96325/torchdcm). Benchmark
changes needed to reproduce the archived manuscript should be made here and
should preserve the source snapshot and authoritative results.

## Support

Use the issue trackers for the corresponding scope:

- [benchmark and reproduction issues](https://github.com/mbc96325/torchdcm-evaluation-benchmark/issues);
- [TorchDCM package issues](https://github.com/mbc96325/torchdcm/issues).
