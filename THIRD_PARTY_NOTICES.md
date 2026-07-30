# Third-party notices

The repository-level [MIT License](LICENSE) applies to the TorchDCM source
snapshot and to the experiment drivers, scripts, tests, documentation, and
generated result files authored for this project. It does not relicense
third-party software or data.

## External software

PyTorch, NumPy, pandas, SciPy, Biogeme, Apollo, `mlogit`, `gmnl`, `xlogit`,
and `jsonlite` are external dependencies. Their source code is not vendored in
this repository. Each dependency remains subject to its upstream license and
is identified by version in the main [README](README.md).

## Included datasets

The files under `data/small/` were obtained from the Biogeme data collection
or from data objects distributed with the R package `mlogit`. Their upstream
locations are recorded in `data/dataset_index.csv` and in each dataset's
`metadata.json`.

- The Biogeme data page makes the listed choice datasets available for
  research and education. The datasets remain attributed to their upstream
  collectors and distributors.
- The `mlogit` package declares `GPL (>= 2)`. The CSV files in this repository
  were generated from its distributed data objects. Users should consult the
  upstream package documentation for the original studies and any
  dataset-specific terms.

These third-party datasets are included only to reproduce the reported
experiments. They are not covered by this repository's MIT License. Users who
redistribute or reuse them should consult the upstream terms and citations.
Large LPMC and NHTS inputs are not redistributed and are downloaded from their
official sources by the preparation scripts.
