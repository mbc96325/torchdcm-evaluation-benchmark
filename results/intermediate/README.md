# Intermediate benchmark evidence

This directory retains paper-producing outputs that precede the final JSON
tables in the parent directory.

- `mnl/` contains the dataset-level MNL outputs aggregated by
  `solver_attempt_matrix_mnl_single_core.json`.
- `advanced_torchdcm_apollo.json` is the TorchDCM/Apollo base run.
- `advanced_swissmetro_3500.json` supplies the Swissmetro 3,500 replacement
  case used by the final EC table.
- `advanced_full_logs/` contains the corresponding case-level Biogeme logs,
  including successful runs, failures, and time-limit records. These inputs
  were merged to create `advanced_full_estimation.json`.

Smoke tests, exploratory diagnostics, and superseded runs are intentionally
excluded because they do not contribute to a reported paper table.
