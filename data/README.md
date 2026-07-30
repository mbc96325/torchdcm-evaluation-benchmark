# Benchmark datasets

This directory contains only datasets used in the paper.

- `small/` stores the 16 GitHub-sized Biogeme and R `mlogit` datasets.
- `large/` documents the LPMC London and NHTS 2022 inputs.
- `dataset_index.csv` records the source and storage rule for all 18 empirical
  datasets reported in the manuscript.

Each small-dataset directory contains a canonical `data.csv` and source
metadata. LPMC is downloaded from the official Biogeme data page and processed
with:

```bash
python scripts/process_lpmc_london.py
```

The NHTS runner downloads the official 2022 CSV release and constructs the
paper's mode-choice sample when its local cache is absent.
