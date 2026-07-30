# Benchmark datasets

This directory contains all datasets used in the paper.

- `small/` stores 16 Biogeme and R `mlogit` datasets.
- `raw/` stores the original LPMC files and the NHTS 2022 public-use archive.
- `large/` documents optional processed derivatives.
- `dataset_index.csv` records the source and storage rule for all 18 empirical
  datasets reported in the manuscript.

Each dataset includes source metadata and checksums. The experiment runners use
the committed files directly. LPMC derivatives may optionally be generated
with:

```bash
python scripts/process_lpmc_london.py
```

These third-party datasets are not covered by the repository's MIT License.
Their attribution and upstream terms are summarized in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
