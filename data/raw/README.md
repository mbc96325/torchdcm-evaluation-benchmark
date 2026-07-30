# Archived raw data

This directory contains the two paper inputs that are larger than the
dataset-specific CSV files under `data/small/`.

- `lpmc_london/data.csv` is the canonical input read by the benchmark.
  `data.dat` is the original upstream table.
- `nhts_2022/csv.zip` is the official 2022 NHTS public-use CSV archive. The
  benchmark reads `tripv2pub.csv` directly from this ZIP.

Both inputs are committed to ordinary Git. A repository clone therefore
contains every empirical input required by the paper and does not need Git LFS
or an external data download. Checksums and provenance are recorded in the
dataset metadata.
