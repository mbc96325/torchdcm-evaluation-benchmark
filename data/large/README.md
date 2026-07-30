# Large benchmark datasets

Large source and processed files are not committed to GitHub.

| Dataset | Preparation |
| --- | --- |
| LPMC London | `python scripts/process_lpmc_london.py` downloads the official Biogeme table when needed and creates canonical wide and long files. |
| NHTS 2022 | `experiments/compare_nhts_mnl.py` downloads the official trip archive when needed and constructs the aligned mode-choice case. |

Downloaded source files and processed tables are stored under ignored
`data/raw/` and `data/large/processed/` directories.
`dataset_sources.csv` records the corresponding source URLs and local artifacts.
