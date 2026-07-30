# Large benchmark datasets

The source inputs are committed under `data/raw/`, so reproducing the paper
does not require an external data download.

| Dataset | Archived input | Use |
| --- | --- | --- |
| LPMC London | `data/raw/lpmc_london/data.csv` and `data.dat` | The estimators read the CSV directly. `scripts/process_lpmc_london.py` can create optional wide and long derivatives. |
| NHTS 2022 | `data/raw/nhts_2022/csv.zip` | The estimators read `tripv2pub.csv` directly from the archive. |

Generated derivatives remain untracked under `data/large/processed/`.
`dataset_sources.csv` records provenance and local paths.
