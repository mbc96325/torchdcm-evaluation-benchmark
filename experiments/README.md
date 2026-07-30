# Experiments

The Python files in this directory implement the paper's full-estimation
experiments. The `apollo/R` and `mlogit/R` directories translate the same
aligned cases to Apollo, `mlogit`, and `gmnl`.

Use the stable root entry point instead of invoking individual drivers when
reproducing an archived paper result:

```bash
python run_exp.py --list
python run_exp.py synthetic-mnl
```

Direct script execution is intended for development, smoke profiles, or custom
output paths. The shared runtime and consistency rules are documented in
[`../BENCHMARK_SYSTEM.md`](../BENCHMARK_SYSTEM.md).
