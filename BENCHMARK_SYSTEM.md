# Benchmark protocol

This document records the conventions shared by the experiments in the
TorchDCM software paper.

## Alignment

Within each row, estimators receive the same data, utility specification,
parameter scale, and starting values. Simulated models use common antithetic
normal draws where the external software accepts user-supplied draws. Apollo
uses its documented draw interface when a shared matrix cannot be supplied.
All empirical inputs are read from the versioned files under `data/`. The
runners do not fetch data during an experiment.

The Torch-Choice adapter maps the same compiled MNL or NL design to its dense
choice-dataset interface and supplies availability masks for ragged choice
sets. It uses float64 tensors, the same starting values, and full L-BFGS
estimation followed by classic covariance construction. Translation into this
package-specific input layout occurs before timing.

The synthetic generator varies sample size, choice-set size, coefficient
dimension, feature correlation, and, for MixL, the number of random
coefficients. NL choices are generated from a two-level nested-logit process,
and MixL choices are generated from random coefficients rather than an MNL
surrogate.

## Runtime scope

Cross-estimator CPU runs use one logical CPU. The runners set OpenMP, BLAS,
NumExpr, and PyTorch thread counts to one, pin the process and its children
where the operating system supports affinity, and set Apollo `nCores=1`.
MNL runs for TorchDCM and Torch-Choice perform one untimed forward/backward
warm-up before optimization so one-time PyTorch kernel and autograd
initialization is excluded symmetrically.

Reported runtime includes parameter estimation and covariance construction.
Data loading or generation, design translation, process startup, and file
input/output are excluded. Compilation performed inside an estimator's fit
call remains included. Stress cases use a 300-second worker limit. Actual-data
NL and MixL cases run in fresh child processes.

The advanced-model suite uses the timing boundary documented in Table EC.8.
Its TorchDCM models are compiled before the timed `fit` call, and the resulting
JSON records this scope. This exception is applied consistently to the six
latent-class, six hybrid-choice, and six panel cases.

The device experiment uses the same TorchDCM specification, starting values,
and simulation draws on CPU and CUDA. Its reported values are medians of three
full estimation runs.

## Numerical comparison

The runners retain final log likelihoods and available parameter, covariance,
and probability diagnostics. A completed solution is marked as clearly worse
when its final log likelihood is below the row best by more than

```text
max(0.25, 1e-5 * abs(best_loglike), 0.01 * n_observations).
```

Such a runtime remains visible but the solution is excluded from consistency.
`Yes` requires at least two remaining solutions within the tolerance. `N.A.`
means fewer than two comparable solutions remain. `Fail` records an attempted
run that did not complete successfully. `Timeout` records a worker that
reaches the 300-second limit or is externally terminated under the same
constrained run. The JSON retains the raw elapsed time and termination message.

## Output policy

The top-level JSON files in `results/` are the authoritative outputs used by
the main paper and electronic companion. `results/intermediate/` retains
subordinate outputs and solver logs used to assemble them. Reproduction
commands use a new output suffix so that the committed results remain
auditable. The JSON files include case dimensions, runtime-policy metadata,
backend status, final likelihoods, and available numerical diagnostics.
