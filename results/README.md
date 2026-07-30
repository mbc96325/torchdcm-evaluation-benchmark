# Archived results

This directory stores the authoritative JSON outputs used to prepare the
paper's main and electronic-companion tables.

The `intermediate/` directory retains the subordinate MNL dataset outputs used
by the solver matrix and the TorchDCM/Apollo base and replacement outputs and
Biogeme logs used to assemble the advanced-model tables. Debugging and
smoke-test artifacts are excluded because they do not contribute to a reported
paper result.

Reproduction commands write files with a `_reproduction` suffix, which is
ignored by Git so that reruns do not overwrite or obscure the archived
evidence. Runtime values may change across machines. Compare case dimensions,
completion status, final log likelihoods, and the consistency diagnostics
recorded in each file.
