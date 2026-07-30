# TorchDCM source snapshot

The `torchdcm/` directory is an unmodified source snapshot of the public
[TorchDCM repository](https://github.com/mbc96325/torchdcm) at:

- release tag: `v0.1.2`
- Git commit: `86bd009795b86f4589403c34c988161d02eb94cc`
- package version: `0.1.2`

The snapshot was extracted with `git archive v0.1.2 torchdcm`. It is committed
here so that the paper's Python experiments use the exact implementation
evaluated in the manuscript without downloading TorchDCM from PyPI or cloning
a second repository.

The standalone benchmark package installs this local directory through the
root `pyproject.toml`. The actively maintained package and its later releases
remain in the separate TorchDCM repository. The copyright holder distributes
this archived source under the repository's [MIT License](LICENSE).
