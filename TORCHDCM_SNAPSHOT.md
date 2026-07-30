# TorchDCM source snapshot

The `torchdcm/` directory is an unmodified source snapshot of the public
[TorchDCM repository](https://github.com/mbc96325/torchdcm) at:

- release tag: `v0.1.1`
- Git commit: `b34ab6924523017aca39f5529c940c2cdd817bde`
- package version: `0.1.1`

The snapshot was extracted with `git archive v0.1.1 torchdcm`. It is committed
here so that the paper's Python experiments use the exact implementation
evaluated in the manuscript without downloading TorchDCM from PyPI or cloning
a second repository.

The standalone benchmark package installs this local directory through the
root `pyproject.toml`. The actively maintained package and its later releases
remain in the separate TorchDCM repository.
