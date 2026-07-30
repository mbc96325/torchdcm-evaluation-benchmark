from __future__ import annotations

import os
import sys
from typing import Any


THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
XLA_SINGLE_THREAD_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
POLICY_NAME = "single-logical-core"
TIMING_SCOPE = "estimation-plus-covariance"


def configure_single_thread_cpu(*, configure_torch: bool = False) -> None:
    """Enforce the CPU policy used by cross-estimator benchmarks.

    The environment is configured before numerical libraries are imported. On
    Linux, CPU affinity additionally limits this process and inherited child
    processes, including R workers, to one logical CPU.
    """

    for name, value in THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    os.environ["XLA_FLAGS"] = XLA_SINGLE_THREAD_FLAGS
    os.environ["TORCHDCM_BENCHMARK_CPU_POLICY"] = POLICY_NAME
    os.environ["TORCHDCM_BENCHMARK_TIMING_SCOPE"] = TIMING_SCOPE

    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if allowed:
            os.sched_setaffinity(0, {allowed[0]})

    if configure_torch:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Inter-op threads can only be set before parallel work starts.
            # CPU affinity still enforces the one-core resource limit.
            pass


def estimation_covariance_total(estimate_seconds: Any, covariance_seconds: Any) -> float | None:
    """Return the common runtime reported in cross-estimator tables."""

    if estimate_seconds is None:
        return None
    return float(estimate_seconds) + float(covariance_seconds or 0.0)


def runtime_policy_metadata() -> dict[str, Any]:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))

    metadata: dict[str, Any] = {
        "cpu_policy": os.environ.get("TORCHDCM_BENCHMARK_CPU_POLICY", "not-enforced"),
        "logical_cpu_affinity": affinity,
        "timing_scope": os.environ.get("TORCHDCM_BENCHMARK_TIMING_SCOPE", TIMING_SCOPE),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }

    torch = sys.modules.get("torch")
    if torch is not None:
        metadata["torch_intraop_threads"] = int(torch.get_num_threads())
        metadata["torch_interop_threads"] = int(torch.get_num_interop_threads())

    try:
        from threadpoolctl import threadpool_info

        metadata["threadpools"] = [
            {
                "user_api": pool.get("user_api"),
                "internal_api": pool.get("internal_api"),
                "num_threads": pool.get("num_threads"),
            }
            for pool in threadpool_info()
        ]
    except Exception:
        metadata["threadpools"] = []
    return metadata
