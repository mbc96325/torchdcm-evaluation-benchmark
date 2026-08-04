from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import torch

try:
    from .advanced_cases import (
        make_hybrid_actual,
        make_hybrid_synthetic,
        make_latent_class_actual,
        make_latent_class_synthetic,
        make_panel_actual,
        make_panel_synthetic,
    )
except ImportError:  # Support direct execution of scripts in experiments/.
    from advanced_cases import (
        make_hybrid_actual,
        make_hybrid_synthetic,
        make_latent_class_actual,
        make_latent_class_synthetic,
        make_panel_actual,
        make_panel_synthetic,
    )


def _aligned_latent_params(params: dict[str, float]) -> dict[str, float]:
    """Use the class ordering reported in the electronic companion."""
    aligned = dict(params)
    if aligned["B_X_C1"] <= aligned["B_X_C2"]:
        return aligned
    for stem in ("B_X", "ASC_B", "ASC_C"):
        aligned[f"{stem}_C1"], aligned[f"{stem}_C2"] = (
            aligned[f"{stem}_C2"],
            aligned[f"{stem}_C1"],
        )
    aligned["CLASS_2"] = -aligned["CLASS_2"]
    aligned["CLASS_2_Z"] = -aligned["CLASS_2_Z"]
    return aligned


def _comparable_params(case: dict) -> dict[str, dict[str, float]]:
    values = {}
    for backend, result in case["results"].items():
        params = result.get("params")
        if (
            result.get("available")
            and not result.get("worse_loglike", False)
            and params
        ):
            values[backend] = (
                _aligned_latent_params(params)
                if case["kind"] == "latent_class"
                else dict(params)
            )
    return values


def _max_pairwise(values: dict[str, np.ndarray]) -> float | None:
    if len(values) < 2:
        return None
    return max(
        float(np.max(np.abs(values[left] - values[right])))
        for left, right in combinations(values, 2)
    )


def _parameter_difference(params: dict[str, dict[str, float]]) -> float | None:
    arrays = {
        backend: np.asarray([values[name] for name in sorted(values)], dtype=float)
        for backend, values in params.items()
    }
    return _max_pairwise(arrays)


def _natural_vector(model, data, params: dict[str, float]) -> torch.Tensor:
    compiled = model.compile(data)
    return torch.as_tensor(
        [params[name] for name in compiled.free_names],
        dtype=torch.float64,
    )


def _latent_diagnostics(case: dict, params: dict[str, dict[str, float]]) -> dict:
    if case["data_type"] == "Synthetic":
        _, data, model, _ = make_latent_class_synthetic(
            case["n_obs"], seed=100 + case["n_obs"]
        )
    else:
        _, data, model, _ = make_latent_class_actual(case["n_obs"])

    shares = {}
    probabilities = {}
    with torch.no_grad():
        for backend, values in params.items():
            vector = _natural_vector(model, data, values)
            shares[backend] = (
                model.class_probabilities(vector, data)
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            probabilities[backend] = (
                model.predict_proba(data, vector).detach().cpu().numpy()
            )
    return {
        "max_parameter_difference": _parameter_difference(params),
        "max_mean_class_share_difference": _max_pairwise(shares),
        "max_probability_difference": _max_pairwise(probabilities),
    }


def _hybrid_diagnostics(case: dict, params: dict[str, dict[str, float]]) -> dict:
    if case["data_type"] == "Synthetic":
        _, data, model, _, _ = make_hybrid_synthetic(
            case["n_obs"],
            case["n_draws"],
            seed=200 + case["n_obs"],
        )
    else:
        _, data, model, _, _ = make_hybrid_actual(
            case["n_obs"],
            case["n_draws"],
            seed=400 + case["n_obs"],
        )

    probabilities = {}
    with torch.no_grad():
        for backend, values in params.items():
            vector = _natural_vector(model, data, values)
            probabilities[backend] = (
                model.predict_proba(
                    data,
                    vector,
                    condition_on_indicators=True,
                )
                .detach()
                .cpu()
                .numpy()
            )
    return {
        "max_parameter_difference": _parameter_difference(params),
        "max_probability_difference": _max_pairwise(probabilities),
    }


def _panel_sequence_probabilities(model, data, params: torch.Tensor) -> np.ndarray:
    compiled = model.compile(data)
    probabilities = model._prob_per_obs_alt_draw(params, data, compiled)
    if compiled.choice_set_width is None:
        raise ValueError("The archived panel cases use balanced choice sets.")
    chosen_local = (data.chosen_row - data.obs_ptr[:-1]).reshape(-1, 1, 1)
    chosen = probabilities.gather(
        1, chosen_local.expand(-1, 1, probabilities.shape[2])
    ).squeeze(1)
    log_chosen = torch.log(
        torch.clamp(chosen, min=torch.finfo(chosen.dtype).tiny)
    )
    if data.obs_to_ind is None:
        raise ValueError("Panel diagnostics require individual identifiers.")
    n_units = int(data.obs_to_ind.max().item()) + 1
    log_sequence = torch.zeros(
        (n_units, probabilities.shape[2]),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    log_sequence.index_add_(0, data.obs_to_ind, log_chosen)
    sequence = torch.exp(
        torch.logsumexp(log_sequence, dim=1)
        - np.log(probabilities.shape[2])
    )
    return sequence.detach().cpu().numpy()


def _panel_diagnostics(case: dict, params: dict[str, dict[str, float]]) -> dict:
    if case["data_type"] == "Synthetic":
        choices = case["extra"]["choices_per_unit"]
        _, data, model, _, _ = make_panel_synthetic(
            case["n_units"],
            choices,
            case["n_draws"],
            seed=500 + case["n_units"] + choices,
        )
    else:
        _, data, model, _, _ = make_panel_actual(
            case["n_units"],
            case["n_draws"],
            seed=800 + case["n_units"],
        )

    probabilities = {}
    with torch.no_grad():
        for backend, values in params.items():
            vector = _natural_vector(model, data, values)
            probabilities[backend] = _panel_sequence_probabilities(
                model, data, vector
            )
    return {
        "max_parameter_difference": _parameter_difference(params),
        "max_sequence_probability_difference": _max_pairwise(probabilities),
    }


_DIAGNOSTICS: dict[str, Callable[[dict, dict], dict]] = {
    "latent_class": _latent_diagnostics,
    "hybrid_choice": _hybrid_diagnostics,
    "panel_likelihood": _panel_diagnostics,
}


def enrich_advanced_diagnostics(payload: dict) -> dict:
    """Add the EC table diagnostics to a full-estimation result payload."""
    for case in payload["cases"]:
        params = _comparable_params(case)
        case["diagnostics"] = _DIAGNOSTICS[case["kind"]](case, params)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    enrich_advanced_diagnostics(payload)
    output = args.output or args.input
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
