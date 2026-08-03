from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch

from torchdcm import MultinomialLogit


BACKEND = "torch_choice"
DESIGN_NAME = "itemsession_design"
DESIGN_KEY = f"{DESIGN_NAME}[constant]"


def unavailable(message: str, seconds: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        backend=BACKEND,
        available=False,
        total_s=seconds,
        seconds=seconds,
        estimate_s=None,
        estimate_seconds=None,
        covariance_s=None,
        covariance_seconds=None,
        loglike=None,
        params=None,
        covariance=None,
        probabilities=None,
        message=message,
    )


def _imports():
    try:
        from torch.func import functional_call
        from torch_choice.data import ChoiceDataset as TorchChoiceDataset
        from torch_choice.model import ConditionalLogitModel, NestedLogitModel
    except ImportError as exc:
        raise ImportError(f"Torch-Choice unavailable: {exc}") from exc
    return functional_call, TorchChoiceDataset, ConditionalLogitModel, NestedLogitModel


def _dense_compiled_design(data, spec):
    """Translate TorchDCM's row-pointer data into Torch-Choice's dense layout."""
    compiled = MultinomialLogit(spec).compile(data)
    row_design = compiled.design.detach().to(device="cpu", dtype=torch.float64)
    if compiled.fixed_values.numel():
        fixed_utility = compiled.fixed_design @ compiled.fixed_values
        row_design = torch.cat(
            [row_design, fixed_utility.detach().to(device="cpu", dtype=torch.float64).reshape(-1, 1)],
            dim=1,
        )
        has_fixed_utility = True
    else:
        has_fixed_utility = False

    n_obs = data.n_obs
    n_items = len(data.alt_names)
    n_design = row_design.shape[1]
    dense = torch.zeros((n_obs, n_items, n_design), dtype=torch.float64)
    available = torch.zeros((n_obs, n_items), dtype=torch.bool)
    widths = (data.obs_ptr[1:] - data.obs_ptr[:-1]).detach().cpu()
    row_obs = torch.repeat_interleave(torch.arange(n_obs), widths)
    row_alt = data.alt_id.detach().cpu().long()
    dense[row_obs, row_alt] = row_design
    available[row_obs, row_alt] = data.availability.detach().cpu().bool()
    chosen_items = row_alt[data.chosen_row.detach().cpu().long()]
    weights = data.weights.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("Torch-Choice adapter requires finite, nonnegative observation weights.")
    return compiled, dense, available, chosen_items, row_obs, row_alt, weights, has_fixed_utility


def _choice_dataset(TorchChoiceDataset, dense, available, chosen_items):
    n_obs, n_items, _ = dense.shape
    dataset = TorchChoiceDataset(
        item_index=chosen_items.long(),
        num_items=n_items,
        num_sessions=n_obs,
        session_index=torch.arange(n_obs, dtype=torch.long),
        item_availability=available,
        **{DESIGN_NAME: dense},
    )
    # Torch-Choice currently casts observables to float32 in its constructor.
    # Restore float64 so all benchmark estimators use the same numerical precision.
    setattr(dataset, DESIGN_NAME, dense)
    return dataset


def _lbfgs(initial: torch.Tensor, objective, max_iter: int):
    warmup = initial.detach().clone().requires_grad_(True)
    (-objective(warmup)).backward()
    parameters = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [parameters],
        max_iter=max_iter,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    closure_evals = 0

    def closure():
        nonlocal closure_evals
        closure_evals += 1
        optimizer.zero_grad(set_to_none=True)
        loss = -objective(parameters)
        loss.backward()
        return loss

    start = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - start
    state = optimizer.state.get(parameters, {})
    return parameters.detach(), elapsed, closure_evals, int(state.get("n_iter", 0))


def run_mnl(data, spec, initial_values: Mapping[str, float], max_iter: int = 500) -> SimpleNamespace:
    start = time.perf_counter()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        _, TorchChoiceDataset, ConditionalLogitModel, _ = _imports()
        (
            compiled,
            dense,
            available,
            chosen_items,
            row_obs,
            row_alt,
            weights,
            has_fixed_utility,
        ) = _dense_compiled_design(data, spec)
        dataset = _choice_dataset(TorchChoiceDataset, dense, available, chosen_items)
        model = ConditionalLogitModel(
            coef_variation_dict={DESIGN_NAME: "constant"},
            num_param_dict={DESIGN_NAME: dense.shape[-1]},
            num_items=dense.shape[1],
            weight_initialization="zero",
        ).double()
        initial = torch.as_tensor(
            [initial_values.get(name, 0.0) for name in compiled.free_names],
            dtype=torch.float64,
        )

        def natural_coefficients(beta: torch.Tensor) -> torch.Tensor:
            if has_fixed_utility:
                return torch.cat([beta, torch.ones(1, dtype=beta.dtype, device=beta.device)])
            return beta

        def loglike(beta: torch.Tensor) -> torch.Tensor:
            utility = model.forward(dataset, {DESIGN_KEY: natural_coefficients(beta)})
            chosen_log_probability = torch.log_softmax(utility, dim=1)[
                torch.arange(len(chosen_items)), chosen_items
            ]
            return torch.dot(weights, chosen_log_probability)

        final, estimate_s, closure_evals, optimizer_iterations = _lbfgs(initial, loglike, max_iter)
        final_loglike = float(loglike(final).detach())
        covariance_start = time.perf_counter()
        hessian = torch.autograd.functional.hessian(loglike, final)
        covariance = torch.linalg.pinv(-hessian.detach(), hermitian=True).cpu().numpy()
        covariance_s = time.perf_counter() - covariance_start
        utility = model.forward(dataset, {DESIGN_KEY: natural_coefficients(final)})
        dense_probability = torch.softmax(utility, dim=1).detach()
        row_probability = dense_probability[row_obs, row_alt].cpu().numpy()
        return SimpleNamespace(
            backend=BACKEND,
            available=True,
            total_s=estimate_s + covariance_s,
            seconds=estimate_s + covariance_s,
            estimate_s=estimate_s,
            estimate_seconds=estimate_s,
            covariance_s=covariance_s,
            covariance_seconds=covariance_s,
            loglike=final_loglike,
            params={name: float(final[i]) for i, name in enumerate(compiled.free_names)},
            covariance=covariance,
            probabilities=row_probability,
            message="",
            closure_evals=closure_evals,
            optimizer_iterations=optimizer_iterations,
        )
    except Exception as exc:
        return unavailable(f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
    finally:
        torch.set_default_dtype(previous_dtype)


def run_nested(
    data,
    spec,
    alternatives: list[str],
    beta_names: list[str],
    nests: Mapping[str, object],
    initial_values: Mapping[str, float],
    lambda_names: list[str],
    lambda_min: float = 1e-5,
    max_iter: int = 500,
) -> SimpleNamespace:
    start = time.perf_counter()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        functional_call, TorchChoiceDataset, _, NestedLogitModel = _imports()
        (
            compiled,
            dense,
            available,
            chosen_items,
            row_obs,
            row_alt,
            weights,
            has_fixed_utility,
        ) = _dense_compiled_design(data, spec)
        if list(compiled.free_names) != list(beta_names):
            raise ValueError(
                "Torch-Choice NL adapter requires beta_names to match the compiled free-parameter order."
            )
        item_dataset = _choice_dataset(TorchChoiceDataset, dense, available, chosen_items)
        nest_names = list(nests)
        alt_to_code = {alternative: index for index, alternative in enumerate(alternatives)}
        nest_to_item = {
            nest_index: [alt_to_code[alternative] for alternative in nests[nest_name].alternatives]
            for nest_index, nest_name in enumerate(nest_names)
        }
        chosen_nests = torch.empty_like(chosen_items)
        for nest_index, item_codes in nest_to_item.items():
            mask = torch.zeros(len(alternatives), dtype=torch.bool)
            mask[item_codes] = True
            chosen_nests[mask[chosen_items]] = nest_index
        nest_dataset = TorchChoiceDataset(
            item_index=chosen_nests,
            num_items=len(nest_names),
            num_sessions=data.n_obs,
            session_index=torch.arange(data.n_obs, dtype=torch.long),
        )
        batch = {"nest": nest_dataset, "item": item_dataset}
        model = NestedLogitModel(
            nest_to_item=nest_to_item,
            nest_coef_variation_dict={},
            nest_num_param_dict={},
            item_coef_variation_dict={DESIGN_NAME: "constant"},
            item_num_param_dict={DESIGN_NAME: dense.shape[-1]},
            shared_lambda=False,
            item_weight_initialization="zero",
        ).double()
        item_parameter_name = f"item_coef_dict.{DESIGN_NAME}.coef"
        free_nest_names = [name for name in nest_names if not nests[name].fixed]
        expected_lambda_names = [f"LAMBDA_{name.upper()}" for name in free_nest_names]
        if list(lambda_names) != expected_lambda_names:
            raise ValueError("Torch-Choice NL adapter received an unexpected lambda-parameter order.")

        beta_initial = [initial_values.get(name, 0.0) for name in beta_names]
        raw_lambda_initial = []
        for nest_name, lambda_name in zip(free_nest_names, expected_lambda_names):
            value = float(initial_values.get(lambda_name, nests[nest_name].init))
            scaled = np.clip((value - lambda_min) / (1.0 - lambda_min), 1e-12, 1.0 - 1e-12)
            raw_lambda_initial.append(float(np.log(scaled / (1.0 - scaled))))
        initial = torch.as_tensor([*beta_initial, *raw_lambda_initial], dtype=torch.float64)

        def unpack(internal: torch.Tensor):
            beta = internal[: len(beta_names)]
            if has_fixed_utility:
                beta = torch.cat([beta, torch.ones(1, dtype=beta.dtype, device=beta.device)])
            free_raw = internal[len(beta_names) :]
            free_lambdas = lambda_min + (1.0 - lambda_min) * torch.sigmoid(free_raw)
            lambda_values = []
            cursor = 0
            for nest_name in nest_names:
                if nests[nest_name].fixed:
                    lambda_values.append(torch.as_tensor(nests[nest_name].init, dtype=internal.dtype))
                else:
                    lambda_values.append(free_lambdas[cursor])
                    cursor += 1
            return beta, torch.stack(lambda_values), free_lambdas

        def loglike(internal: torch.Tensor) -> torch.Tensor:
            beta, lambda_values, _ = unpack(internal)
            parameters = {item_parameter_name: beta, "lambda_weight": lambda_values}
            log_probability = functional_call(model, parameters, (batch,), strict=False)
            chosen_log_probability = log_probability[torch.arange(len(chosen_items)), chosen_items]
            return torch.dot(weights, chosen_log_probability)

        final_internal, estimate_s, closure_evals, optimizer_iterations = _lbfgs(initial, loglike, max_iter)
        final_loglike = float(loglike(final_internal).detach())
        covariance_start = time.perf_counter()
        hessian = torch.autograd.functional.hessian(loglike, final_internal)
        covariance_internal = torch.linalg.pinv(-hessian.detach(), hermitian=True)

        def natural_vector(internal: torch.Tensor) -> torch.Tensor:
            beta = internal[: len(beta_names)]
            _, _, free_lambdas = unpack(internal)
            return torch.cat([beta, free_lambdas])

        jacobian = torch.autograd.functional.jacobian(natural_vector, final_internal)
        covariance = (jacobian @ covariance_internal @ jacobian.T).detach().cpu().numpy()
        covariance_s = time.perf_counter() - covariance_start
        final_beta, final_lambda_values, final_free_lambdas = unpack(final_internal)
        log_probability = functional_call(
            model,
            {item_parameter_name: final_beta, "lambda_weight": final_lambda_values},
            (batch,),
            strict=False,
        )
        dense_probability = log_probability.detach().exp()
        row_probability = dense_probability[row_obs, row_alt].cpu().numpy()
        natural = torch.cat([final_internal[: len(beta_names)], final_free_lambdas])
        parameter_names = [*beta_names, *lambda_names]
        return SimpleNamespace(
            backend=BACKEND,
            available=True,
            total_s=estimate_s + covariance_s,
            seconds=estimate_s + covariance_s,
            estimate_s=estimate_s,
            estimate_seconds=estimate_s,
            covariance_s=covariance_s,
            covariance_seconds=covariance_s,
            loglike=final_loglike,
            params={name: float(natural[i]) for i, name in enumerate(parameter_names)},
            covariance=covariance,
            probabilities=row_probability,
            message="",
            closure_evals=closure_evals,
            optimizer_iterations=optimizer_iterations,
        )
    except Exception as exc:
        return unavailable(f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
    finally:
        torch.set_default_dtype(previous_dtype)
