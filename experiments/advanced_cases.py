from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchdcm import (
    Beta,
    ChoiceDataset,
    ChoiceLatentEffect,
    ContinuousIndicator,
    HybridChoiceModel,
    LatentClassLogit,
    LatentVariable,
    MixedLogit,
    RandomCoefficient,
    UtilitySpec,
)

from compare_mnl_estimators import load_biogeme_swissmetro


# Actual-data cases use the canonical tables committed with this benchmark.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "small"

PARAMS_LATENT = {
    "ASC_B_C1": -0.45,
    "ASC_C_C1": 0.25,
    "B_X_C1": -1.10,
    "ASC_B_C2": 0.75,
    "ASC_C_C2": -0.35,
    "B_X_C2": -0.35,
    "CLASS_2": -0.25,
    "CLASS_2_Z": 0.90,
}

PARAMS_HYBRID = {
    "ASC_B": 0.25,
    "B_X": 0.70,
    "G_Q": 0.65,
    "SIGMA_LV": 0.80,
    "B_ATT": 0.90,
    "SIGMA_Y1": 0.70,
    "A2": 0.20,
    "L2": 0.75,
    "SIGMA_Y2": 0.85,
}

PARAMS_PANEL = {
    "ASC_B": 0.35,
    "B_X": -0.80,
    "ASC_C": -0.20,
    "ASC_D": 0.10,
    "SIGMA_B_X": 0.55,
}

STARTS_LATENT = {
    "ASC_B_C1": -0.30,
    "ASC_C_C1": 0.15,
    "B_X_C1": -0.80,
    "ASC_B_C2": 0.50,
    "ASC_C_C2": -0.20,
    "B_X_C2": -0.20,
    "CLASS_2": -0.10,
    "CLASS_2_Z": 0.60,
}

STARTS_HYBRID = {
    "ASC_B": 0.10,
    "B_X": 0.40,
    "G_Q": 0.40,
    "SIGMA_LV": 0.60,
    "B_ATT": 0.60,
    "SIGMA_Y1": 0.80,
    "A2": 0.10,
    "L2": 0.60,
    "SIGMA_Y2": 0.90,
}

STARTS_PANEL = {
    "ASC_B": 0.20,
    "B_X": -0.50,
    "ASC_C": -0.10,
    "ASC_D": 0.05,
    "SIGMA_B_X": 0.40,
}

def antithetic_draws(n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = (n_draws + 1) // 2
    base = rng.standard_normal(half)
    return np.concatenate([base, -base])[:n_draws]


def make_latent_model(
    frame: pd.DataFrame,
) -> tuple[ChoiceDataset, LatentClassLogit, torch.Tensor]:
    alternatives = ["A", "B", "C"]
    data = ChoiceDataset.from_wide(
        frame,
        alternatives=alternatives,
        choice="choice",
        variables={
            "x": {alt: f"x_{alt}" for alt in alternatives},
            "z": {alt: "z" for alt in alternatives},
        },
        availability={alt: f"av_{alt}" for alt in alternatives},
        obs_id="id",
    )
    specs = []
    for suffix in ("C1", "C2"):
        spec = UtilitySpec()
        spec.utility("A", Beta(f"B_X_{suffix}", init=STARTS_LATENT[f"B_X_{suffix}"]) * "x")
        spec.utility(
            "B",
            Beta(f"ASC_B_{suffix}", init=STARTS_LATENT[f"ASC_B_{suffix}"])
            + Beta(f"B_X_{suffix}", init=STARTS_LATENT[f"B_X_{suffix}"]) * "x",
        )
        spec.utility(
            "C",
            Beta(f"ASC_C_{suffix}", init=STARTS_LATENT[f"ASC_C_{suffix}"])
            + Beta(f"B_X_{suffix}", init=STARTS_LATENT[f"B_X_{suffix}"]) * "x",
        )
        specs.append(spec)
    membership = [
        Beta("CLASS_2", init=STARTS_LATENT["CLASS_2"])
        + Beta("CLASS_2_Z", init=STARTS_LATENT["CLASS_2_Z"]) * "z"
    ]
    model = LatentClassLogit(specs, class_membership=membership)
    compiled = model.compile(data)
    params = torch.as_tensor([PARAMS_LATENT[name] for name in compiled.free_names], dtype=torch.float64)
    return data, model, params


def make_latent_class_actual(
    n_obs: int,
) -> tuple[pd.DataFrame, ChoiceDataset, LatentClassLogit, torch.Tensor]:
    raw, _, _, _ = load_biogeme_swissmetro(n_obs)
    frame = pd.DataFrame(
        {
            "id": np.arange(len(raw)),
            "choice": raw["choice"].map({"TRAIN": "A", "SM": "B", "CAR": "C"}),
            "choice_code": raw["choice"].map({"TRAIN": 1, "SM": 2, "CAR": 3}).astype(int),
            "x_A": raw["time_train"].astype(float),
            "x_B": raw["time_sm"].astype(float),
            "x_C": raw["time_car"].astype(float),
            "z": raw["GA"].astype(float),
            "av_A": raw["avail_train"].astype(int),
            "av_B": raw["avail_sm"].astype(int),
            "av_C": raw["avail_car"].astype(int),
        }
    )
    data, model, params = make_latent_model(frame)
    return frame, data, model, params


def make_latent_class_synthetic(
    n_obs: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, LatentClassLogit, torch.Tensor]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_obs)
    x = rng.standard_normal((n_obs, 3))
    class_2_probability = 1.0 / (
        1.0 + np.exp(-(PARAMS_LATENT["CLASS_2"] + PARAMS_LATENT["CLASS_2_Z"] * z))
    )
    class_index = (rng.random(n_obs) < class_2_probability).astype(int)
    utility = np.empty((n_obs, 3), dtype=float)
    for index, suffix in enumerate(("C1", "C2")):
        selected = class_index == index
        utility[selected, 0] = PARAMS_LATENT[f"B_X_{suffix}"] * x[selected, 0]
        utility[selected, 1] = (
            PARAMS_LATENT[f"ASC_B_{suffix}"]
            + PARAMS_LATENT[f"B_X_{suffix}"] * x[selected, 1]
        )
        utility[selected, 2] = (
            PARAMS_LATENT[f"ASC_C_{suffix}"]
            + PARAMS_LATENT[f"B_X_{suffix}"] * x[selected, 2]
        )
    gumbel = -np.log(-np.log(rng.random((n_obs, 3))))
    chosen = np.argmax(utility + gumbel, axis=1)
    labels = np.asarray(["A", "B", "C"])
    frame = pd.DataFrame(
        {
            "id": np.arange(n_obs),
            "choice": labels[chosen],
            "choice_code": chosen + 1,
            "x_A": x[:, 0],
            "x_B": x[:, 1],
            "x_C": x[:, 2],
            "z": z,
            "av_A": np.ones(n_obs, dtype=int),
            "av_B": np.ones(n_obs, dtype=int),
            "av_C": np.ones(n_obs, dtype=int),
        }
    )
    data, model, params = make_latent_model(frame)
    return frame, data, model, params


def make_hybrid_model(
    frame: pd.DataFrame,
    draws: np.ndarray,
) -> tuple[ChoiceDataset, HybridChoiceModel, torch.Tensor]:
    data = ChoiceDataset.from_wide(
        frame,
        alternatives=["A", "B"],
        choice="choice",
        variables={"x": {"A": "x_A", "B": "x_B"}},
        obs_variables={"q": "q", "y1": "y1", "y2": "y2"},
        obs_id="id",
    )
    spec = UtilitySpec()
    spec.utility("A", Beta("ASC_A", init=0.0, fixed=True))
    spec.utility(
        "B",
        Beta("ASC_B", init=STARTS_HYBRID["ASC_B"])
        + Beta("B_X", init=STARTS_HYBRID["B_X"]) * "x",
    )
    model = HybridChoiceModel(
        spec,
        latent_variables=[
            LatentVariable(
                "ATT",
                intercept=0.0,
                coefficients={"q": Beta("G_Q", init=STARTS_HYBRID["G_Q"])},
                sigma_name="SIGMA_LV",
                sigma_init=STARTS_HYBRID["SIGMA_LV"],
                sigma_fixed=False,
            )
        ],
        choice_effects=[
            ChoiceLatentEffect(
                "B", "ATT", Beta("B_ATT", init=STARTS_HYBRID["B_ATT"])
            )
        ],
        indicators=[
            ContinuousIndicator(
                "y1",
                "ATT",
                intercept=0.0,
                loading=1.0,
                sigma_name="SIGMA_Y1",
                sigma_init=STARTS_HYBRID["SIGMA_Y1"],
                sigma_fixed=False,
            ),
            ContinuousIndicator(
                "y2",
                "ATT",
                intercept=Beta("A2", init=STARTS_HYBRID["A2"]),
                loading=Beta("L2", init=STARTS_HYBRID["L2"]),
                sigma_name="SIGMA_Y2",
                sigma_init=STARTS_HYBRID["SIGMA_Y2"],
                sigma_fixed=False,
            ),
        ],
        draws=torch.as_tensor(draws[:, None], dtype=torch.float64),
        panel=False,
    )
    compiled = model.compile(data)
    params = torch.as_tensor([PARAMS_HYBRID[name] for name in compiled.free_names], dtype=torch.float64)
    return data, model, params


def make_hybrid_synthetic(
    n_obs: int,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, HybridChoiceModel, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n_obs)
    x = rng.standard_normal(n_obs)
    latent = PARAMS_HYBRID["G_Q"] * q + PARAMS_HYBRID["SIGMA_LV"] * rng.standard_normal(n_obs)
    utility_b = (
        PARAMS_HYBRID["ASC_B"]
        + PARAMS_HYBRID["B_X"] * x
        + PARAMS_HYBRID["B_ATT"] * latent
    )
    probability_b = 1.0 / (1.0 + np.exp(-utility_b))
    choice_b = rng.random(n_obs) < probability_b
    y1 = latent + PARAMS_HYBRID["SIGMA_Y1"] * rng.standard_normal(n_obs)
    y2 = (
        PARAMS_HYBRID["A2"]
        + PARAMS_HYBRID["L2"] * latent
        + PARAMS_HYBRID["SIGMA_Y2"] * rng.standard_normal(n_obs)
    )
    frame = pd.DataFrame(
        {
            "id": np.arange(n_obs),
            "choice": np.where(choice_b, "B", "A"),
            "choice_code": np.where(choice_b, 2, 1),
            "x_A": np.zeros(n_obs),
            "x_B": x,
            "x": x,
            "q": q,
            "y1": y1,
            "y2": y2,
        }
    )
    draws = antithetic_draws(n_draws, seed + 1000)
    data, model, params = make_hybrid_model(frame, draws)
    return frame, data, model, params, draws


def standardized(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    return (array - array.mean()) / array.std(ddof=0)


def make_hybrid_actual(
    n_obs: int,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, HybridChoiceModel, torch.Tensor, np.ndarray]:
    raw = pd.read_csv(DATA_ROOT / "biogeme_optima" / "data.csv")
    valid = (
        raw["Choice"].isin([1, 2])
        & (raw["Envir01"] > 0)
        & (raw["Envir02"] > 0)
        & np.isfinite(
            raw[["TimePT_scaled", "TimeCar_scaled", "ScaledIncome"]].to_numpy(dtype=float)
        ).all(axis=1)
    )
    raw = raw.loc[valid].sample(frac=1.0, random_state=7321).reset_index(drop=True)
    q = standardized(raw["ScaledIncome"])
    x = standardized(raw["TimePT_scaled"] - raw["TimeCar_scaled"])
    y1 = standardized(raw["Envir01"])
    y2 = standardized(raw["Envir02"])
    raw = raw.iloc[:n_obs].copy()
    frame = pd.DataFrame(
        {
            "id": np.arange(len(raw)),
            "choice": np.where(raw["Choice"].to_numpy(dtype=int) == 2, "B", "A"),
            "choice_code": raw["Choice"].to_numpy(dtype=int),
            "x_A": np.zeros(len(raw)),
            "x_B": x[: len(raw)],
            "x": x[: len(raw)],
            "q": q[: len(raw)],
            "y1": y1[: len(raw)],
            "y2": y2[: len(raw)],
        }
    )
    draws = antithetic_draws(n_draws, seed + 1000)
    data, model, params = make_hybrid_model(frame, draws)
    return frame, data, model, params, draws


def make_panel_model(
    frame: pd.DataFrame,
    alternatives: list[str],
    draws: np.ndarray,
) -> tuple[ChoiceDataset, MixedLogit, torch.Tensor]:
    data = ChoiceDataset.from_wide(
        frame,
        alternatives=alternatives,
        choice="choice",
        variables={"x": {alt: f"x_{alt}" for alt in alternatives}},
        obs_id="obs_id",
        individual_id="person_id",
    )
    spec = UtilitySpec()
    spec.utility("A", Beta("B_X", init=STARTS_PANEL["B_X"]) * "x")
    for alt in alternatives[1:]:
        spec.utility(
            alt,
            Beta(f"ASC_{alt}", init=STARTS_PANEL[f"ASC_{alt}"])
            + Beta("B_X", init=STARTS_PANEL["B_X"]) * "x",
        )
    model = MixedLogit(
        spec,
        [RandomCoefficient("B_X", sigma_init=STARTS_PANEL["SIGMA_B_X"])],
        draws=torch.as_tensor(draws[:, None], dtype=torch.float64),
        panel=True,
    )
    compiled = model.compile(data)
    params = torch.as_tensor([PARAMS_PANEL[name] for name in compiled.free_names], dtype=torch.float64)
    return data, model, params


def make_panel_synthetic(
    n_units: int,
    choices_per_unit: int,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, MixedLogit, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    person_id = np.repeat(np.arange(n_units), choices_per_unit)
    n_obs = len(person_id)
    alternatives = ["A", "B", "C", "D"]
    x = rng.standard_normal((n_obs, len(alternatives)))
    individual_beta = (
        PARAMS_PANEL["B_X"]
        + PARAMS_PANEL["SIGMA_B_X"] * rng.standard_normal(n_units)
    )[person_id]
    utilities = individual_beta[:, None] * x
    utilities[:, 1] += PARAMS_PANEL["ASC_B"]
    utilities[:, 2] += PARAMS_PANEL["ASC_C"]
    utilities[:, 3] += PARAMS_PANEL["ASC_D"]
    gumbel = -np.log(-np.log(rng.random((n_obs, len(alternatives)))))
    chosen = np.argmax(utilities + gumbel, axis=1)
    labels = np.asarray(alternatives)
    frame = pd.DataFrame(
        {
            "obs_id": np.arange(n_obs),
            "person_id": person_id,
            "choice": labels[chosen],
            "choice_code": chosen + 1,
            "x_A": x[:, 0],
            "x_B": x[:, 1],
            "x_C": x[:, 2],
            "x_D": x[:, 3],
        }
    )
    draws = antithetic_draws(n_draws, seed + 1000)
    data, model, params = make_panel_model(frame, alternatives, draws)
    return frame, data, model, params, draws


def make_panel_actual(
    n_units: int,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, ChoiceDataset, MixedLogit, torch.Tensor, np.ndarray]:
    raw = pd.read_csv(DATA_ROOT / "mlogit_electricity" / "data.csv")
    counts = raw.groupby("id").size()
    complete_ids = counts.index[counts == 12].to_numpy()
    selected_ids = np.sort(complete_ids)[:n_units]
    raw = raw.loc[raw["id"].isin(selected_ids)].reset_index(drop=True)
    alternatives = ["A", "B", "C", "D"]
    price = raw[["pf1", "pf2", "pf3", "pf4"]].to_numpy(dtype=float)
    price = (price - price.mean()) / price.std(ddof=0)
    choice_code = raw["choice"].to_numpy(dtype=int)
    labels = np.asarray(alternatives)
    frame = pd.DataFrame(
        {
            "obs_id": np.arange(len(raw)),
            "person_id": raw["id"].to_numpy(dtype=int),
            "choice": labels[choice_code - 1],
            "choice_code": choice_code,
            "x_A": price[:, 0],
            "x_B": price[:, 1],
            "x_C": price[:, 2],
            "x_D": price[:, 3],
        }
    )
    draws = antithetic_draws(n_draws, seed + 1000)
    data, model, params = make_panel_model(frame, alternatives, draws)
    return frame, data, model, params, draws
