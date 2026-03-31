from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


# =========================================================
# 1. Small containers
# =========================================================

@dataclass
class GridSearchResult:
    optimal_price: float
    optimal_objective: float
    price_grid: np.ndarray
    objective_values: np.ndarray


@dataclass
class PricingExperimentResult:
    observed_objective: float
    optimal_price_model: float
    model_objective_at_optimal_price: float
    counterfactual_objective_mean: float
    counterfactual_objective_std: float
    counterfactual_objective_values: np.ndarray
    price_grid: np.ndarray
    model_curve: np.ndarray


# =========================================================
# 2. Helpers: MNL probabilities from fitted coefficients
# =========================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def compute_mnl_probabilities_with_outside(
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    price_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute MNL choice probabilities with outside option baseline u0 = 0.

    Parameters
    ----------
    alpha_arr : np.ndarray, shape (N,)
        Customer-specific price sensitivities.
    beta_mat : np.ndarray, shape (N, J)
        Customer-specific product preferences.
        beta_mat[:, 0] corresponds to product 1 (A).
    price_vec : np.ndarray, shape (J,)
        Posted prices for products 1..J.

    Returns
    -------
    prob_mat : np.ndarray, shape (N, J+1)
        Column 0 = outside option probability
        Column 1..J = product probabilities
    """
    alpha_arr = np.asarray(alpha_arr, dtype=float)
    beta_mat = np.asarray(beta_mat, dtype=float)
    price_vec = np.asarray(price_vec, dtype=float)

    if beta_mat.ndim != 2:
        raise ValueError("beta_mat must be 2D with shape (N, J).")
    if alpha_arr.ndim != 1:
        raise ValueError("alpha_arr must be 1D with shape (N,).")
    if beta_mat.shape[0] != alpha_arr.shape[0]:
        raise ValueError("alpha_arr and beta_mat must have matching N.")
    if beta_mat.shape[1] != price_vec.shape[0]:
        raise ValueError("price_vec length must match beta_mat.shape[1].")

    # utilities for products 1..J
    util_products = beta_mat - alpha_arr[:, None] * price_vec[None, :]

    # outside utility is always 0
    util_full = np.concatenate(
        [np.zeros((alpha_arr.shape[0], 1)), util_products],
        axis=1
    )

    return softmax(util_full, axis=1)


# =========================================================
# 3. Model-based objective for A only
# =========================================================

def expected_objective_for_target_product_mnl(
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    price_vec: np.ndarray,
    target_product_idx: int = 1,
    cost_arr: Optional[np.ndarray] = None,
    objective: str = "revenue",
) -> float:
    """
    Expected objective for one target product only under fitted MNL.

    Parameters
    ----------
    target_product_idx : int
        Product label in {1, ..., J}. For A, use 1.
    objective : {"revenue", "profit"}
        revenue: p_j * E[count_j]
        profit : (p_j - c_j) * E[count_j]
    """
    prob_mat = compute_mnl_probabilities_with_outside(
        alpha_arr=alpha_arr,
        beta_mat=beta_mat,
        price_vec=price_vec,
    )

    j = target_product_idx
    if j < 1 or j > price_vec.shape[0]:
        raise ValueError("target_product_idx must be in {1, ..., J}.")

    purchase_prob = prob_mat[:, j]  # because col 0 = outside
    expected_count = purchase_prob.sum()

    if objective == "revenue":
        unit_value = price_vec[j - 1]
    elif objective == "profit":
        if cost_arr is None:
            raise ValueError("cost_arr is required when objective='profit'.")
        unit_value = price_vec[j - 1] - cost_arr[j - 1]
    else:
        raise ValueError("objective must be either 'revenue' or 'profit'.")

    return float(unit_value * expected_count)


def optimize_single_product_price_grid_mnl(
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    base_price_vec: np.ndarray,
    price_grid: np.ndarray,
    target_product_idx: int = 1,
    cost_arr: Optional[np.ndarray] = None,
    objective: str = "revenue",
) -> GridSearchResult:
    """
    Optimize only one product price (e.g. A) using fitted MNL.
    Other product prices remain fixed.
    """
    base_price_vec = np.asarray(base_price_vec, dtype=float).copy()
    price_grid = np.asarray(price_grid, dtype=float)

    values: List[float] = []

    for p in price_grid:
        price_vec = base_price_vec.copy()
        price_vec[target_product_idx - 1] = p

        obj_val = expected_objective_for_target_product_mnl(
            alpha_arr=alpha_arr,
            beta_mat=beta_mat,
            price_vec=price_vec,
            target_product_idx=target_product_idx,
            cost_arr=cost_arr,
            objective=objective,
        )
        values.append(obj_val)

    values_arr = np.asarray(values, dtype=float)
    best_idx = int(np.argmax(values_arr))

    return GridSearchResult(
        optimal_price=float(price_grid[best_idx]),
        optimal_objective=float(values_arr[best_idx]),
        price_grid=price_grid,
        objective_values=values_arr,
    )


# =========================================================
# 4. Observed objective from existing simulated dataset
# =========================================================

def compute_observed_objective_for_target_product(
    df: pd.DataFrame,
    target_product_idx: int = 1,
    price_col_prefix: str = "price_",
    cost_arr: Optional[np.ndarray] = None,
    objective: str = "revenue",
) -> float:
    """
    Compute observed revenue/profit for one product from existing simulated data.

    This supports varying observed prices across rows/periods.
    For product A (idx=1), it sums:
        revenue = sum(price_1 for rows with choice == 1)
        profit  = sum(price_1 - cost_1 for rows with choice == 1)
    """
    choice_col = "choice"
    price_col = f"{price_col_prefix}{target_product_idx}"

    if choice_col not in df.columns:
        raise ValueError("df must contain a 'choice' column.")
    if price_col not in df.columns:
        raise ValueError(f"df must contain '{price_col}' column.")

    chosen_mask = (df[choice_col].to_numpy() == target_product_idx)
    chosen_prices = df.loc[chosen_mask, price_col].to_numpy(dtype=float)

    if objective == "revenue":
        return float(chosen_prices.sum())

    if objective == "profit":
        if cost_arr is None:
            raise ValueError("cost_arr is required when objective='profit'.")
        unit_margin = chosen_prices - float(cost_arr[target_product_idx - 1])
        return float(unit_margin.sum())

    raise ValueError("objective must be either 'revenue' or 'profit'.")


# =========================================================
# 5. Counterfactual evaluation via true DGP
# =========================================================

def make_constant_price_schedule(
    price_vec: np.ndarray,
    n_periods: int,
) -> np.ndarray:
    """
    Create a common posted price schedule of shape (J, T),
    where each product has constant price over time.
    """
    price_vec = np.asarray(price_vec, dtype=float)
    return np.repeat(price_vec[:, None], repeats=n_periods, axis=1)


def compute_counterfactual_objective_from_generated_df(
    df_cf: pd.DataFrame,
    price_vec: np.ndarray,
    target_product_idx: int = 1,
    cost_arr: Optional[np.ndarray] = None,
    objective: str = "revenue",
) -> float:
    """
    Counterfactual objective under fixed price policy.

    Since counterfactual price is fixed across periods in the generated policy,
    we can compute:
        revenue = target_price * count(target chosen)
        profit  = (target_price - target_cost) * count(target chosen)
    """
    choice_arr = df_cf["choice"].to_numpy()
    target_count = int((choice_arr == target_product_idx).sum())

    target_price = float(price_vec[target_product_idx - 1])

    if objective == "revenue":
        return target_price * target_count

    if objective == "profit":
        if cost_arr is None:
            raise ValueError("cost_arr is required when objective='profit'.")
        unit_margin = target_price - float(cost_arr[target_product_idx - 1])
        return unit_margin * target_count

    raise ValueError("objective must be either 'revenue' or 'profit'.")


def simulate_counterfactual_objective_via_true_dgp(
    generate_multinomial_dgp_func,
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    price_vec: np.ndarray,
    n_periods: int,
    target_product_idx: int = 1,
    cost_arr: Optional[np.ndarray] = None,
    objective: str = "revenue",
    n_rep: int = 100,
    error_type: str = "probit",
    base_seed: Optional[int] = None,
    **dgp_kwargs: Any,
) -> np.ndarray:
    """
    Re-generate counterfactual datasets under the true DGP using a new fixed price policy,
    then evaluate target product revenue/profit.

    Parameters
    ----------
    generate_multinomial_dgp_func : callable
        Your existing generate_multinomial_dgp function.
    alpha_arr, beta_mat : true parameters used by the DGP
    price_vec : np.ndarray, shape (J,)
        New counterfactual posted prices.
    n_periods : int
        Number of periods to simulate.
    n_rep : int
        Number of repeated simulations for Monte Carlo evaluation.
    """
    values = []

    n_customers = alpha_arr.shape[0]
    n_products = beta_mat.shape[1]
    price_schedule = make_constant_price_schedule(price_vec, n_periods=n_periods)

    for r in range(n_rep):
        seed_r = None if base_seed is None else base_seed + r

        df_cf, _ = generate_multinomial_dgp_func(
            n_customers=n_customers,
            n_products=n_products,
            n_periods=n_periods,
            beta_mat=beta_mat,
            alpha_arr=alpha_arr,
            price_schedule=price_schedule,
            seed=seed_r,
            error_type=error_type,
            **dgp_kwargs,
        )

        value_r = compute_counterfactual_objective_from_generated_df(
            df=df_cf,
            price_vec=price_vec,
            target_product_idx=target_product_idx,
            cost_arr=cost_arr,
            objective=objective,
        )
        values.append(value_r)

    return np.asarray(values, dtype=float)


# =========================================================
# 6. End-to-end experiment
# =========================================================

def run_single_product_pricing_experiment(
    observed_df: pd.DataFrame,
    generate_multinomial_dgp_func,
    true_alpha_arr: np.ndarray,
    true_beta_mat: np.ndarray,
    est_alpha_arr: np.ndarray,
    est_beta_mat: np.ndarray,
    base_price_vec: np.ndarray,
    price_grid: np.ndarray,
    n_periods_counterfactual: int,
    target_product_idx: int = 1,
    cost_arr: Optional[np.ndarray] = None,
    objective: str = "revenue",
    n_counterfactual_rep: int = 100,
    true_error_type: str = "probit",
    base_seed: Optional[int] = None,
    **dgp_kwargs: Any,
) -> PricingExperimentResult:
    """
    Full pipeline:

    1) observed objective from existing simulated dataset
    2) fit된 model coefficients(est_alpha_arr, est_beta_mat)로 A optimal price 찾기
    3) optimal price를 true DGP에 넣어서 counterfactual objective 평가
    """
    observed_objective = compute_observed_objective_for_target_product(
        df=observed_df,
        target_product_idx=target_product_idx,
        cost_arr=cost_arr,
        objective=objective,
    )

    grid_result = optimize_single_product_price_grid_mnl(
        alpha_arr=est_alpha_arr,
        beta_mat=est_beta_mat,
        base_price_vec=base_price_vec,
        price_grid=price_grid,
        target_product_idx=target_product_idx,
        cost_arr=cost_arr,
        objective=objective,
    )

    optimal_price = grid_result.optimal_price
    cf_price_vec = np.asarray(base_price_vec, dtype=float).copy()
    cf_price_vec[target_product_idx - 1] = optimal_price

    cf_values = simulate_counterfactual_objective_via_true_dgp(
        generate_multinomial_dgp_func=generate_multinomial_dgp_func,
        alpha_arr=true_alpha_arr,
        beta_mat=true_beta_mat,
        price_vec=cf_price_vec,
        n_periods=n_periods_counterfactual,
        target_product_idx=target_product_idx,
        cost_arr=cost_arr,
        objective=objective,
        n_rep=n_counterfactual_rep,
        error_type=true_error_type,
        base_seed=base_seed,
        **dgp_kwargs,
    )

    return PricingExperimentResult(
        observed_objective=float(observed_objective),
        optimal_price_model=float(optimal_price),
        model_objective_at_optimal_price=float(grid_result.optimal_objective),
        counterfactual_objective_mean=float(cf_values.mean()),
        counterfactual_objective_std=float(cf_values.std(ddof=1)) if len(cf_values) > 1 else 0.0,
        counterfactual_objective_values=cf_values,
        price_grid=grid_result.price_grid,
        model_curve=grid_result.objective_values,
    )
