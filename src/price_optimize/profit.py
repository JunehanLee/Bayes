from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# =========================================================
# 1. Result containers
# =========================================================

@dataclass
class PeriodOptimizationResult:
    period: int
    observed_price_vec: np.ndarray          # shape (J,)
    optimal_price_A_vec: np.ndarray         # shape (N,)
    optimal_revenue_A: float
    customer_revenue_curve: np.ndarray      # shape (N, G)


@dataclass
class PricingExperimentResult:
    observed_revenue_A: float
    counterfactual_revenue_A_mean: float
    counterfactual_revenue_A_std: float
    counterfactual_revenue_A_values: np.ndarray
    optimal_price_tensor: np.ndarray        # shape (N, T, J)
    personalized_price_A: np.ndarray        # shape (N, T)
    observed_price_path: np.ndarray         # shape (J, T)
    period_results: List[PeriodOptimizationResult]


# =========================================================
# 2. Softmax and MNL probability helpers
# =========================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def compute_choice_probs_mnl_personalized(
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    price_mat: np.ndarray,
) -> np.ndarray:
    """
    Compute MNL choice probabilities with outside option utility fixed at 0.

    Parameters
    ----------
    alpha_arr : np.ndarray, shape (N,)
        Customer-specific price sensitivities.
    beta_mat : np.ndarray, shape (N, J)
        Customer-specific product preferences.
    price_mat : np.ndarray, shape (N, J)
        Customer-specific product prices for one period.

    Returns
    -------
    prob_mat : np.ndarray, shape (N, J+1)
        Column 0 is outside option.
        Columns 1..J are product choice probabilities.
    """
    alpha_arr = np.asarray(alpha_arr, dtype=float)
    beta_mat = np.asarray(beta_mat, dtype=float)
    price_mat = np.asarray(price_mat, dtype=float)

    if alpha_arr.ndim != 1:
        raise ValueError("alpha_arr must be 1D.")
    if beta_mat.ndim != 2:
        raise ValueError("beta_mat must be 2D.")
    if price_mat.ndim != 2:
        raise ValueError("price_mat must be 2D.")
    if beta_mat.shape != price_mat.shape:
        raise ValueError("beta_mat and price_mat must have the same shape.")
    if beta_mat.shape[0] != alpha_arr.shape[0]:
        raise ValueError("alpha_arr and beta_mat must match on first dimension.")

    util_products = beta_mat - alpha_arr[:, None] * price_mat
    util_full = np.concatenate(
        [np.zeros((alpha_arr.shape[0], 1)), util_products],
        axis=1,
    )
    return softmax(util_full, axis=1)


# =========================================================
# 3. Expected revenue for A under personalized prices
# =========================================================

def compute_expected_revenue_A_personalized(
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    price_mat: np.ndarray,
    target_product_idx: int = 1,
) -> float:
    """
    Expected revenue for target product under customer-specific prices:

        sum_i p_iA * P_i(A)

    Notes
    -----
    target_product_idx is 1-based over actual products:
    - prob column 0 = outside option
    - prob column 1 = product 1
    - ...
    """
    prob_mat = compute_choice_probs_mnl_personalized(
        alpha_arr=alpha_arr,
        beta_mat=beta_mat,
        price_mat=price_mat,
    )

    j = target_product_idx
    if j < 1 or j > price_mat.shape[1]:
        raise ValueError("target_product_idx must be in {1, ..., J}.")

    price_A_vec = price_mat[:, j - 1]
    prob_A_vec = prob_mat[:, j]

    return float(np.sum(price_A_vec * prob_A_vec))


# =========================================================
# 4. Observed price path extraction (common posted prices by trip)
# =========================================================

def extract_price_path(
    df: pd.DataFrame,
    n_periods: int,
    n_products: int,
    period_col: str = "period",
    price_col_prefix: str = "price_",
) -> np.ndarray:
    """
    Extract observed posted price path of shape (J, T).

    Assumes prices are common across customers within each period.
    """
    if period_col not in df.columns:
        raise ValueError(f"df must contain '{period_col}' column.")

    price_path = np.zeros((n_products, n_periods), dtype=float)

    for t in range(n_periods):
        df_t = df[df[period_col] == t]
        if df_t.empty:
            raise ValueError(f"No rows found for period {t}.")

        for j in range(1, n_products + 1):
            col = f"{price_col_prefix}{j}"
            if col not in df.columns:
                raise ValueError(f"Missing price column: {col}")

            unique_prices = df_t[col].unique()
            if len(unique_prices) != 1:
                raise ValueError(
                    f"Price column {col} has multiple values in period {t}. "
                    "Expected common posted price within each period."
                )

            price_path[j - 1, t] = float(unique_prices[0])

    return price_path


# =========================================================
# 5. Observed realized revenue
# =========================================================

def compute_realized_revenue(
    df: pd.DataFrame,
    target_product_idx: int = 1,
    price_col_prefix: str = "price_",
    choice_col: str = "choice",
) -> float:
    """
    Realized revenue in dataframe:
        sum(price_target over rows where choice == target_product_idx)
    """
    price_col = f"{price_col_prefix}{target_product_idx}"

    if choice_col not in df.columns:
        raise ValueError(f"df must contain '{choice_col}'.")
    if price_col not in df.columns:
        raise ValueError(f"df must contain '{price_col}'.")

    chosen_mask = df[choice_col].to_numpy() == target_product_idx
    return float(df.loc[chosen_mask, price_col].to_numpy(dtype=float).sum())


# =========================================================
# 6. Single-period personalized optimization for A
# =========================================================

def optimize_price_A_t_personalized(
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    observed_price_vec_t: np.ndarray,
    price_grid: np.ndarray,
    target_product_idx: int = 1,
    period: Optional[int] = None,
) -> PeriodOptimizationResult:
    """
    For one period t:
    - B,C,... remain fixed at observed trip-t prices
    - A is optimized separately for each customer over price_grid

    Parameters
    ----------
    alpha_arr : shape (N,)
    beta_mat  : shape (N, J)
    observed_price_vec_t : shape (J,)
        Observed posted prices in trip t, common across customers.
    price_grid : shape (G,)
    """
    alpha_arr = np.asarray(alpha_arr, dtype=float)
    beta_mat = np.asarray(beta_mat, dtype=float)
    observed_price_vec_t = np.asarray(observed_price_vec_t, dtype=float)
    price_grid = np.asarray(price_grid, dtype=float)

    if alpha_arr.ndim != 1:
        raise ValueError("alpha_arr must be 1D.")
    if beta_mat.ndim != 2:
        raise ValueError("beta_mat must be 2D.")
    if beta_mat.shape[0] != alpha_arr.shape[0]:
        raise ValueError("alpha_arr and beta_mat must match on first dimension.")
    if observed_price_vec_t.ndim != 1:
        raise ValueError("observed_price_vec_t must be 1D.")
    if beta_mat.shape[1] != observed_price_vec_t.shape[0]:
        raise ValueError("observed_price_vec_t length must match number of products.")
    if price_grid.ndim != 1 or len(price_grid) == 0:
        raise ValueError("price_grid must be a non-empty 1D array.")

    N, J = beta_mat.shape
    j0 = target_product_idx - 1
    if target_product_idx < 1 or target_product_idx > J:
        raise ValueError("target_product_idx must be in {1, ..., J}.")

    optimal_price_A_vec = np.zeros(N, dtype=float)
    customer_revenue_curve = np.zeros((N, len(price_grid)), dtype=float)

    for i in range(N):
        # customer i sees fixed competitor prices at trip t
        # only A price varies over the grid
        for g, p in enumerate(price_grid):
            price_vec_i = observed_price_vec_t.copy()
            price_vec_i[j0] = p
            price_mat_i = price_vec_i[None, :]  # shape (1, J)

            rev_i = compute_expected_revenue_A_personalized(
                alpha_arr=alpha_arr[i:i+1],
                beta_mat=beta_mat[i:i+1, :],
                price_mat=price_mat_i,
                target_product_idx=target_product_idx,
            )
            customer_revenue_curve[i, g] = rev_i

        best_idx = int(np.argmax(customer_revenue_curve[i, :]))
        optimal_price_A_vec[i] = float(price_grid[best_idx])

    # total expected revenue at optimized personalized prices
    price_mat_opt = np.tile(observed_price_vec_t, (N, 1))
    price_mat_opt[:, j0] = optimal_price_A_vec

    optimal_revenue_A = compute_expected_revenue_A_personalized(
        alpha_arr=alpha_arr,
        beta_mat=beta_mat,
        price_mat=price_mat_opt,
        target_product_idx=target_product_idx,
    )

    return PeriodOptimizationResult(
        period=-1 if period is None else int(period),
        observed_price_vec=observed_price_vec_t.copy(),
        optimal_price_A_vec=optimal_price_A_vec,
        optimal_revenue_A=float(optimal_revenue_A),
        customer_revenue_curve=customer_revenue_curve,
    )


# =========================================================
# 7. Period-by-period personalized optimization
# =========================================================

def optimize_price_path_A_personalized(
    observed_df: pd.DataFrame,
    est_alpha_arr: np.ndarray,
    est_beta_mat: np.ndarray,
    price_grid: np.ndarray,
    n_periods: int,
    n_products: int = 3,
    target_product_idx: int = 1,
    period_col: str = "period",
    price_col_prefix: str = "price_",
) -> List[PeriodOptimizationResult]:
    """
    For each period t:
    - use observed period-t posted price vector
    - keep competitors fixed
    - optimize A separately for each customer
    """
    observed_price_path = extract_price_path(
        df=observed_df,
        n_periods=n_periods,
        n_products=n_products,
        period_col=period_col,
        price_col_prefix=price_col_prefix,
    )

    results: List[PeriodOptimizationResult] = []

    for t in range(n_periods):
        observed_price_vec_t = observed_price_path[:, t]

        result_t = optimize_price_A_t_personalized(
            alpha_arr=est_alpha_arr,
            beta_mat=est_beta_mat,
            observed_price_vec_t=observed_price_vec_t,
            price_grid=price_grid,
            target_product_idx=target_product_idx,
            period=t,
        )
        results.append(result_t)

    return results


# =========================================================
# 8. Build counterfactual personalized price tensor
# =========================================================

def build_cf_price_tensor_personalized(
    period_results: List[PeriodOptimizationResult],
    target_product_idx: int = 1,
) -> np.ndarray:
    """
    Build counterfactual personalized price tensor of shape (N, T, J)

    - A uses personalized customer-trip optimal prices
    - B,C,... remain at observed trip-specific posted prices
    """
    if len(period_results) == 0:
        raise ValueError("period_results is empty.")

    n_periods = len(period_results)
    n_products = len(period_results[0].observed_price_vec)
    n_customers = len(period_results[0].optimal_price_A_vec)

    j0 = target_product_idx - 1
    price_tensor_cf = np.zeros((n_customers, n_periods, n_products), dtype=float)

    for t, res in enumerate(period_results):
        for i in range(n_customers):
            price_vec_it = res.observed_price_vec.copy()
            price_vec_it[j0] = res.optimal_price_A_vec[i]
            price_tensor_cf[i, t, :] = price_vec_it

    return price_tensor_cf


# =========================================================
# 9. DGP with support for common or personalized price schedule
# =========================================================

def generate_multinomial_dgp(
    n_customers: int,
    n_products: int,
    n_periods: int,
    beta_mat: Optional[np.ndarray] = None,
    alpha_arr: Optional[np.ndarray] = None,
    price_schedule: Optional[np.ndarray] = None,
    price_range: Tuple[float, float] = (1.0, 3.0),
    seed: Optional[int] = None,
    error_type: str = "probit",
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Generate panel data from multinomial choice model.

    Utility:
        U_ijt = beta_ij - alpha_i * price_ijt + eps_ijt
        U_i0t = 0 + eps_i0t

    Choice:
        y_it = argmax over {0,1,...,J}

    Parameters
    ----------
    price_schedule : optional
        Either:
        - shape (J, T): common posted prices by trip
        - shape (N, T, J): customer-trip personalized prices
        If None, common posted prices are randomly generated with shape (J, T).
    error_type : {"probit", "logit"}
        probit -> Normal(0,1) errors
        logit  -> Gumbel(0,1) errors
    """
    rng = np.random.default_rng(seed)

    if beta_mat is None:
        beta_mat = rng.normal(loc=1.0, scale=0.5, size=(n_customers, n_products))
    else:
        beta_mat = np.asarray(beta_mat, dtype=float)

    if alpha_arr is None:
        alpha_arr = np.abs(rng.normal(loc=1.0, scale=0.5, size=n_customers))
    else:
        alpha_arr = np.asarray(alpha_arr, dtype=float)

    if beta_mat.shape != (n_customers, n_products):
        raise ValueError("beta_mat must have shape (n_customers, n_products).")
    if alpha_arr.shape != (n_customers,):
        raise ValueError("alpha_arr must have shape (n_customers,).")

    if price_schedule is None:
        price_schedule = rng.uniform(
            price_range[0],
            price_range[1],
            size=(n_products, n_periods),
        )
    else:
        price_schedule = np.asarray(price_schedule, dtype=float)

    common_price_mode = False
    personalized_price_mode = False

    if price_schedule.ndim == 2 and price_schedule.shape == (n_products, n_periods):
        common_price_mode = True
    elif price_schedule.ndim == 3 and price_schedule.shape == (n_customers, n_periods, n_products):
        personalized_price_mode = True
    else:
        raise ValueError(
            "price_schedule must have shape (J, T) or (N, T, J)."
        )

    rows = []

    for i in range(n_customers):
        for t in range(n_periods):
            if common_price_mode:
                price_vec = price_schedule[:, t]
            else:
                price_vec = price_schedule[i, t, :]

            deterministic_util = beta_mat[i, :] - alpha_arr[i] * price_vec

            if error_type == "probit":
                eps0 = rng.normal(loc=0.0, scale=1.0)
                epsj = rng.normal(loc=0.0, scale=1.0, size=n_products)
            elif error_type == "logit":
                # standard Gumbel errors induce multinomial logit choice rule
                eps0 = rng.gumbel(loc=0.0, scale=1.0)
                epsj = rng.gumbel(loc=0.0, scale=1.0, size=n_products)
            else:
                raise ValueError("error_type must be 'probit' or 'logit'.")

            util_full = np.concatenate(
                [[0.0 + eps0], deterministic_util + epsj],
                axis=0,
            )
            choice = int(np.argmax(util_full))

            row = {
                "customer": i,
                "period": t,
                "choice": choice,
            }
            for j in range(n_products):
                row[f"price_{j+1}"] = float(price_vec[j])

            rows.append(row)

    df = pd.DataFrame(rows)

    meta = {
        "alpha_arr": alpha_arr,
        "beta_mat": beta_mat,
        "price_schedule": price_schedule,
    }
    return df, meta


# =========================================================
# 10. Counterfactual revenue simulation
# =========================================================

def simulate_cf_revenue_personalized(
    generate_multinomial_dgp_func,
    true_alpha_arr: np.ndarray,
    true_beta_mat: np.ndarray,
    price_tensor_cf: np.ndarray,
    target_product_idx: int = 1,
    n_rep: int = 100,
    true_error_type: str = "probit",
    base_seed: Optional[int] = None,
    price_col_prefix: str = "price_",
    **dgp_kwargs: Any,
) -> np.ndarray:
    """
    Simulate counterfactual revenue under true DGP using personalized A prices.

    price_tensor_cf shape: (N, T, J)
    """
    true_alpha_arr = np.asarray(true_alpha_arr, dtype=float)
    true_beta_mat = np.asarray(true_beta_mat, dtype=float)
    price_tensor_cf = np.asarray(price_tensor_cf, dtype=float)

    if price_tensor_cf.ndim != 3:
        raise ValueError("price_tensor_cf must be 3D with shape (N, T, J).")

    n_customers, n_periods, n_products = price_tensor_cf.shape

    if true_alpha_arr.shape != (n_customers,):
        raise ValueError("true_alpha_arr shape mismatch.")
    if true_beta_mat.shape != (n_customers, n_products):
        raise ValueError("true_beta_mat shape mismatch.")

    values = []

    for r in range(n_rep):
        seed_r = None if base_seed is None else base_seed + r

        df_cf, _ = generate_multinomial_dgp_func(
            n_customers=n_customers,
            n_products=n_products,
            n_periods=n_periods,
            beta_mat=true_beta_mat,
            alpha_arr=true_alpha_arr,
            price_schedule=price_tensor_cf,
            seed=seed_r,
            error_type=true_error_type,
            **dgp_kwargs,
        )

        revenue_r = compute_realized_revenue(
            df=df_cf,
            target_product_idx=target_product_idx,
            price_col_prefix=price_col_prefix,
            choice_col="choice",
        )
        values.append(revenue_r)

    return np.asarray(values, dtype=float)


# =========================================================
# 11. Full pricing experiment
# =========================================================

def run_pricing_experiment_personalized(
    observed_df: pd.DataFrame,
    generate_multinomial_dgp_func,
    true_alpha_arr: np.ndarray,
    true_beta_mat: np.ndarray,
    est_alpha_arr: np.ndarray,
    est_beta_mat: np.ndarray,
    price_grid: np.ndarray,
    n_periods: int,
    n_products: int = 3,
    target_product_idx: int = 1,
    n_counterfactual_rep: int = 100,
    true_error_type: str = "probit",
    base_seed: Optional[int] = None,
    period_col: str = "period",
    price_col_prefix: str = "price_",
    **dgp_kwargs: Any,
) -> PricingExperimentResult:
    """
    Full personalized pricing pipeline:

    1) Compute observed realized revenue for A from observed data
    2) Optimize customer-trip personalized A prices using fitted MNL
    3) Build counterfactual personalized price tensor
    4) Simulate counterfactual revenue under true DGP
    """
    observed_revenue_A = compute_realized_revenue(
        df=observed_df,
        target_product_idx=target_product_idx,
        price_col_prefix=price_col_prefix,
        choice_col="choice",
    )

    period_results = optimize_price_path_A_personalized(
        observed_df=observed_df,
        est_alpha_arr=est_alpha_arr,
        est_beta_mat=est_beta_mat,
        price_grid=price_grid,
        n_periods=n_periods,
        n_products=n_products,
        target_product_idx=target_product_idx,
        period_col=period_col,
        price_col_prefix=price_col_prefix,
    )

    price_tensor_cf = build_cf_price_tensor_personalized(
        period_results=period_results,
        target_product_idx=target_product_idx,
    )

    cf_values = simulate_cf_revenue_personalized(
        generate_multinomial_dgp_func=generate_multinomial_dgp_func,
        true_alpha_arr=true_alpha_arr,
        true_beta_mat=true_beta_mat,
        price_tensor_cf=price_tensor_cf,
        target_product_idx=target_product_idx,
        n_rep=n_counterfactual_rep,
        true_error_type=true_error_type,
        base_seed=base_seed,
        price_col_prefix=price_col_prefix,
        **dgp_kwargs,
    )

    observed_price_path = extract_price_path(
        df=observed_df,
        n_periods=n_periods,
        n_products=n_products,
        period_col=period_col,
        price_col_prefix=price_col_prefix,
    )

    personalized_price_A = np.column_stack(
        [res.optimal_price_A_vec for res in period_results]
    )  # shape (N, T)

    return PricingExperimentResult(
        observed_revenue_A=float(observed_revenue_A),
        counterfactual_revenue_A_mean=float(cf_values.mean()),
        counterfactual_revenue_A_std=float(cf_values.std(ddof=1)) if len(cf_values) > 1 else 0.0,
        counterfactual_revenue_A_values=cf_values,
        optimal_price_tensor=price_tensor_cf.copy(),
        personalized_price_A=personalized_price_A,
        observed_price_path=observed_price_path.copy(),
        period_results=period_results,
    )
