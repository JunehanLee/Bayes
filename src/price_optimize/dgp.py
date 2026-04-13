"""
Data generation module for multinomial discrete choice experiments.

This module provides a function to generate panel data for a multinomial
probit discrete choice model with correlated errors.  The data includes
prices for each product and the chosen product index for each
customer/time observation.

The generated dataset can be used as an oracle baseline for evaluating
misspecified models (e.g. multinomial logit) and for computing
optimal pricing policies.
"""

from __future__ import annotations
from xml.parsers.expat import errors

import numpy as np
import pandas as pd
# import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def generate_multinomial_dgp(
    n_customers: int,
    n_products: int,
    n_periods: int,
    beta_mat: Optional[np.ndarray] = None,
    alpha_arr: Optional[np.ndarray] = None,
    corr: float = 0.0,
    price_range: Tuple[float, float] = (1.0, 10.0),
    seed: Optional[int] = None,
    error_type: str = "probit",
    price_schedule: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Generate a panel dataset from a multinomial probit / logit model.

    Parameters
    ----------
    n_customers : int
        Number of distinct customers in the panel.
    n_products : int
        Number of alternatives (products).  Must be >= 2.
    n_periods : int
        Number of periods (time points) per customer.
    beta : np.ndarray or None, optional
        Array of length n_products containing the true intercepts for each
        product.  If None, defaults to a linearly spaced vector
        ``[0,1,...,n_products-1]``.  Intercepts for the baseline product
        (indexed 0) can be set to zero by convention.
    alpha : float, optional
        Price sensitivity (common across products and customers).  The
        utility is specified as ``U[j] = beta[j] - alpha * price[j] + error[j]``.
    corr : float, optional
        Pairwise correlation coefficient for the multivariate normal
        error terms.  Must satisfy ``-1/(n_products-1) < corr < 1`` to
        produce a positive-definite covariance matrix.
    price_range : tuple of float, optional
        Lower and upper bounds of the uniform distribution from which
        product prices are drawn.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    data : pandas.DataFrame
        A DataFrame with columns:

        - ``customer``: customer index (0-based)
        - ``period``: period index (0-based)
        - ``choice``: index of the chosen product (0-based)
        - ``price_0, price_1, ..., price_{J-1}``: price of each product

    true_params : dict
        Dictionary containing the true ``beta`` and ``alpha`` used in
        generating the data, and the correlation ``corr``.

    Notes
    -----
    The covariance matrix for the multivariate normal errors is
    constructed so that all off-diagonal elements equal ``corr`` and
    diagonal elements equal 1.  Utilities are computed for each product
    and the product with the maximum utility is chosen.
    """
    # Validate inputs
    if n_products < 2:
        raise ValueError("n_products must be at least 2")
    if beta_mat.shape != (n_customers, n_products):
        raise ValueError(f"beta_mat must have shape (N,J)={(n_customers, n_products)}")
    if alpha_arr.shape != (n_customers,):
        raise ValueError("alpha_arr must have shape (N,)")
    if corr <= -1.0 / (n_products - 1) or corr >= 1.0:
        raise ValueError(
            f"corr must be in (-1/(J-1), 1) for positive-definiteness; got {corr}"
        )
    rng = np.random.default_rng(seed)
    price_low, price_high = price_range
    if price_schedule is None:
        price_low, price_high = price_range
        price_schedule = rng.uniform(
        price_low,
        price_high,
        size=(n_products, n_periods),
        )
    else:
        price_schedule = np.asarray(price_schedule)
        if not (
            (price_schedule.ndim == 2 and price_schedule.shape == (n_products, n_periods))
            or
            (price_schedule.ndim == 3 and price_schedule.shape == (n_customers, n_periods, n_products))
        ):
            raise ValueError(
                f"price_schedule must have shape {(n_products, n_periods)} "
                f"or {(n_customers, n_periods, n_products)}; got {price_schedule.shape}"
            )

    if error_type == "probit":
        if corr <= -1.0 / (n_products - 1) or corr >= 1.0:
            raise ValueError(
                f"corr must be in (-1/(J-1), 1); got {corr}"
            )
        Sigma = np.full((n_products, n_products), corr)
        np.fill_diagonal(Sigma, 1.0)
        chol = np.linalg.cholesky(Sigma)

    records = []
    for i in range(n_customers):
        alpha_i = float(alpha_arr[i])
        beta_i = beta_mat[i]

        for t in range(n_periods):
            if price_schedule.ndim == 2:
                prices_t = price_schedule[:, t]
            else:
                prices_t = price_schedule[i, t, :]

            if error_type == "probit":
                # product errors
                z = rng.normal(size=n_products)
                errors_prod = chol @ z
                # separate outside error
                error_outside = rng.normal(loc=0.0, scale=1.0)

            elif error_type == "logit":
                errors_prod = rng.gumbel(loc=0.0, scale=0.78, size=n_products)
                error_outside = rng.gumbel(loc=0.0, scale=0.78)

            else:
                raise ValueError("error_type must be 'probit' or 'logit'")

            utilities = beta_i - alpha_i * prices_t + errors_prod
            u0 = error_outside

            # choose among outside(0) + products(1..J)
            choice = int(np.argmax(np.concatenate(([u0], utilities))))  # 0..J
            # Record observation
            row = {
                "customer": i,
                "period": t,
                "choice": choice,
            }
            for j in range(n_products):
                row[f"price_{j+1}"] = float(prices_t[j])
            records.append(row)
    # Build DataFrame
    data = pd.DataFrame.from_records(records)
    return data, {"beta": beta_mat, "alpha": alpha_arr, "corr": corr}


def export_dataframe_to_csv(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    index: bool = False,
    encoding: str = "utf-8",
    float_format: Optional[str] = None,
    make_dirs: bool = True,
) -> Path:
    """
    Save a DataFrame to a CSV file.

    한국어:
    - df를 CSV로 저장한다.
    - out_path의 부모 디렉토리가 없으면 자동 생성(make_dirs=True).
    - index는 기본 False (재현성: 불필요한 인덱스 컬럼 방지).

    English:
    - Save df to CSV.
    - Creates parent directories automatically if make_dirs=True.
    - index defaults to False to avoid accidental extra columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    out_path : str | Path
        Output file path (should end with .csv).
    index : bool
        Whether to write row index.
    encoding : str
        File encoding.
    float_format : str | None
        Optional float formatting (e.g., '%.6f').
    make_dirs : bool
        Create parent directories if not exist.

    Returns
    -------
    Path
        The resolved path to the saved CSV file.
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".csv":
        raise ValueError(f"out_path must end with .csv, got: {out_path}")

    if make_dirs:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=index, encoding=encoding, float_format=float_format)
    return out_path.resolve()


def export_dataframe_to_csv(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    index: bool = False,
    encoding: str = "utf-8",
    float_format: Optional[str] = None,
    make_dirs: bool = True,
) -> Path:
    """
    Save a DataFrame to a CSV file.

    한국어:
    - df를 CSV로 저장한다.
    - out_path의 부모 디렉토리가 없으면 자동 생성(make_dirs=True).
    - index는 기본 False (재현성: 불필요한 인덱스 컬럼 방지).

    English:
    - Save df to CSV.
    - Creates parent directories automatically if make_dirs=True.
    - index defaults to False to avoid accidental extra columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    out_path : str | Path
        Output file path (should end with .csv).
    index : bool
        Whether to write row index.
    encoding : str
        File encoding.
    float_format : str | None
        Optional float formatting (e.g., '%.6f').
    make_dirs : bool
        Create parent directories if not exist.

    Returns
    -------
    Path
        The resolved path to the saved CSV file.
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".csv":
        raise ValueError(f"out_path must end with .csv, got: {out_path}")

    if make_dirs:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=index, encoding=encoding, float_format=float_format)
    return out_path.resolve()


# def export_metadata_to_json(
#     metadata: Dict[str, Any],
#     out_path: str | Path,
#     *,
#     ensure_ascii: bool = False,
#     indent: int = 2,
#     make_dirs: bool = True,
# ) -> Path:
#     """
#     Save experiment metadata to a JSON file.

#     한국어:
#     - 실험 파라미터/seed/설정 등을 JSON으로 저장.
#     English:
#     - Save experiment params/seeds/configs as JSON.

#     Notes:
#     - numpy types 등이 있으면 json 직렬화가 실패할 수 있으니,
#       필요하면 caller에서 float/int 변환해주는 것을 권장.

#     Returns
#     -------
#     Path
#         The resolved path to the saved JSON file.
#     """
#     out_path = Path(out_path)
#     if out_path.suffix.lower() != ".json":
#         raise ValueError(f"out_path must end with .json, got: {out_path}")

#     if make_dirs:
#         out_path.parent.mkdir(parents=True, exist_ok=True)

#     with out_path.open("w", encoding="utf-8") as f:
#         json.dump(metadata, f, ensure_ascii=ensure_ascii, indent=indent)
#     return out_path.resolve()
