from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np

@dataclass(frozen=True)
class CoeffDraws:
    alpha_arr: np.ndarray   # (N,)
    beta_mat: np.ndarray    # (N, J)
    mu_beta_j: np.ndarray   # (J,)
    meta: Dict[str, Any] 

def sample_random_coefficients(
    N: int,
    J: int,
    *,
    mu_alpha: float,
    sd_alpha: float,
    mu_beta_range: Tuple[float, float],
    sd_beta: float,
    seed: int | None = None,
    normalize_beta0: bool = True,
) -> CoeffDraws:
    """
    한국어:
    - 고객별 alpha_i ~ N(mu_alpha, sd_alpha)
    - 제품별 평균 mu_beta_j ~ Uniform(mu_beta_range)
    - 고객×제품 beta_ij = mu_beta_j + Normal(0, sd_beta)
    - 식별을 위해 beta_{i,0}=0으로 강제(normalize_beta0=True) 가능

    English:
    - alpha_i ~ N(mu_alpha, sd_alpha)
    - mu_beta_j ~ Uniform(mu_beta_range)
    - beta_ij = mu_beta_j + Normal(0, sd_beta)
    - Optional identification: force beta_{i,0}=0
    """
    if J < 2:
        raise ValueError("J must be >= 2")
    rng = np.random.default_rng(seed)

    alpha_arr = rng.normal(mu_alpha, sd_alpha, size=N)           # (N,)
    mu_beta_j = rng.uniform(mu_beta_range[0], mu_beta_range[1], size=J)   # (J,)
    beta_mat = rng.normal(0.0, sd_beta, size=(N, J)) + mu_beta_j          # (N,J)

    if normalize_beta0:
        beta_mat[:, 0] = 0.0
        mu_beta_j[0] = 0.0

    meta = {
        "seed": seed,
        "mu_alpha": mu_alpha,
        "sd_alpha": sd_alpha,
        "mu_beta_range": mu_beta_range,
        "sd_beta": sd_beta,
        "normalize_beta0": normalize_beta0,
        "N": N,
        "J": J,
    }

    return CoeffDraws(alpha_arr=alpha_arr, beta_mat=beta_mat, mu_beta_j=mu_beta_j, meta=meta)