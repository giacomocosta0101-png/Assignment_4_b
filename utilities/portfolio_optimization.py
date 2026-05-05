"""
Portfolio optimization utilities.

Implements:
- Minimum variance portfolio (closed-form solution)
- Mean-variance portfolio (closed-form solution)
- Equal risk contribution portfolio (optimization-based)

References:
- Maillard, S., Roncalli, T., & Teïletche, J. (2010).
  "The properties of equally weighted risk contribution portfolios."
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
"""

from typing import Any
import numpy as np
from functools import partial
from scipy.optimize import minimize
from utilities.covariance_utilities import (
    _validate_covariance_matrix,
    risk_contribution,
)

def minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the minimum variance portfolio weights given a covariance matrix.
    In particular the weights are given by:
    w = (Σ * 1) / (1^T * Σ * 1), i.e. the solution of the optimization problem:
    min_w w^T * Σ * w, subject to 1^T * w = 1.

    Parameters:
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        np.ndarray: Weights of the minimum variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    n = cov_matrix.shape[0]
    ones_vec = np.ones((n, 1))

    min_var_ptf_numerator = np.linalg.solve(cov_matrix, ones_vec)

    C=(ones_vec.T @ min_var_ptf_numerator)

    min_var_ptf_weights = min_var_ptf_numerator / C
    return min_var_ptf_weights.flatten()


def mean_variance_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Calculate the classic mean-variance portfolio weights given expected returns and a
    covariance matrix.

    In particular the weights solve:
    max_w mu^T * w - (gamma / 2) * w^T * Sigma * w, subject to 1^T * w = 1,
    where mu are the expected returns and gamma is the risk-aversion parameter.

    Parameters:
        expected_returns (np.ndarray): Expected returns vector.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_aversion (float): Risk-aversion parameter gamma. Must be strictly positive.

    Returns:
        np.ndarray: Weights of the mean-variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    expected_returns = np.asarray(expected_returns, dtype=float)
    if expected_returns.ndim == 2 and 1 in expected_returns.shape:
        expected_returns = expected_returns.reshape(-1)
    elif expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be one-dimensional or a single-column vector"
        )

    if expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError(
            "expected_returns and cov_matrix must refer to the same number of assets, "
            f"got {expected_returns.shape[0]} and {cov_matrix.shape[0]}"
        )

    if not np.isfinite(expected_returns).all():
        raise ValueError("expected_returns contains NaN or Inf values")

    if not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be finite")

    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be strictly positive, got {risk_aversion}"
        )

    n = cov_matrix.shape[0]
    ones_vec = np.ones(n)
   
    inv_cov_mu = np.linalg.solve(cov_matrix, expected_returns)
    inv_cov_ones = np.linalg.solve(cov_matrix, ones_vec)

    mv_unconstrained = (1.0 / risk_aversion) * inv_cov_mu
    w_min_var = inv_cov_ones / (ones_vec @ inv_cov_ones)
    
    # Adjust to satisfy sum-to-one constraint
    mean_var_ptf_weights = w_min_var + mv_unconstrained - (ones_vec @ mv_unconstrained) * w_min_var

    return mean_var_ptf_weights.flatten()


def inverse_volatility_portfolio(covariance: np.ndarray) -> np.ndarray:
    """
    Compute the inverse-volatility (naive risk-parity) portfolio.

    Each weight is proportional to the inverse of the asset's volatility,
    ``w_i ∝ 1 / sigma_i``, with ``sum_i w_i = 1``. Lab notes Question 2 shows
    that this coincides with the ERC solution when the assets are uncorrelated.

    Parameters:
        covariance (np.ndarray): Covariance matrix of asset returns. Only the
            diagonal is used.

    Returns:
        np.ndarray: Inverse-volatility weights summing to 1.
    """

    covariance = _validate_covariance_matrix(
        covariance,
        name="covariance",
        require_positive_definite=True,
        positive_definite_message=(
            "covariance must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    vols = np.sqrt(np.diag(covariance))
    inv_vols = 1/vols

    weights_inv_vol = inv_vols/np.sum(inv_vols)

    return weights_inv_vol


def erc_objective_function(weights: np.ndarray, covariance: np.ndarray) -> float:
    """
    Equal risk contribution objective function implemented as the variance of the risk
    contributions. Minimizing this function leads to equal risk contributions across assets.

    Parameters:
        weights (np.ndarray): Portfolio weights.
        covariance (np.ndarray): Covariance matrix of asset returns.

    Returns:
        float: Objective function value.
    """

    non_normalized_risk_contributions = (
        np.multiply(weights.dot(covariance), weights)
    ).reshape(-1, 1)

    return len(non_normalized_risk_contributions) * np.sum(
        np.square(non_normalized_risk_contributions)
    ) - np.sum(non_normalized_risk_contributions @ non_normalized_risk_contributions.T)


def equal_risk_contribution_portfolio(
    covariance: np.ndarray,
    initial_solution: np.ndarray | None = None,
    options: dict[str, Any] | None = None,
    pcr_tolerance: float = 0.001,
    ignore_objective: bool = False,
) -> np.ndarray:
    covariance = _validate_covariance_matrix(
        covariance,
        name="covariance",
        require_positive_definite=True,
        positive_definite_message=(
            "covariance must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    n = covariance.shape[0]

    # ERC is invariant to positive rescalings of Sigma — rescale to keep the
    # objective at O(1) so SLSQP's ftol is meaningful
    scale_factor = np.trace(covariance) / n
    covariance_scaled = covariance / scale_factor

    if initial_solution is None:
        x0 = inverse_volatility_portfolio(covariance_scaled)
    else:
        x0 = np.asarray(initial_solution, dtype=float).flatten()
        if x0.shape[0] != n:
            raise ValueError(
                "initial_solution and covariance must refer to the same number of assets, "
                f"got {x0.shape[0]} and {n}"
            )

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))

    solver_options = {"ftol": 1e-12, "maxiter": 2000, "disp": False}
    if options is not None:
        solver_options.update(options)

    result = minimize(
        erc_objective_function,
        x0,
        args=(covariance_scaled,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=solver_options,
    )

    weights = result.x
    weights = weights / weights.sum()

    if not ignore_objective:
        if not result.success:
            raise RuntimeError(
                f"ERC optimization failed to converge: {result.message}"
            )

        # Sanity check on the *original* covariance — PCRs are scale-invariant anyway
        rc = risk_contribution(weights, covariance)
        pcr = rc / rc.sum()
        pcr_dispersion = pcr.max() - pcr.min()
        if pcr_dispersion > pcr_tolerance:
            raise RuntimeError(
                f"ERC PCR dispersion {pcr_dispersion:.4g} exceeds tolerance {pcr_tolerance}"
            )

    return weights
