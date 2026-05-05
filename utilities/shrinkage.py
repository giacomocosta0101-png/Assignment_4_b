from typing import Any, Dict

import numpy as np
import pandas as pd


def constant_corr_shrinkage(
    returns: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate the shrinkage target, the optimal shrinkage intensity and the shrunk covariance
    matrix for the constant correlation shrinkage. No check is done on the input data.
    Implementation follows the Ledoit-Wolf (2003) paper "Honey, I Shrunk the Sample Covariance
    Matrix".

    Parameters:
        returns (pd.DataFrame): The returns matrix (T x N) where T is time periods and N is assets

    Returns:
        Dict[str, Any]: A dictionary containing the shrinkage target matrix ("target"), the optimal
            shrinkage intensity ("intensity"), the sample covariance matrix ("sample_cov") and the
            shrunk covariance matrix ("shrunk_cov").
    """

    cov_matrix = returns.cov()

    T, N = returns.shape  # T = time periods, N = number of assets

    # Convert to numpy arrays for efficiency
    returns_np = returns.values
    S = cov_matrix.values  # Sample covariance matrix

    # Extract standard deviations
    variances = np.diag(S)
    std_devs = np.sqrt(variances) #Fix

    # Compute correlation matrix from covariance matrix
    std_outer = np.outer(std_devs, std_devs)
    corr_matrix = S / std_outer

    # Calculate average correlation
    # Average of off-diagonal correlations
    sum_corr = np.sum(corr_matrix) - N  # subtract diagonal (N ones)
    avg_corr = sum_corr / (N * (N - 1))

    ## Target
    # Calculate target matrix (constant correlation)
    constant_corr_cov = avg_corr*std_outer

    # Set diagonal elements to original variances
    np.fill_diagonal(constant_corr_cov, variances)

    target = pd.DataFrame(
        constant_corr_cov, index=cov_matrix.index, columns=cov_matrix.columns
    )

    ## Intensity
    sample_means = np.mean(returns_np, axis=0)

    # Center the returns
    centered_returns = returns_np - sample_means

    # Calculate pi-hat and rho-hat using vectorized operations
    # Pre-compute matrices for efficiency
    sqrt_ratio_matrix = np.outer(std_devs, 1 / std_devs)  # sqrt(S[i,i]/S[j,j])

    pi_hat = 0.0
    rho_hat = 0.0
    diag_S_matrix = np.outer(variances, np.ones(N))  # S[i,i] for all i,j
    diag_S_matrix_T = diag_S_matrix.T  # S[j,j] for all i,j
    # TODO: Further vectorization possible?
    for t in range(T):
        y_t = centered_returns[t, :]
        outer_prod = np.outer(y_t, y_t)
        diff = outer_prod - S

        # Pi-hat calculation
        pi_hat += np.sum(diff**2)

        # Rho-hat calculation
        # Diagonal terms: (y_t[i]**2 - S[i,i])**2
        diag_terms = y_t**2 - variances
        rho_hat += np.sum(diag_terms**2)

        # Off-diagonal terms
        # Create matrices for vectorized computation
        y_squared_matrix = np.outer(y_t**2, np.ones(N))  # y_t[i]**2 for all i,j
        y_squared_matrix_T = y_squared_matrix.T  # y_t[j]**2 for all i,j

        # Calculate theta terms vectorized
        theta_ii_ij = (y_squared_matrix - diag_S_matrix) * (outer_prod - S)
        theta_jj_ij = (y_squared_matrix_T - diag_S_matrix_T) * (outer_prod - S)

        # Calculate contribution to rho-hat (off-diagonal only)
        off_diag_contrib = (
            avg_corr/2* (sqrt_ratio_matrix.T * theta_ii_ij + sqrt_ratio_matrix * theta_jj_ij)
        )

        # Sum only off-diagonal elements
        np.fill_diagonal(off_diag_contrib, 0)
        rho_hat += np.sum(off_diag_contrib)

    
    pi_hat /= T
    rho_hat /= T

    # Calculate gamma-hat: ||F - S||^2 where F is the target matrix
    gamma_hat = np.sum((target.values - S) ** 2)

    # Calculate optimal shrinkage intensity
    # k = (pi - rho) / gamma, but we need to be careful about numerical stability
    numerator = pi_hat - rho_hat
    denominator = gamma_hat

    if denominator == 0:
        intensity = 0.0
    else:
        # Ensure shrinkage intensity is between 0 and 1
        intensity = max(0.0, min(1.0, numerator / (T * denominator)))

    return {
        "target": target,
        "intensity": intensity,
        "sample_cov": cov_matrix,
        "shrunk_cov": intensity*target + (1-intensity)*cov_matrix,
    }


def market_factor_shrinkage(
    returns: pd.DataFrame, market_returns: pd.Series
) -> Dict[str, Any]:
    """
    Calculate the shrinkage target, the optimal shrinkage intensity and the shrunk covariance
    matrix for the market factor shrinkage. No check is done on the input data.
    Implementation follows the Ledoit-Wolf (2002) paper "Improved estimation of the covariance
    matrix of stock returns with an application to portfolio selection".

    Parameters:
        returns (pd.DataFrame): The returns matrix (T x N) where T is time periods and N is assets
        market_returns (pd.Series): The market returns (T x 1)

    Returns:
        Dict[str, Any]: A dictionary containing the shrinkage target matrix ("target"), the optimal
            shrinkage intensity ("intensity"), the sample covariance matrix ("sample_cov") and the
            shrunk covariance matrix ("shrunk_cov").
    """

    # Align indices to ensure proper calculation
    aligned_data = pd.concat([returns, market_returns], axis=1, join="inner")
    returns_aligned = aligned_data.iloc[:, :-1]
    market_aligned = aligned_data.iloc[:, -1]

    T = returns_aligned.shape[0]

    ## Target
    # Calculate market variance
    market_variance = market_aligned.var()

    # Calculate betas for all assets (vectorized)
    returns_np = returns_aligned.values
    market_np = market_aligned.values

    # Compute covariances between all assets and market at once
    # Stack returns and market, compute covariance matrix, extract asset-market covariances
    combined = np.column_stack([returns_np, market_np])
    cov_matrix_full = np.cov(combined.T)
    cov_with_market = cov_matrix_full[:-1, -1]  # Covariances of each asset with market
    betas = cov_with_market / market_variance

    # Calculate residual variances: Var(asset) - β² * Var(market) (vectorized)
    asset_variances = np.diag(cov_matrix_full[:-1, :-1])
    residual_variances = asset_variances - betas**2 * market_variance

    # Ensure residual variances are positive (vectorized)
    residual_variances = np.maximum(residual_variances, 1e-8)

    # Construct target matrix
    betas_outer = np.outer(betas, betas)
    residual_matrix = np.diag(residual_variances)

    # Final target matrix
    target = pd.DataFrame(
    market_variance * betas_outer + residual_matrix,
    index=returns.columns,
    columns=returns.columns,
    )

    ## Intensity
    cov_matrix = returns_aligned.cov()
    S = cov_matrix.values

    # Center the returns
    sample_means = np.mean(returns_np, axis=0)
    centered_returns = returns_np - sample_means
    market_centered_returns = market_np - market_np.mean()

    # Calculate pi-hat and rho-hat
    # Pre-compute matrices for efficiency
    variances = np.diag(S)
    target_np = target.values

    pi_hat = 0.0
    rho_hat = 0.0

    for t in range(T):
        y_t = centered_returns[t, :]
        outer_prod = np.outer(y_t, y_t)
        diff = outer_prod - S

        # Pi-hat calculation
        pi_hat += np.sum(diff**2)

        # Rho-hat calculation
        m_t = market_centered_returns[t]

        # Diagonal terms: (y_t[i]**2 - S[i,i])**2
        diag_terms = y_t**2 - variances
        rho_hat += np.sum(diag_terms**2)

        # Off-diagonal terms
        # Create matrices: betas[j] * y_t[i] for all i,j
        betas_y_matrix = np.outer(y_t, betas)  # y_t[i] * betas[j]
        betas_y_matrix_T = betas_y_matrix.T  # y_t[j] * betas[i]

        # Calculate the rho-hat contribution for off-diagonal elements
        off_diag_term = (
            betas_y_matrix_T + betas_y_matrix - betas_outer * m_t
        ) * m_t * outer_prod - target_np * S

        # Sum only off-diagonal elements
        np.fill_diagonal(off_diag_term, 0)
        rho_hat += np.sum(off_diag_term)

    rho_hat /= T
    pi_hat /= T

    # Calculate gamma-hat: ||F - S||^2 where F is the target matrix
    gamma_hat = np.sum((target_np - S) ** 2)

    # Calculate optimal shrinkage intensity
    # k = (pi - rho) / gamma, but we need to be careful about numerical stability
    numerator = pi_hat - rho_hat
    denominator = gamma_hat

    if denominator == 0:
        intensity = 0.0
    else:
        intensity = max(0.0, min(1.0, numerator / (T * denominator)))

    return {
        "target": target,
        "intensity": intensity,
        "sample_cov": cov_matrix,
        "shrunk_cov": intensity * target + (1 - intensity) * cov_matrix,
    }
