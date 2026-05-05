import pandas as pd


def _align_portfolios_and_returns(
    portfolios: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align a portfolio schedule with the return matrix on the common investable universe.

    Parameters:
        portfolios (pd.DataFrame): Portfolio weights indexed by rebalance date.
        returns (pd.DataFrame): Asset returns indexed by trading date.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The aligned portfolio and return matrices.
    """
    if portfolios.empty or portfolios.shape[1] == 0:
        raise ValueError("portfolios must contain at least one asset")

    overlapping_assets = portfolios.columns.intersection(returns.columns)
    if overlapping_assets.empty:
        raise ValueError("portfolios and returns must share at least one asset")

    aligned_portfolios = (
        portfolios.loc[:, overlapping_assets]
        .reindex(returns.index)
        .ffill()
        .shift(1)
        .dropna(axis=0, how="all")
    )
    if aligned_portfolios.empty:
        raise ValueError(
            "portfolios and returns must overlap on at least one investable date"
        )

    aligned_returns = returns.loc[aligned_portfolios.index, overlapping_assets]

    return aligned_portfolios.fillna(0.0), aligned_returns


def portfolio_returns(portfolios: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Compute the time series of realized portfolio returns from a rebalance schedule.

    Parameters:
        portfolios (pd.DataFrame): DataFrame where each column represents the weights of a
            portfolio over time (dates as index).
        returns (pd.DataFrame): DataFrame of asset returns (dates as index, assets as columns).

    Returns:
        pd.Series: Series of realized portfolio returns over time.
    """
    aligned_portfolios, aligned_returns = _align_portfolios_and_returns(
        portfolios=portfolios,
        returns=returns,
    )

    return aligned_portfolios.multiply(aligned_returns).sum(axis=1)


def backtest(portfolios: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative portfolio performance from a rebalance schedule.

    Parameters:
        portfolios (pd.DataFrame): DataFrame where each column represents the weights of a
            portfolio over time (dates as index).
        returns (pd.DataFrame): DataFrame of asset returns (dates as index, assets as columns).

    Returns:
        pd.Series: Series representing the cumulative returns of the portfolios over time.
    """
    return portfolio_returns(portfolios=portfolios, returns=returns).add(1).cumprod()
