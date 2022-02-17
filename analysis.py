#!/usr/bin/env python3

"""
Author:     Group 1
Date:       08.02.2022

This code is provided "As Is"
"""

# Third Party
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series

# Local
from data import obtain_SP500
from utils import log

# Matplotlib parameters
plt.style.use("seaborn")
mpl.rcParams["figure.dpi"] = 250


def compute_normalized_time_series(time_series: DataFrame) -> DataFrame:
    """
    Normalizes the data to be between the values of 0 and 1
    Args:
        time_series: the price of an asset pice over time

    Returns: a pandas DataFrame containing the normalized asset price over time
    """
    log("Compute normalized time series")

    # Remove NaN values
    time_series.dropna(inplace=True)

    # Normalize data between 1 and 2 (log-returns doesn't like 0 to 1)
    norm_time_series = (
        time_series.apply(lambda x: (x - x.min()) / (x.max() - x.min())) + 1
    )

    # Make sure return type is DataFrame
    if isinstance(norm_time_series, Series):
        return norm_time_series.to_frame()
    else:
        return norm_time_series


def compute_log_returns(
    time_series: DataFrame, lookback_timeframe: int = 1
) -> DataFrame:
    """
    Computes and returns the log returns of an asset price

    Args:
        time_series: the price of an asset over time
        lookback_timeframe: timeframe on which to calculate the difference for log returns

    Returns: A pandas DataFrame containing the log-returns given an asset time series
    """
    # Different methods to calculate log returns
    log("Compute log returns")

    # Remove NaN values
    time_series.dropna(inplace=True)  # NaN

    # Calculate log-returns
    log_returns = time_series.apply(
        lambda x: np.array(np.log(x / x.shift(lookback_timeframe)))
    ).dropna()

    # Make sure return type is DataFrame
    if isinstance(log_returns, Series):
        return log_returns.to_frame()
    else:
        return log_returns


def compute_normalized_log_returns(log_returns: DataFrame) -> DataFrame:
    """
    Args:
        log_returns: Computed log returns of asset price

    Returns: a pandas DataFrame containing the normalized log-returns
    """
    log("Compute normalized log returns")

    # Remove NaN values
    log_returns.dropna(inplace=True)  # NaN

    # Normalize data between -1 and 1
    norm_log_returns = log_returns.apply(
        lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
    ).dropna()

    # Make sure return type is DataFrame
    if isinstance(norm_log_returns, Series):
        return norm_log_returns.to_frame()
    else:
        return norm_log_returns


def compute_historical_volatility(
    log_returns: DataFrame, lookback_timeframe: int = 1
) -> float:
    """
    Returns the historical volatility of a given price data given it's log-returns

    Input:
        log_returns: dataframe containing the log-returns of the time series price data
        lookback_timeframe: timeframe on which to calculate the difference for log returns

    Returns: volatility of the price data for the given timeframe.
    """
    log("Computing historical volatility of S&P 500 daily price data.")

    # Calculate volatility
    return np.sqrt(len(log_returns)) * np.nanstd(log_returns)


def __test__():
    """
    When run directly 'python analysis.py' this script runs some default cases to check functionality of this script
    """
    # Perform various tests
    df = obtain_SP500()
    df.plot(title="SP500 Price")

    # Compute log-returns
    tst1 = compute_log_returns(df)
    tst1.plot(title="SP500 Log Returns")

    # Compute volatility
    vol = compute_historical_volatility(tst1)
    log("Volatility : {}", vol)

    # Compute normalized log returns
    tst2 = compute_normalized_log_returns(tst1)
    tst2.plot(title="SP500 Normalized Log Returns")

    # Clear Memory
    del df, tst1, tst2

    # Same tests with normalized price
    df = obtain_SP500()
    df = compute_normalized_time_series(df)
    df.plot(title="(N) SP500 Price")

    tst1 = compute_log_returns(df)
    tst1.plot(title="(N) SP500 Log Returns")

    tst2 = compute_normalized_log_returns(tst1)
    tst2.plot(title="(N) SP500 Normalized Log Returns")
    plt.show()


if __name__ == "__main__":
    __test__()
