#!/usr/bin/env python3

"""
Author:     Group 1
Date:       08.02.2022

This code is provided "As Is"
"""

# Third-party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pandas import DataFrame

# Local imports
from analysis import (
    compute_historical_volatility,
    compute_log_returns,
    compute_normalized_log_returns,
    compute_normalized_time_series,
)
from utils import save_figure

# Matplotlib parameters
plt.style.use("seaborn")
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["figure.figsize"] = [6, 4]

class Stats:
    def __init__(self, time_series: DataFrame, normalize: bool = False) -> None:
        """
        Info:
            This class takes one or multiple asset price time series as constructing
            arguments performs a series of operations which, ultimately, retrive
            normalized log returns, mean and mean standard error

        Input:
            time_series: A dataframe containing one (or more) asset-price time series.
            normalize: dictates whether or not to perform normalization on the dataframe

        Returns: an object with attributes containing the desired dataframes
        """
        # Check if data has to be normalized
        if normalize:  # Not clear if this is a worth-while endeavour
            self.time_series = compute_normalized_time_series(time_series)
        else:
            self.time_series = time_series

        # Compute log-returns
        self.log_returns = compute_log_returns(self.time_series)

        # Compute volatility
        self.volatility = compute_historical_volatility(self.log_returns)

        # Compute normalized log returns
        self.norm_log_returns = compute_normalized_log_returns(self.log_returns)

        # Means
        self.ts_mean = self.time_series.mean(axis=1)
        self.lr_mean = self.log_returns.mean(axis=1)
        self.nlr_mean = self.norm_log_returns.mean(axis=1)

        # Standard error
        if self.time_series.shape[1] >= 2:
            self.std_error = stats.sem(self.time_series, axis=1)
        else:
            self.std_error = np.zeros(self.time_series.shape).flatten()


def __test__(normalize: bool):
    """
    When run directly 'python Stats.py' this script runs some default cases to check functionality of this script
    """
    # Import only on main, avoid circular imports
    from data import obtain_BTC, obtain_Gauss, obtain_SP500

    # Experiment Setup
    norm_str = [" Normalized" if normalize else ""]

    # Gaussian
    fig1, axes1 = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    title = f"Gaussian Data{norm_str[0]}"
    fig1.suptitle(title)
    gauss = obtain_Gauss()
    gauss_stats = Stats(gauss, normalize=normalize)
    gauss_stats.ts_mean.plot(title="Time Series (Mean)", ax=axes1[0])
    gauss_stats.nlr_mean.plot(title="Normalized Log Returns (Mean)", ax=axes1[1])
    save_figure(title, "/norm")

    # SP500
    fig2, axes2 = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    title = f"SP500{norm_str[0]}"
    fig2.suptitle(title)
    sp500 = obtain_SP500()
    sp500_stats = Stats(sp500, normalize=normalize)
    sp500_stats.time_series.plot(title="Time Series", ax=axes2[0])
    sp500_stats.norm_log_returns.plot(title="Normalized Log Returns", ax=axes2[1])
    save_figure(title, "/norm")

    # Bitcoin
    fig3, axes3 = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    title = f"BTC{norm_str[0]}"
    fig3.suptitle(title)
    btc = obtain_BTC()
    btc_stats = Stats(btc, normalize=normalize)
    btc_stats.time_series.plot(title="Time Series", ax=axes3[0])
    btc_stats.norm_log_returns.plot(title="Normalized Log Returns", ax=axes3[1])
    save_figure(title, "/norm")

    # Show plots
    # plt.show()


if __name__ == "__main__":
    __test__(normalize=False)
    __test__(normalize=True)
