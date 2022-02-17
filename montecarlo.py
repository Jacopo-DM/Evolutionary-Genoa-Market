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
from pandas import DataFrame
from tqdm import trange

# Local
from utils import log

# Matplotlib parameters
plt.style.use("seaborn")
mpl.rcParams["figure.dpi"] = 250


def black_scholes(
    start_price: float,
    mu: float,
    sigma: float,
    num_of_days: float,
    num_paths: int,
    num_time_points: int,
) -> DataFrame:
    """
    Returns a 2D array containing (row-wise) asset price process time series
    as computed using Euler discretization of the solution to the
    Black-Scholes SDE.

    Input:
        start_price: the starting price of the asset.
        mu: the drift of the price process.
        sigma: the volatility of the price process.
        num_of_days: the total simulation time.
        num_paths: the number of asset paths that are to be simulated.
        num_time_points: the number of time points in each asset path.
        lookback_timeframe: the time frame on which log returns will be computed
        by calling the plot_log_returns() function as imported from
        PriceAnalysisFunctions.py.
        plot_title: the string that will be displayed as the title in the plot
        output of the plot_log_returns() function as imported from
        PriceAnalysisFunctions.py.

    Return:
        A 2D array containing a number of columns corresponding to num_paths
        with each row being an asset price time series of size num_time_points+1.
    """
    log("Simulating standard {} asset paths", "Black Scholes")

    delta_t = num_of_days / num_time_points  # compute delta t
    asset_array = np.zeros(num_time_points + 1)  # initialize single asset path array
    asset_array[0] = start_price  # assign initial asset value
    asset_price_matrix = np.zeros(
        (num_paths, num_time_points + 1)
    )  # initialize array of asset paths

    # iterate over all asset paths
    for m in trange(num_paths):
        # iterate over all time points
        for n in range(1, num_time_points + 1):
            # sample from standard normal distribution
            Zn = np.random.normal(0, 1)
            # assign asset value at time point n
            asset_array[n] = asset_array[n - 1] * np.exp(
                (mu - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * Zn
            )
        # store asset path in matrix
        asset_price_matrix[m] = asset_array
    return DataFrame(asset_price_matrix).T


def svj(
    S_0: float,
    mu: float,
    sigma: float,
    T: float,
    kappa: float,
    xi: float,
    rho: float,
    lamda: float,
    num_paths: int,
    num_time_points: int,
) -> list:
    """
    Info:
        Returns a 2D array containing (row-wise) asset price process time series
        as computed using Euler discretization of the solution to the
        stochastic volatility with jumps model SDE.

    Input:
        S_0: the starting price of the asset.
        mu: the drift of the price process.
        sigma: the initial value of the stochastic volatility of the price process.
        T: the total simulation time.
        kappa: the strength of the mean reversion process.
            xi: the volatility parameter of the stochastic volatility.
        rho: the degree of correlation between the random movement of the
            stochastic volatility and the random movement of the asset process itself.
        lamda: the intensity of the random jumps process.
        num_paths: the number of asset paths that are to be simulated.
        num_time_points: the number of time points in each asset path.
        lookback_timeframe: the time frame in number of days on which log returns
            will be computed by calling the plot_log_returns() function as imported from
            PriceAnalysisFunctions.py.
        plot_title: the string that will be displayed as the title in the plot
            output of the plot_log_returns() function as imported from
            PriceAnalysisFunctions.py.
    Output:
        A 2D array containing a number of columns corresponding to num_paths
        with each row being an asset price time series of size num_time_points+1.
    """
    print("Simulating and plotting SVJ model asset paths...")

    delta_t = T / num_time_points  # compute delta t
    theta = sigma  # long-run mean volatility

    # initialize single volatility array
    vol_array = np.zeros(num_time_points + 1)
    # assign initial volatility value equal to half times long-run mean
    vol_array[0] = theta
    # initialize array of volatility paths
    vol_matrix = np.zeros((num_paths, num_time_points + 1))

    # initialize single asset path array
    asset_array = np.zeros(num_time_points + 1)
    # assign initial asset value
    asset_array[0] = S_0
    # initialize array of asset paths
    asset_price_matrix = np.zeros((num_paths, num_time_points + 1))

    # iterate over all asset paths
    for m in trange(num_paths):
        # iterate over all time points
        for n in range(1, num_time_points + 1):
            # sample from standard normal distribution for volatility Brownian motion
            Zn_v = np.random.normal(0, 1)
            # assign volatility value at time point n
            vol_array[n] = (
                vol_array[n - 1]
                + kappa * (theta - max(vol_array[n - 1], 0)) * delta_t
                + xi * np.sqrt(max(vol_array[n - 1], 0)) * Zn_v
            )

            # sample from standard normal distribution
            Zn = np.random.normal(0, 1)
            # asset path Brownian motion correlated with volatility Brownian motion
            Zn_a = rho * Zn_v + np.sqrt(1 - rho**2) * Zn
            J = np.random.normal(0, 1)
            # sample jump from Poisson distribution
            dN = np.random.poisson(lamda * delta_t)
            # assign asset value at time point n
            asset_array[n] = asset_array[n - 1] * (
                np.exp(
                    (mu - lamda * J - 0.5 * max(vol_array[n - 1], 0)) * delta_t
                    + np.sqrt(max(vol_array[n - 1], 0)) * np.sqrt(delta_t) * Zn_a
                )
                + J * dN
            )
        vol_matrix[m] = vol_array
        # store asset path in matrix
        asset_price_matrix[m] = asset_array

    return DataFrame(asset_price_matrix).T


def __test__():
    """
    When run directly 'python montecarlo.py' this script runs some default
    cases to check functionality of this script
    """
    from data import obtain_SP500
    from Stats import Stats

    # Get S&P500 data and set simulation parameters
    sp500 = obtain_SP500()
    sp500_stats = Stats(sp500)

    start_price = float(sp500_stats.time_series.values[0][0])
    sigma = sp500_stats.volatility
    num_time_points = len(sp500_stats.time_series["sp500"])
    print(start_price, sigma, num_time_points)
    num_of_days = num_time_points / 365
    mu = 0.0
    num_paths = 1000

    # Simulate and plot standard Black-Scholes data
    output1 = black_scholes(
        start_price, mu, sigma, num_of_days, num_paths, num_time_points
    )
    print(output1)

    # Simulate and plot Heston model data
    T = num_time_points / 365
    kappa = 4.0
    xi = sigma
    rho = 0.25
    lamda = 1

    output2 = svj(
        start_price, mu, sigma, T, kappa, xi, rho, lamda, num_paths, num_time_points
    )
    print(output2)


if __name__ == "__main__":
    __test__()
