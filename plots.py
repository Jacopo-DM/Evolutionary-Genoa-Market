#!/usr/bin/env python3

"""
Author:     Group 1
Date:       08.02.2022

This code is provided "As Is"
"""

# Third Party
from statistics import mode
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Local
from data import obtain_BTC, obtain_BlackSholes, obtain_EvoGenoa, obtain_Gauss, obtain_SP500, obtain_SVJ
from Stats import Stats
from utils import save_figure

# Matplotlib parameters
plt.style.use("seaborn")
mpl.rcParams["figure.dpi"] = 250


def plot_log_returns(ts_stats: Stats, plot_title: str) -> None:
    """
    This function takes a Stats object (created for the ABM course) and produces plots
    of log-returns(also saves them locally)

    Input:
        ts_stats: An object containing data regarding a given asset-price time series
        plot_title: the string that is to be displayed as the title of the output plot.
    """

    # Define plot
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))

    # Plot normalized log returns
    ax[0].plot(
        np.arange(0, ts_stats.nlr_mean.shape[0]),
        ts_stats.nlr_mean,
        linewidth=1,
        color="black",
    )
    ax[0].set(xlabel="Time $(t)$", ylabel="Normalized Log Returns")

    # Plot histogram of normalized log returns and store bins
    hist, bins, _ = ax[1].hist(
        ts_stats.nlr_mean[~np.isnan(ts_stats.nlr_mean)], bins="auto"
    )
    # Evaluate normal probability distribution function given normalized log return data and bins
    norm_pdf = stats.norm.pdf(
        bins,
        np.nanmean(ts_stats.nlr_mean),
        np.nanstd(ts_stats.nlr_mean),
    )
    # Overlay normal probability distribution function
    ax[1].plot(bins, norm_pdf * hist.sum() / norm_pdf.sum(), color="black")
    ax[1].set(xlabel="Log of returns", ylabel="Frequency")

    # Plot embellishments
    fig.suptitle(plot_title)
    plt.tight_layout()
    save_figure(plot_title, "/price_path")


def plot_time_series(ts_stats: Stats, plot_title: str) -> None:
    """
    This function takes a Stats object (created for the ABM course) and produces plots
    of the price path of an asset (also saves them locally)

    Input:
        ts_stats: An object containing data regarding a given asset-price time series
        plot_title: the string that is to be displayed as the title of the output plot.
    """
    # Make plots
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot price line
    ax.plot(
        np.arange(0, len(ts_stats.ts_mean)),
        ts_stats.ts_mean,
        linewidth=1,
        color="black",
        label="Mean Asset Price",
    )

    # Plot standard error overlay
    ax.fill_between(
        np.arange(0, len(ts_stats.ts_mean)),
        ts_stats.ts_mean - ts_stats.std_error,
        ts_stats.ts_mean + ts_stats.std_error,
        alpha=0.5,
        label="Standard Error",
    )

    # Plot embellishments
    ax.set(xlabel="Time $(t)$", ylabel="Price")
    ax.legend(loc="best")
    fig.suptitle(plot_title)
    plt.tight_layout()
    save_figure(plot_title, "/price_path")


def __test__():
    """
    When run directly 'python plots.py' this script runs some default cases to check functionality of this script
    """
    # Gaussian
    gauss = obtain_Gauss()
    gauss_stats = Stats(gauss)
    plot_time_series(gauss_stats, "Gaussian Price Path")
    plot_log_returns(gauss_stats, "Gaussian Log Returns")

    # BTC
    btc = obtain_BTC()
    btc_stats = Stats(btc)
    plot_time_series(btc_stats, "BTC Price Path")
    plot_log_returns(btc_stats, "BTC Log Returns")

    # SP500
    sp500 = obtain_SP500()
    sp500_stats = Stats(sp500)
    plot_time_series(sp500_stats, "SP500 Price Path")
    plot_log_returns(sp500_stats, "SP500 Log Returns")

    # Standard Black-Scholes Model (SP500)
    bs = obtain_BlackSholes(mode="SP500")
    bs_stats = Stats(bs)
    plot_time_series(bs_stats, "Black-Scholes Price Path (SP500)")
    plot_log_returns(bs_stats, "Black-Scholes Log Returns (SP500)")

    # Standard Black-Scholes Model (BTC)
    bs = obtain_BlackSholes(mode="BTC")
    bs_stats = Stats(bs)
    plot_time_series(bs_stats, "Black-Scholes Price Path (BTC)")
    plot_log_returns(bs_stats, "Black-Scholes Log Returns (BTC)")

    # Standard Black-Scholes Model (SP500)
    svj = obtain_SVJ(mode="SP500")
    svj_stats = Stats(svj)
    plot_time_series(svj_stats, "Stochastic Volatility With Jumps Price Path (SP500)")
    plot_log_returns(svj_stats, "Stochastic Volatility With Jumps Log Returns (SP500)")

    # Standard Black-Scholes Model (BTC)
    svj = obtain_SVJ(mode="BTC")
    svj_stats = Stats(svj)
    plot_time_series(svj_stats, "Stochastic Volatility With Jumps Price Path (BTC)")
    plot_log_returns(svj_stats, "Stochastic Volatility With Jumps Log Returns (BTC)")

    # Evolutionary Genoa Model
    evo = obtain_EvoGenoa()
    evo_stats = Stats(evo)
    plot_time_series(evo_stats, "Evolutionary Genoa Price Path")
    plot_log_returns(evo_stats, "Evolutionary Genoa Log Returns")

    # Show Plots
    # plt.show()


if __name__ == "__main__":
    __test__()
