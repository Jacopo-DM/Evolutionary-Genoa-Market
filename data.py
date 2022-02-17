#!/usr/bin/env python3

"""
Author:     Group 1
Date:       08.02.2022

Easily extendible to other datasets!

This code is provided "As Is"
"""

# Default
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from mesa.batchrunner import batch_run

# Third Party
import pandas as pd
import pandas_datareader.data as data
from pandas import DataFrame
from EvoMarket import Market

# Local
from utils import log
from montecarlo import black_scholes, svj


def format_date(date: str, formatting: str = "%Y-%m-%d") -> datetime:
    """Formats date (as string) to correct format for 'pandas DataReader'

    Example:
        "2016-01-01" -> "2016-01-01 00:00:00"

    Args:
        date: A date in string format, eg. "2016-01-01"
        formatting: Specifies the formatting of the 'date' argument

    Returns: A date in the correct format, eg. "2016-01-01 00:00:00"
    """
    return datetime.strptime(date, formatting)


def obtain_data(
    source_name: str, asset_name: str, start_date: str, end_date: str
) -> DataFrame:
    """Generic function which returns a pandas dataframe containing the prices over time of an asset from a source

    Args:
        source_name: Name of data provider, eg. 'fred', pandas specific
        asset_name: Name of asset to query, eg. 'sp500', pandas specific
        start_date: Oldest data to query , eg. "2016-01-01" -> must be YYYY-MM-DD !
        end_date: Newest data to query, eg. "2018-12-31" -> must be YYYY-MM-DD !

    Returns: A pandas dataframe with prices of the asset over the given date range
    """
    # Format dates to correct format
    start = format_date(start_date)
    end = format_date(end_date)

    # Define allowed sources
    sources = {"fred": "Federal Reserve Economic Data", "yahoo": "Yahoo"}

    # Catch unexpected sources
    if source_name not in sources.keys():
        try:
            raise KeyError(f"Illegal argument 'source_name': {source_name}")
        except Exception as error:
            sys.stdout.write(f"Caught other error: {repr(error)}\n")
            raise

    # Print source and asset information to standard output
    log(
        "Obtaining '{}' daily price from '{}' for the period from {} to {}",
        asset_name,
        sources[source_name],
        start,
        end,
    )

    # Check existence of data folder
    dir_name = f"{os.getcwd()}/data"
    Path(dir_name).mkdir(parents=True, exist_ok=True)

    # Formulate file name
    _start = "{:%Y_%m_%d}".format(start)
    _end = "{:%Y_%m_%d}".format(end)
    file_name = f"{dir_name}/{source_name}_{asset_name}_{_start}_{_end}.csv"

    # Locate file locally or retrive data remotely
    if Path(file_name).is_file():
        price = pd.read_csv(file_name)  # Load file
        log("Loading {} from local csv", asset_name)
        assert type(price) == DataFrame
    else:
        # Import historical daily prices from source of asset as pandas DataFrame
        price = data.DataReader(asset_name, source_name, start, end)

        # Remove NaN values
        price.dropna(inplace=True)

        # Save file locally
        price.to_csv(file_name)
        log("Loading {} from remote source: {}", asset_name, source_name)

    # Checks if price isn't type 'TextFileReader'
    assert isinstance(price, DataFrame), "Price dataframe isn't the correct type!"

    # Drop the "DATE" column if it exists in the data
    if "DATE" in price:
        price.drop(columns=["DATE"], inplace=True)

    # Remove NaN values
    price.dropna(inplace=True)

    # Notify user of completion
    log("PROCESS FINISHED: price of {} retrived", asset_name)
    return price


def obtain_SP500(
    start_date: str = "2016-01-01", end_date: str = "2018-12-31"
) -> DataFrame:
    """Abstracted method to return Standard and Poor's 500 Data

    Args:
        start_date: Oldest data to query , eg. "2016-01-01" -> must be YYYY-MM-DD !
        end_date: Newest data to query, eg. "2018-12-31" -> must be YYYY-MM-DD !

    Returns: A pandas dataframe with prices of the asset over the given date range
    """
    source_name = "fred"
    asset_name = "sp500"
    return obtain_data(source_name, asset_name, start_date, end_date)


def obtain_BTC(
    start_date: str = "2019-01-01", end_date: str = "2021-12-31"
) -> DataFrame:
    """Abstracted method to return Bitcoin Data

    Args:
        start_date: Oldest data to query , eg. "2019-01-01" -> must be YYYY-MM-DD !
        end_date: Newest data to query, eg. "2021-12-31" -> must be YYYY-MM-DD !

    Returns: A pandas dataframe with prices of the asset over the given date range
    """
    source_name = "yahoo"
    asset_name = "BTC-USD"
    btc_df = obtain_data(source_name, asset_name, start_date, end_date)
    return btc_df.loc[:, ["Close"]]


def obtain_BlackSholes(mode: str = "SP500"):
    if mode == "SP500":
        # Model after SP500
        start_price = 2012.66
        sigma = 0.22458654712530263
        num_time_points = 754
    elif mode == "BTC":
        # Model after BTC-USD
        start_price = 3843.52001953125
        sigma = 1.2982005749014607
        num_time_points = 1097
    num_of_days = num_time_points / 365
    mu = 0.0
    num_paths = 1000

    # Simulate and plot standard Black-Scholes data
    return black_scholes(
        start_price, mu, sigma, num_of_days, num_paths, num_time_points
    )


def obtain_SVJ(mode="SP500"):
    if mode == "SP500":
        # Model after SP500
        start_price = 2012.66
        sigma = 0.22458654712530263
        num_time_points = 754
    elif mode == "BTC":
        # Model after BTC-USD
        start_price = 3843.52001953125
        sigma = 1.2982005749014607
        num_time_points = 1097
    mu = 0.0
    num_paths = 1000
    T = num_time_points / 365
    kappa = 4.0
    xi = sigma
    rho = 0.25
    lamda = 1

    # Simulate and plot Heston model data
    return svj(
        start_price, mu, sigma, T, kappa, xi, rho, lamda, num_paths, num_time_points
    )


def obtain_EvoGenoa():
    params = {
        "initial_cash": 50000,
        "initial_assets": 1000,
        "mutation": 0.2,
        "std": 0.01,
        "asset_price": 50,
    }

    # Experiment parameters
    steps = 730
    results = batch_run(
        Market,
        parameters=params,
        max_steps=steps - 1,
        iterations=25,
        number_processes=8,
        data_collection_period=1,
        display_progress=True,
    )
    results = DataFrame(results)["Asset Price"]
    results = results.to_numpy()
    height = results.shape[0]
    results = results.reshape([steps, int(height / steps)])
    return DataFrame(results)


def obtain_Gauss(
    num_of_days: int = 1000,
    mean: float = 0,
    std: float = 0.1,
    experiments: int = 100,
    seed: int = 42,
) -> DataFrame:
    """Function that returns gaussian data

    Args:
        num_of_days: size of random dataset
        mean: mean of the gaussian from which to sample

    Returns: A pandas dataframe with artificial gaussian prices
    """
    log(
        "Generate {} data for {} days (mean: {}, std: {})",
        "Gaussian",
        num_of_days,
        mean,
        std,
    )
    # Seed seed for repeatability
    random.seed(seed)

    # Produce random numbers
    gauss_df = []
    for i in range(experiments):
        gauss = []
        _init = (
            random.gauss(mean, std) + 10
        )  # 10 is adjustment to avoid negative values
        for _ in range(num_of_days):
            gauss.append(_init)
            _init += random.gauss(mean, std)
        gauss_df.append(DataFrame(gauss, columns=[i]))
    return pd.concat(gauss_df, axis=1)


def __test__():
    """
    When run directly 'python data.py' this script runs some default cases to check functionality of this script
    """
    # Get SP500 data
    obtain_SP500()

    # Get BTC data
    obtain_BTC()

    # Get Gauss data
    obtain_Gauss()


if __name__ == "__main__":
    __test__()
