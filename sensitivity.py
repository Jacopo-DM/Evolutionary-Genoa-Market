#!/usr/bin/env python3

"""
Author:     Group 1
Date:       08.02.2022

This code is provided "As Is"
"""
# Default
import pickle

# Thirdparty
from mesa.batchrunner import batch_run
from pandas import DataFrame
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import numpy as np

# Local
from EvoMarket import Market


def Sobol():
    # Variables to experiment over
    problem = {
        "num_vars": 5,
        "names": ["initial_cash", "initial_assets", "mutation", "std", "asset_price"],
        "bounds": [[5000, 500000], [100, 10000], [0, 1], [0, 1], [1, 5000]],
    }

    replicates = 50
    max_steps = 730  # 2 years
    distinct_samples = 2**6  # convergence property

    # We get all our samples here
    param_values = saltelli.sample(problem, distinct_samples)

    print(f"== {len(param_values)} ==")
    for i, __ in enumerate(param_values):
        if i <= 135:
            continue
        params = {}
        params[i] = {}
        for j, var in enumerate(problem["names"]):
            params[i][var] = param_values[i][j]
        print(f"-- {i} --: {params[i]}")
        results = batch_run(
            Market,
            parameters=params[i],
            max_steps=max_steps,
            iterations=replicates,
            number_processes=8,
            data_collection_period=1,
            display_progress=True,
        )
        file_name = (
            f"./data/sobol/Sobol_{i}_r{replicates}_m{max_steps}_s{distinct_samples}.csv"
        )
        DataFrame(results).to_csv(file_name)


def OFAT():
    # Variables to experiment over
    problem = {
        "num_vars": 5,
        "names": ["initial_cash", "initial_assets", "mutation", "std", "asset_price"],
        "bounds": [[5000, 500000], [100, 10000], [0, 1], [0, 1], [1, 5000]],
    }

    # Experiment parameters
    replicates = 50
    max_steps = 730  # 2 years
    distinct_samples = 2**6  # convergence property

    for i, var in enumerate(problem["names"]):
        samples = list(
            np.round(np.linspace(*problem["bounds"][i], num=distinct_samples), 3)
        )
        params = {}
        params[var] = samples

        results = batch_run(
            Market,
            parameters=params,
            max_steps=max_steps,
            iterations=replicates,
            number_processes=8,
            data_collection_period=1,
            display_progress=True,
        )

        file_name = (
            f"./data/ofat/OFAT_{var}_r{replicates}_m{max_steps}_s{distinct_samples}.csv"
        )
        DataFrame(results).to_csv(file_name)


if __name__ == "__main__":
    OFAT()
    Sobol()
