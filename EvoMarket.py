#!/usr/bin/env python3

"""
Author:     Group 1
Date:       08.02.2022

This code is provided "As Is"
"""

# Base
import random
from operator import itemgetter
from typing import Tuple

# Part of original env
import matplotlib as mpl
import matplotlib.pyplot as plt
from mesa.batchrunner import batch_run
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler
from pandas import DataFrame

# 3rd Party
from tqdm import trange

# Local
from intersect import intersection
from EvoTrader import Trader
from utils import save_figure

"""
NOTE: Should all actors be given the same amount max possible operations per day?
NOTE: Should all actors be given the same amount of initial assets and cash?
NOTE:   |-> If not, should the distribution be uniform or normal?
NOTE: Should orders be lists or dicts?
NOTE: Should initial asset_price be initial_cash / initial_assets
NOTE: Inheritance? Should children get a combination of the money of the parents?

TODO:
- [ ]: Graph number of orders accepted (also as total orders / orders accepted)
- [ ]: Better comments
- [ ]: self implementation of intersection
- [~]: Make 'gets' properties -> not necessary
- [x]: Datacollector
- [x]: Graph lowest and high cash
- [x]: Graph lowest and high asset holder
- [x]: Eliminate broke (zero assets, zero cash) agents
- [x]: probabilistic selection (vs top 20%)
- [x]: EC Fitness Eval after X amount of steps
"""

# Matplotlib parameters
plt.style.use("seaborn")
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["figure.figsize"] = [6, 4]

WHILE_LOOP_TOL = 25
STEPS = 730
BIRTH_MODE = "std"  # "std", "uniform", "gauss"
WORTH_MODE = "whole"  # "whole", "cash", "assets", "assets_ext" "unbalanced"
OP_PROB = 0.1
BUY_PROB = 0.5
EC_ACTIVE = True  # Conditional to start the EC
EC_MODE = "learn"  # "learn", "new"
SELECTION = "combo"  # "top", "pareto", "linear", "combo"
TOP = 0.2  # Percentage of top performers
DAYS_B4_DEATH = 5  # number of eval days [int((STEPS) * 0.01)]


class Market(Model):
    """
    Genoa Market Model
    """

    def __init__(
        self,
        initial_cash: float = 50000,
        initial_assets: float = 1000,
        asset_price: float = 50,
        mutation: float = 0.2,
        std: float = 0.01,
        mean: float = 1.01,
    ) -> None:
        super().__init__()

        # Variables
        self.mutation = mutation
        self.initial_cash = initial_cash
        self.initial_assets = initial_assets

        self.mean = mean
        self.std = std

        # Mixed
        self.asset_price = asset_price
        # print("", mutation, initial_cash, initial_assets, asset_price, "\n")

        # Fixed
        self.n_ops = 10
        self.n_traders = 200
        self.plot_intersect = False

        # Operation lists
        self.buys = []
        self.sells = []

        # Define Scheduler
        self.schedule = BaseScheduler(self)

        # Create Traders
        self.init_population(self.n_traders)

        # Perform data collection
        self.n_days = 0
        self.datacollector = DataCollector(
            {
                "Asset Price": lambda m: self.asset_price,
            }
        )
        self.running = True

    def init_population(self, n: int) -> None:
        """
        Method that provides an easy way of making a bunch of agents at once.
        """
        for __ in range(n):
            if BIRTH_MODE == "std":
                self.new_agent(
                    OP_PROB, BUY_PROB, self.initial_cash, self.initial_assets
                )
            elif BIRTH_MODE == "uniform":
                self.new_agent(
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    self.initial_cash,
                    self.initial_assets,
                )
            elif BIRTH_MODE == "gauss":
                self.new_agent(
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    self.initial_cash,
                    self.initial_assets,
                    gaussian=True,
                )

    def new_agent(
        self,
        ops_prob: float,
        buy_prob: float,
        initial_cash: float,
        initial_assets: int,
        gaussian: bool = False,
    ) -> None:
        """
        Method that creates a new agent, and adds it to the correct scheduler.
        """
        if gaussian:
            kwargs = {
                "ops_prob": ops_prob,
                "buy_prob": buy_prob,
                "initial_cash": abs(
                    random.gauss(mu=initial_cash, sigma=initial_cash * 0.1)
                ),
                "initial_assets": abs(
                    random.gauss(mu=initial_assets, sigma=initial_assets * 0.1)
                ),
            }
        else:
            kwargs = {
                "ops_prob": ops_prob,
                "buy_prob": buy_prob,
                "initial_cash": initial_cash,
                "initial_assets": initial_assets,
            }
        agent = Trader(self.next_id(), self, **kwargs)
        self.schedule.add(agent)

    def sort_orders(self) -> None:
        """
        Info:
            Orders placed by traders (self.buys/sells) to the market have to be sorted by maximum/minimum buy/sell price.
            Sorting a list of list based on 'from operator import itemgetter':
            Uses itemgetter as constructor for the 3rd element of every sublist within self.buys/sells
        Input:
            None, uses class variables
        Output:
            None, updates class variables
        """
        self.buys = sorted(self.buys, key=itemgetter(2))
        self.sells = sorted(self.sells, key=itemgetter(2))

    def pre_process_orders(self) -> None:
        # To Numpy Arrays
        self.n_sells = np.array([item[1] for item in self.sells])
        self.n_buys = np.array([item[1] for item in self.buys])[::-1]

        self.p_sells = np.array([item[2] for item in self.sells]).flatten()
        self.p_buys = np.array([item[2] for item in self.buys])[::-1].flatten()

        # Cumming Them Sums
        self.n_sells_cum = np.cumsum(self.n_sells).flatten()
        self.n_buys_cum = np.cumsum(self.n_buys).flatten()

    def find_intersection(self) -> Tuple[float, float]:
        intersection_coordinate = intersection(
            self.p_sells, self.n_sells_cum, self.p_buys, self.n_buys_cum
        )
        if intersection_coordinate[0].size == 0:
            return (np.array([-1]), np.array([-1]))

        return intersection_coordinate[0], intersection_coordinate[1]

    def plot_orders(
        self, x_intersect: float, y_intersect: float, intersect: bool = True
    ) -> None:
        plt.plot(self.p_sells, self.n_sells_cum, label="Supply")
        plt.plot(self.p_buys, self.n_buys_cum, label="Demand")
        if intersect:
            plt.plot(x_intersect, y_intersect, "ro")
        plt.legend(loc="best")
        save_figure(str(self.n_days), "/intersect")

    def fitness_function(self, num: int, mode: str = "combo"):
        if mode == "top":  # Select top TOP% of individuals
            _top = int(num * TOP)
            weights = [0 if i < _top else 1 for i in range(1, num + 1)]
        elif mode == "pareto":  # Pareto-like weights for selection
            a = 1.0
            m = 1.0
            weights = [
            (a * m**a) / (i ** (a + 1)) for i in range(1, num + 1)
            ]
        elif mode == "linear":  # Linearly-decreasing weights for selection
            weights = np.linspace(0, 1, num)[::-1]
        elif mode == "combo":  # Combo between linear and pareto
            a = 1.0
            m = 1.0
            pareto_w = [
            (a * m**a) / (i ** (a + 1)) for i in range(1, num + 1)
            ]
            linear_w = np.linspace(0, a, num)[::-1]
            weights = (linear_w + pareto_w) / 2
        weights /= np.sum(weights)
        return weights


    def step(self):
        """
        Method that calls the step method for each of the sheep, and then for each of the wolves.
        """
        self.schedule.step()
        self.sort_orders()
        self.pre_process_orders()
        x_intersect, y_intersect = self.find_intersection()

        while_count = 0
        intersect_found = True
        while x_intersect[0] == -1:
            self.buys = []
            self.sells = []
            self.datacollector.collect(self)
            self.schedule.step()
            self.sort_orders()
            self.pre_process_orders()
            x_intersect, __ = self.find_intersection()
            if while_count > WHILE_LOOP_TOL:  # Catch blow-up
                intersect_found = False
                break
            while_count += 1

        # If intersect is found, assign new_asset_price the intersection value
        if intersect_found:
            new_asset_price = x_intersect[0]
            # Save figure of intersections
            if self.plot_intersect:
                self.plot_orders(x_intersect, y_intersect)
        # Else, the new_asset_price stays the same as previous run
        else:
            new_asset_price = self.asset_price

        buys = [buy for buy in self.buys if buy[2] >= new_asset_price]
        for buy_order in buys:
            assert buy_order[2] >= new_asset_price

            new_cash = buy_order[0].cash - buy_order[1] * new_asset_price

            _err = [
                buy_order[0].unique_id,
                buy_order[0].cash,
                new_asset_price,
                buy_order[2],
                buy_order[1] * new_asset_price,
            ]
            assert new_cash > 0, f"BUY: {_err}"

            buy_order[0].cash -= buy_order[1] * new_asset_price
            buy_order[0].assets += buy_order[1]

        sells = [sell for sell in self.sells if sell[2] <= new_asset_price]
        for sell_order in sells:
            assert sell_order[2] <= new_asset_price

            new_assets = sell_order[0].assets - sell_order[1]

            _err = [
                sell_order[0].unique_id,
                sell_order[0].assets,
                new_asset_price,
                sell_order[1],
            ]
            assert new_assets > 0, f"SELL: {_err}"

            sell_order[0].assets -= sell_order[1]
            sell_order[0].cash += sell_order[1] * new_asset_price

        value = [
            [agent, agent.get_self_worth(mode=WORTH_MODE)]
            for agent in self.schedule.agent_buffer()
        ]
        value = sorted(value, key=itemgetter(1))

        #  --- EC ---
        if EC_ACTIVE:
            # Perform fitness evaluation only every DAYS_B4_DEATH amount of days
            if self.n_days % DAYS_B4_DEATH == 0:
                weights = self.fitness_function(len(value), SELECTION)
                agents = [agent for agent, __ in value]
                selection = np.random.choice(agents, len(agents), p=weights)

                # Get genome of selected traders
                genome = [(agent.ops_prob, agent.buy_prob) for agent in selection]

                genome = []
                ids = {}
                for agent in selection:
                    genome.append((agent.ops_prob, agent.buy_prob))
                    if agent.unique_id in ids:
                        ids[agent.unique_id] += 1
                    else:
                        ids[agent.unique_id] = 0

                # ids_items = sorted(list(ids.items()), key=itemgetter(1), reverse=True)
                # keys = [str(i) for i, j in ids_items if j > 0]
                # values = [i for __, i in ids_items if i > 0]
                # plt.hist(keys, weights=values, bins=len(value))
                # plt.xticks(rotation=90, fontsize=4)
                # plt.tight_layout()
                # save_figure(str(self.n_days), f"/ec/selection/{SELECTION}")

                # plt.scatter(np.array(genome)[:, 0], np.array(genome)[:, 1], marker="x")
                # plt.tight_layout()
                # save_figure(str(self.n_days), f"/ec/dist/{SELECTION}")

                # Find removed traders
                discarded = [
                    agent
                    for agent in self.schedule.agent_buffer()
                    if agent not in selection
                ]

                # Update scheduler (crossover + mutation)
                for agent in discarded:
                    cross_over = random.uniform(0, 1)

                    ops_prob_p1, buy_prob_p1 = random.choice(genome)
                    ops_prob_p2, buy_prob_p2 = random.choice(genome)

                    # Weighted Avarage
                    ops_prob = ops_prob_p1 * cross_over + ops_prob_p2 * (1 - cross_over)
                    buy_prob = buy_prob_p1 * cross_over + buy_prob_p2 * (1 - cross_over)

                    if random.uniform(0, 1) >= 1 - self.mutation:  # Chance of mutation
                        ops_prob += random.gauss(mu=0, sigma=0.5)

                    if random.uniform(0, 1) >= 1 - self.mutation:  # Chance of mutation
                        buy_prob += random.gauss(mu=0, sigma=0.5)

                    # Cap mutations between 0 and 1
                    buy_prob = min(1, buy_prob)
                    buy_prob = max(0, buy_prob)
                    ops_prob = min(1, ops_prob)
                    ops_prob = max(0, ops_prob)

                    if EC_MODE == "learn":
                        # Update ("Learning From The Best")
                        self.new_agent(
                            ops_prob=ops_prob,
                            buy_prob=buy_prob,
                            initial_assets=agent.assets,
                            initial_cash=agent.cash,
                        )
                    elif EC_MODE == "new":
                        # Replacement ("A New Start")
                        self.new_agent(ops_prob=ops_prob, buy_prob=buy_prob)

                    # Delete Old
                    try:
                        self.schedule.remove(agent)
                    except KeyError:
                        pass

        # Save the statistics
        self.n_days += 1
        self.datacollector.collect(self)

        # Update market parameter values for next step
        self.asset_price = new_asset_price
        self.buys = []
        self.sells = []

    def run_model(self, step_count=STEPS):
        """
        Method that runs the model for a specific amount of steps.
        """
        for __ in trange(step_count):
            try:
                self.step()
            except ValueError:
                break


def __test__() -> None:
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
        max_steps=steps-1,
        iterations=2,
        number_processes=8,
        data_collection_period=1,
        display_progress=True,
    )
    results = DataFrame(results)["Asset Price"]
    results = results.to_numpy()
    height = results.shape[0]
    results = results.reshape([steps, int(height/steps)])
    results = DataFrame(results)
    print(results)

if __name__ == "__main__":
    __test__()
