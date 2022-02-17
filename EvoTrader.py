#!/usr/bin/env python3

"""
Author:     Group 1
Date:       08.02.2022

This code is provided "As Is"
"""

# Default
import random
from copy import deepcopy

# Thirdparty
from mesa import Agent, Model

""" Various notes
- NOTE: "total_assets" in 'buy operation' have to possible calculations which
        are not specified in the original paper:
        1. total_assets = int(total_cost / self.model.asset_price)
        2. total_assets = int(total_cost / max_buy_price)
- NOTE: "trading_cash" in 'buy operation' have to possible calculations which
        1. trading_cash -= total_assets * max_buy_price
        2. trading_cash -= total_assets * self.model.asset_price
- NOTE: For "buy" operations, cash is removed, inversely in "sell" operations,
        assets are removed, but assets aren't added for the first nor cash is for
        the second. This is because there is no guarantee the operations go through,
        allowing for potential impossible bids if, for example, an agent does
        [sell, buy, sell], it may have more assets to sell then he actually would have if
        the middle buy operation gets rejected.
- NOTE: Total assets can't be fractional, therefore but in 'sell' and 'buy' operations
        the value has to be rounded, in this code 'int()' was used, rounding down,
        but 'ceil()' or 'round()' could be equally good
"""


class Trader(Agent):
    """
    The only type of agent in the Genoa model.
    Traders perform trades with a probability 'ops_prob'.
    The trades are buy orders with a probability 'buy_prob',
    or they are sell orders with a probability of '1 - buy_prob'
    """

    def __init__(
        self,
        unique_id: int,
        model: Model,
        ops_prob: float,
        buy_prob: float,
        initial_cash: float,
        initial_assets: int,
    ) -> None:
        super().__init__(unique_id, model)
        self.ops_prob = ops_prob
        self.buy_prob = buy_prob
        self.cash = initial_cash
        self.assets = initial_assets

    def step(self):
        # Start Trade Cycle
        trading_cash = deepcopy(self.cash)
        trading_assets = deepcopy(self.assets)

        # Perform operations
        for __ in range(self.model.n_ops):
            # Decide to make an operation
            if self.ops_prob >= random.uniform(0, 1):
                # Decide to make a "sell" or a "buy" operation
                if self.buy_prob >= random.uniform(0, 1):
                    # Proportion of available cash to buy with on trade
                    total_cost = 0.9 * random.uniform(0, 1) * trading_cash

                    # Maximum amount of cash willing to spend per asset
                    max_buy_price = self.model.asset_price * random.gauss(
                        mu=self.model.mean, sigma=self.model.std
                    )

                    # Actual number of assets to buy
                    total_assets = int(total_cost / max_buy_price)

                    # Check if trader has enough cash to make trade @ max buy price)
                    cash_check = trading_cash - (total_assets * max_buy_price)
                    if cash_check <= 0:
                        total_assets = int(trading_cash / max_buy_price)

                    # Check buy order quantity is non-zero
                    if total_assets >= 1:
                        # Update available cash to buy with
                        trading_cash -= total_assets * max_buy_price

                        # Place buy order to Market
                        # print("Buy:", total_assets, trading_cash, max_buy_price)
                        self.model.buys.append(
                            [
                                self,
                                total_assets,
                                max_buy_price,
                            ]
                        )
                else:  # Decide To Sell
                    # Proportion of available assets to sell on trade
                    total_assets = int(0.9 * random.uniform(0, 1) * trading_assets)

                    # Min on sell orders
                    min_sell_price = self.model.asset_price / random.gauss(
                        mu=self.model.mean, sigma=self.model.std
                    )

                    # Asset check
                    asset_check = trading_assets - total_assets
                    if asset_check <= 0:
                        total_assets = trading_assets

                    # Check sell order quantity is non-zero
                    if total_assets >= 1:
                        # Update available assets to sell
                        trading_assets -= total_assets

                        # Place sell order to Market
                        # print("Sell:", total_assets, trading_assets, min_sell_price)
                        self.model.sells.append(
                            [
                                self,
                                total_assets,
                                min_sell_price,
                            ]
                        )
             # No Trade
            else:
                pass

    def get_self_worth(self, mode: str = "whole"):
        """
        Returns the value of the agent.
        Calculated as: cash + (assets * asset_price)
        """
        if mode == "whole":
            return self.cash + (self.assets * self.model.asset_price)
        elif mode == "unbalanced":
            return self.cash + self.assets
        elif mode == "cash":
            return self.cash
        elif mode == "assets":
            return self.assets
        elif mode == "assets_ext":
            return self.assets * self.model.asset_price
        else:
            raise ValueError
