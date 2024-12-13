# region imports
from AlgorithmImports import *
# endregion
import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx


class TradingGraphEnvironment:
    def __init__(
        self,
        historical_data,  # A structure holding daily market data (prices, alphas, factor data)
        start_index=0,
        end_index=None,
        window_size=30,
    ):
        """_summary_

        Args:
            historical_data: _description_
            end_index: _description_. Defaults to None.
            window_size: _description_. Defaults to 30.
        """
        self.historical_data = historical_data
        self.start_index = start_index
        self.current_step = start_index
        self.end_index = (
            end_index if end_index is not None else len(historical_data) - 1
        )
        self.window_size = window_size

        # Portfolio state: Initialize with equal weights or empty portfolio.
        # This could be stored as a dictionary {symbol: holdings} or a NumPy array.
        self.portfolio = {}

    def reset(self):
        """Reset the environment to the start of a new episode."""
        self.current_step = self.start_index
        self.portfolio = {}  # reset portfolio (if needed)
        initial_graph = self._construct_graph(self.current_step)
        return initial_graph

    def step(self, action):
        """_summary_

        Args:
            action: _description_

        Returns:
            _description_
        """
        # 1. Apply the action to update the portfolio holdings
        self._apply_action(action)

        # 2. Move to the next time step
        self.current_step += 1

        # 3. Compute the reward as change in portfolio value from last step
        reward = self._compute_reward()

        # 4. Check if we are done (e.g., reached the end of the historical data)
        done = self.current_step >= self.end_index

        # 5. Construct the next state (pyg_graph)
        next_graph = self._construct_graph(self.current_step)

        info = {}  # Additional info if needed
        return next_graph, reward, done, info

    def _apply_action(self, action):
        """
        Update the portfolio based on the action.
        For example, if action is a vector of target weights, rebalance portfolio accordingly.
        """
        # This logic depends on how you define actions.
        # If action = array of weights for each stock:
        #   self.portfolio = {symbol: target_weight * total_capital / price}
        pass

    def _compute_reward(self):
        """
        Compute reward as a function of portfolio performance.
        For example, daily PnL or daily return of the portfolio.
        """
        # Example: calculate portfolio value today and yesterday,
        # reward = (value_today - value_yesterday) / value_yesterday
        # This depends on how you track portfolio holdings and asset prices.
        return 0.0

    def _construct_graph(self, step):
        """
        Construct the pyg_graph at the given step.
        This involves:
          - Selecting the data for the current day
          - Building the Nx graph
          - Converting to pyg_graph
        """
        # 1. Extract the day's data
        day_data = self.historical_data[step]

        # Example: day_data might have {'symbols': [...], 'alphas': [...], 'prices': [...], 'cov_matrix': np.array([...])}

        symbols = day_data["symbols"]
        alphas = day_data["alphas"]
        prices_history = day_data[
            "prices_history"
        ]  # array shape [num_stocks, window_size]
        cov_matrix = day_data["cov_matrix"]  # shape [num_stocks, num_stocks]

        # 2. Build Nx graph
        G = nx.Graph()
        num_stocks = len(symbols)
        for i, sym in enumerate(symbols):
            G.add_node(sym, alpha=alphas[i], price=prices_history[i], index=i)

        for i in range(num_stocks):
            for j in range(i + 1, num_stocks):
                weight = cov_matrix[i, j]
                G.add_edge(symbols[i], symbols[j], weight=weight)

        # 3. Convert to pyg_graph
        pyg_graph = from_networkx(G)

        # Create node features: concat alpha and prices
        alphas_tensor = torch.tensor(
            [G.nodes[node]["alpha"] for node in G.nodes], dtype=torch.float
        ).view(-1, 1)
        prices_tensor = torch.tensor(
            [G.nodes[node]["price"] for node in G.nodes], dtype=torch.float
        )

        pyg_graph.x = torch.cat([alphas_tensor, prices_tensor], dim=1)

        return pyg_graph
