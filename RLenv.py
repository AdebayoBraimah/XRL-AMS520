import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx

from typing import Any, Dict, List, Callable

from AlgorithmImports import *

import gym
from gym import spaces
import numpy as np
import torch
from torch_geometric.data import Data


class TradingGraphEnvironmentPPO(gym.Env):
    """
    Custom trading environment with graph-based state representation.
    """

    def __init__(
        self,
        stocks_dict,
        cov_total: np.ndarray,
        calc_trans_cost: Callable,
        calc_port_val: Callable,
        calc_vol_penalty: Callable,
    ):
        super(TradingGraphEnvironmentPPO, self).__init__()

        # Define action space
        # Example: continuous actions representing portfolio allocations per stock
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(stocks_dict),), dtype=np.float32
        )

        self.stocks_dict = stocks_dict
        self.cov_total = cov_total

        self.num_stocks = len(stocks_dict)

        # Number of features per stock (alpha + 30 prices)
        self.num_features = 31

        # Define observation space
        # Assuming graph data is represented as PyG Data objects
        # self.observation_space = spaces.Dict(
        #     {
        #         "graph": spaces.Box(
        #             low=-np.inf,
        #             high=np.inf,
        #             shape=(self.num_stocks, self.num_features),
        #             dtype=np.float32,
        #         )  # Placeholder
        #     }
        # )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_stocks * self.num_features,),
            dtype=np.float32,
        )

        # Variables to help compute reward
        self.portfolio_value: Callable = calc_port_val
        self.transaction_cost: Callable = calc_trans_cost
        self.est_vol_penalty: Callable = calc_vol_penalty

        self.port_val_prev_day = self.portfolio_value()
        self.current_step = 0
        self.max_steps = len(stocks_dict) - 1

    def reset(self):
        # Dict-based approach
        # self.current_step = 0
        # self.portfolio = {}
        # self.port_val_prev_day = self.portfolio_value()

        # graph_np = self._construct_graph()

        # # Validate observation shape
        # assert (
        #     graph_np.shape == self.observation_space.shape
        # ), f"Observation shape {graph_np.shape} does not match expected shape {self.observation_space.shape}"

        # return {"graph": graph_np}

        # Flattened approach
        self.current_step = 0
        self.portfolio = {}
        self.port_val_prev_day = self.portfolio_value()

        # Construct the initial graph and flatten
        graph_np = self._construct_graph().flatten()

        # Validate observation shape
        assert (
            graph_np.shape == self.observation_space.shape
        ), f"Observation shape {graph_np.shape} does not match expected shape {self.observation_space.shape}"

        return graph_np

    def _apply_action(self, action):
        # Ensure action is a NumPy array
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        # Validate action shape
        assert (
            action.shape == self.action_space.shape
        ), f"Action shape {action.shape} does not match action space shape {self.action_space.shape}"

        # Normalize action to sum to <=1 (allowing for leverage)
        action_sum = np.sum(np.abs(action))
        if action_sum > 1.0:
            normalized_action = action / action_sum
        else:
            normalized_action = action

        # Example: 50% long, 50% short
        long_weights = normalized_action.clip(min=0)
        short_weights = -normalized_action.clip(max=0)

        # Normalize long and short weights separately if needed
        if long_weights.sum() > 0:
            long_weights /= long_weights.sum()
        if short_weights.sum() > 0:
            short_weights /= short_weights.sum()

        # Scale weights by desired allocation
        allocation_long = 0.5  # 50% of portfolio
        allocation_short = 0.5  # 50% of portfolio

        final_long_weights = long_weights * allocation_long
        final_short_weights = short_weights * allocation_short

        # Combine long and short weights
        portfolio_weights = final_long_weights - final_short_weights

        # Get current portfolio value
        total_portfolio_value = self.portfolio_value()

        # Initialize new portfolio
        new_portfolio = {}

        for idx, stocks_info in self.stocks_dict.items():
            symbol = stocks_info.get("name")
            prices = stocks_info.get("price", [])

            if not prices:
                continue  # Skip if no price data

            current_price = prices[-1]
            if np.isnan(current_price) or current_price <= 0:
                continue  # Skip invalid prices

            weight = portfolio_weights[idx]
            target_dollar = total_portfolio_value * weight
            shares = target_dollar / (current_price + 1e-8)  # Avoid division by zero

            new_portfolio[symbol] = shares

        # Update the portfolio
        self.portfolio = new_portfolio

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # Apply the action (portfolio adjustments)
        self._apply_action(action)

        # Calculate reward (e.g., portfolio return)
        reward = self._calculate_reward()

        # Check if done
        done = self.current_step >= self.max_steps

        # Get next state
        graph = self._construct_graph().flatten()

        info = {}

        self.current_step += 1

        # return {"graph": graph}, reward, done, info
        return graph, reward, done, info

    def _construct_graph(self):
        """
        Construct the pyg_graph at the given step.
        This involves:
          - Selecting the data for the current day
          - Building the Nx graph
          - Converting to pyg_graph
        """
        # Init data/variables
        stocks_dict = self.stocks_dict
        cov_total = self.cov_total

        # Create a NetworkX graph
        G = nx.Graph()

        # 1. Add names and price data as features
        for idx, stocks_info in stocks_dict.items():
            symbol = stocks_info.get("name")
            alpha = stocks_info.get("alpha")
            _price: List[float] = stocks_info.get("price")

            if not _price:
                _price: List = []
                price: np.array = np.empty(30)
            else:
                price: np.array = np.array(_price)

            # Pad price data array to have 30 elements
            if len(price) < 30:
                price: np.array = np.pad(
                    price, (0, 30 - len(price)), "constant", constant_values=np.nan
                )

            G.add_node(symbol, alpha=alpha, price=price, index=idx)

        # 2. Add edge weights
        num_stocks = len(stocks_dict)
        for i in range(num_stocks):
            for j in range(i + 1, num_stocks):
                # Access stock information for stock 'i'
                stock_i = stocks_dict[i]
                name_i = stock_i.get("name")

                # Access stock information for stock 'j'
                stock_j = stocks_dict[j]
                name_j = stock_j.get("name")

                try:
                    if cov_total.any():
                        G.add_edge(name_i, name_j, weight=cov_total[i, j])
                    else:
                        G.add_edge(name_i, name_j, weight=0)
                except KeyboardInterrupt:
                    return None
                except:
                    G.add_edge(name_i, name_j, weight=0)

        # # BEGIN: DEBUG
        # for node in G.nodes:
        #     if len(G.nodes[node]["price"]) != 30:
        #         # self.Debug(f"Node {node} has length {len(G.nodes[node]['price'])}")
        # # END: DEBUG

        # Convert the NetworkX graph to PyTorch Geometric format
        pyg_graph = from_networkx(G)

        # Convert node features (alpha and prices) into tensors
        alphas_tensor = torch.tensor(
            [G.nodes[node]["alpha"] for node in G.nodes], dtype=torch.float
        ).view(-1, 1)
        prices_tensor = torch.tensor(
            [G.nodes[node]["price"] for node in G.nodes], dtype=torch.float
        )  # .view(-1, 1)

        # Assign node features as a concatenation of alphas and prices
        pyg_graph.x = torch.cat([alphas_tensor, prices_tensor], dim=1)

        graph_np = pyg_graph.x.numpy()  # Shape: (num_stocks, num_features)

        return graph_np

    def _calculate_reward(self):
        """Simple daily return-based reward, possibly adjusted by a penalty for large drawdowns or volatility.

        NOTE: Work in progress.
        """
        # Compute portfolio value today (e.g., sum of holdings * current prices)
        portfolio_value_yesterday = self.port_val_prev_day
        portfolio_value_today = self.portfolio_value()

        # Update portfolio value for previous day
        self.port_val_prev_day = portfolio_value_today

        raw_return = (portfolio_value_today - portfolio_value_yesterday) / (
            portfolio_value_yesterday + 1e-6
        )

        # Penalize large position changes or excessive leverage
        transaction_costs = self.transaction_cost()
        # Optional: risk penalty (e.g., if volatility is high, penalize)
        volatility_penalty = self.est_vol_penalty()

        reward = raw_return - transaction_costs - volatility_penalty
        return reward

    def _portfolio_value(self):
        # Implement logic to calculate the current portfolio value
        return self.portfolio_value  # Placeholder

    def render(self, mode="human"):
        # Optional: implement visualization
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value}")


class TradingGraphEnvironmentDDQN:
    def __init__(
        self,
        # stocks_dict: Dict[Dict[str, str | float | List[float]]],
        stocks_dict,
        cov_total: np.ndarray,
        calc_trans_cost: Callable,
        calc_port_val: Callable,
        calc_vol_penalty: Callable,
        start_index: int = 0,
        end_index: int | None = None,
    ):
        self.start_index = start_index
        self.current_step = start_index
        self.end_index = end_index if end_index is not None else len(stocks_dict) - 1

        self.stocks_dict = stocks_dict
        self.cov_total = cov_total

        # Variables to help compute reward
        self.portfolio_value: Callable = calc_port_val
        self.transaction_cost: Callable = calc_trans_cost
        self.est_vol_penalty: Callable = calc_vol_penalty

        self.port_val_prev_day = self.portfolio_value()

    def reset(self):
        """Reset the environment to the start of a new episode."""
        # self.current_step = self.start_index
        self.portfolio = {}  # reset portfolio (if needed)
        # initial_graph = self._construct_graph(self.current_step)
        initial_graph = self._construct_graph()
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
        next_graph = self._construct_graph()

        info = {}  # Additional info if needed
        return next_graph, reward, done, info

    def _apply_action(self, action):
        """
        Update the portfolio based on a single action integer.
        Action Mapping:
            0 - Hold: Do nothing.
            1 - Buy Top 10%: Long the top 10% of stocks.
            2 - Sell Bottom 10%: Short the bottom 10% of stocks.
            3 - Rebalance: Long top 10% and Short bottom 10%.
        """
        if not isinstance(action, int):
            raise ValueError("Action must be an integer.")

        num_nodes = len(self.stocks_dict)
        cutoff = max(1, num_nodes // 10)  # 10% cutoff

        # Extract alpha and compute node strength for each stock
        alpha_values = self.current_graph["x"][:, 0].cpu().numpy()
        edge_weights = (
            self.current_graph["edge_attr"][:, 0].cpu().numpy()
            if "edge_attr" in self.current_graph
            else np.zeros(self.current_graph["edge_index"].shape[1])
        )

        # Compute node strength (sum of absolute edge weights)
        node_strength = np.zeros(num_nodes)
        src_nodes = self.current_graph["edge_index"][0].cpu().numpy()
        dst_nodes = self.current_graph["edge_index"][1].cpu().numpy()

        for e, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
            w = edge_weights[e]
            node_strength[u] += abs(w)
            node_strength[v] += abs(w)

        # Compute the score = alpha * node_strength
        scores = alpha_values * node_strength

        # Sort stocks by their score
        sorted_indices = np.argsort(scores)  # ascending order

        bottom_indices = sorted_indices[:cutoff]
        top_indices = sorted_indices[-cutoff:]

        node_symbols = [self.stocks_dict[i]["name"] for i in range(num_nodes)]
        top_symbols = [node_symbols[i] for i in top_indices]
        bottom_symbols = [node_symbols[i] for i in bottom_indices]

        if action == 0:
            # Hold: Do nothing
            # self.Debug("Action: Hold - No changes to portfolio.")
            return

        elif action == 1:
            # Buy Top 10%
            # self.Debug("Action: Buy Top 10% Stocks.")
            # Liquidate positions not in top set
            current_invested = self.get_invested_positions()
            for sym in current_invested:
                if sym not in top_symbols:
                    self.Liquidate(sym)

            # Allocate equal weights to top 10%
            if len(top_symbols) > 0:
                long_weight = 0.5 / len(top_symbols)  # 50% of portfolio
                for sym in top_symbols:
                    self.SetHoldings(sym, long_weight)

        elif action == 2:
            # Sell Bottom 10%
            # self.Debug("Action: Sell Bottom 10% Stocks.")
            # Liquidate positions not in bottom set
            current_invested = self.get_invested_positions()
            for sym in current_invested:
                if sym not in bottom_symbols:
                    self.Liquidate(sym)

            # Allocate equal weights to short bottom 10%
            if len(bottom_symbols) > 0:
                short_weight = -0.5 / len(bottom_symbols)  # 50% of portfolio
                for sym in bottom_symbols:
                    self.SetHoldings(sym, short_weight)

        elif action == 3:
            # Rebalance: Long top 10% and Short bottom 10%
            # self.Debug("Action: Rebalance - Long Top 10% and Short Bottom 10% Stocks.")
            # Liquidate positions not in top or bottom sets
            current_invested = self.get_invested_positions()
            for sym in current_invested:
                if sym not in top_symbols and sym not in bottom_symbols:
                    self.Liquidate(sym)

            # Allocate equal weights to long and short positions
            if len(top_symbols) > 0:
                long_weight = 0.25 / len(top_symbols)  # 25% long
                for sym in top_symbols:
                    self.SetHoldings(sym, long_weight)

            if len(bottom_symbols) > 0:
                short_weight = -0.25 / len(bottom_symbols)  # 25% short
                for sym in bottom_symbols:
                    self.SetHoldings(sym, short_weight)

        # else:
        #     # self.Debug(f"Action: Undefined action {action}. No changes made.")

        # """
        # Update the portfolio based on the action vector.
        # Assume action is a 1D numpy array of length N_stocks, where each entry corresponds to
        # the target weight of that stock in the portfolio.

        # Steps:
        # - Normalize action to sum to 1 (if not already).
        # - Compute target dollar allocation per stock.
        # - Convert to number of shares based on last price.
        # - Update self.portfolio.
        # """
        # if not isinstance(action, np.ndarray):
        #     action = np.array(action, dtype=float)

        # # Normalize weights if necessary
        # weight_sum = action.sum()
        # if not np.isclose(weight_sum, 1.0):
        #     action = action / (weight_sum + 1e-6)

        # total_value = self.portfolio_value()

        # new_portfolio = {}
        # for i, stocks_info in self.stocks_dict.items():
        #     symbol = stocks_info.get("name")
        #     prices = stocks_info.get("price", [])
        #     if not prices or len(prices) == 0:
        #         # No price data for this stock, skip it
        #         continue

        #     current_price = prices[-1]  # last known closing price
        #     if np.isnan(current_price) or current_price <= 0:
        #         # If invalid price, skip this stock
        #         continue

        #     target_weight = action[i]
        #     target_dollar = total_value * target_weight
        #     shares = target_dollar / (current_price + 1e-6)  # avoid division by zero
        #     new_portfolio[symbol] = shares

        # # Update portfolio holdings
        # self.portfolio = new_portfolio

    def _compute_reward(self):
        """Simple daily return-based reward, possibly adjusted by a penalty for large drawdowns or volatility.

        NOTE: Work in progress.
        """
        # Compute portfolio value today (e.g., sum of holdings * current prices)
        portfolio_value_yesterday = self.port_val_prev_day
        portfolio_value_today = self.portfolio_value()

        # Update portfolio value for previous day
        self.port_val_prev_day = portfolio_value_today

        raw_return = (portfolio_value_today - portfolio_value_yesterday) / (
            portfolio_value_yesterday + 1e-6
        )

        # Penalize large position changes or excessive leverage
        transaction_costs = self.transaction_cost()
        # Optional: risk penalty (e.g., if volatility is high, penalize)
        volatility_penalty = self.est_vol_penalty()

        reward = raw_return - transaction_costs - volatility_penalty
        return reward

    def _construct_graph(self):
        """
        Construct the pyg_graph at the given step.
        This involves:
          - Selecting the data for the current day
          - Building the Nx graph
          - Converting to pyg_graph
        """
        # Init data/variables
        stocks_dict = self.stocks_dict
        cov_total = self.cov_total

        # Create a NetworkX graph
        G = nx.Graph()

        # 1. Add names and price data as features
        for idx, stocks_info in stocks_dict.items():
            symbol = stocks_info.get("name")
            alpha = stocks_info.get("alpha")
            _price: List[float] = stocks_info.get("price")

            if not _price:
                _price: List = []
                price: np.array = np.empty(30)
            else:
                price: np.array = np.array(_price)

            # Pad price data array to have 30 elements
            if len(price) < 30:
                price: np.array = np.pad(
                    price, (0, 30 - len(price)), "constant", constant_values=np.nan
                )

            G.add_node(symbol, alpha=alpha, price=price, index=idx)

        # 2. Add edge weights
        num_stocks = len(stocks_dict)
        for i in range(num_stocks):
            for j in range(i + 1, num_stocks):
                # Access stock information for stock 'i'
                stock_i = stocks_dict[i]
                name_i = stock_i.get("name")

                # Access stock information for stock 'j'
                stock_j = stocks_dict[j]
                name_j = stock_j.get("name")

                try:
                    if cov_total.any():
                        G.add_edge(name_i, name_j, weight=cov_total[i, j])
                    else:
                        G.add_edge(name_i, name_j, weight=0)
                except KeyboardInterrupt:
                    return None
                except:
                    G.add_edge(name_i, name_j, weight=0)

        # # BEGIN: DEBUG
        # for node in G.nodes:
        #     if len(G.nodes[node]["price"]) != 30:
        #         # self.Debug(f"Node {node} has length {len(G.nodes[node]['price'])}")
        # # END: DEBUG

        # Convert the NetworkX graph to PyTorch Geometric format
        pyg_graph = from_networkx(G)

        # Convert node features (alpha and prices) into tensors
        alphas_tensor = torch.tensor(
            [G.nodes[node]["alpha"] for node in G.nodes], dtype=torch.float
        ).view(-1, 1)
        prices_tensor = torch.tensor(
            [G.nodes[node]["price"] for node in G.nodes], dtype=torch.float
        )  # .view(-1, 1)

        # Assign node features as a concatenation of alphas and prices
        pyg_graph.x = torch.cat([alphas_tensor, prices_tensor], dim=1)

        return pyg_graph
