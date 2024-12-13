import numpy as np
import pandas as pd

import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from datetime import timedelta
from tqdm import trange

# region imports
from AlgorithmImports import *

from universe import NorthFieldUniverse, NorthFieldInvestableUniverse
from alpha import NorthFieldAlpha
from factor_definition import NorthFieldFactorDefinition
from factor_correlation import NorthFieldFactorCorrelation
from factor_exposure import NorthFieldFactorExposure

from typing import Dict, List

# from models import GCNModel, DoubleDQNAgent
from models import PPOAgent, GCNModel
from RLenv import TradingGraphEnvironmentPPO
from custom_policy import CustomPPOPolicy, CustomPPOPolicyFlat

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env


# endregion

# Documentation:
# - README: https://www.dropbox.com/scl/fi/5lubbi4p7b9art7ig73v6/read-me.docx?rlkey=bwvgomxo4b1kh3dtkzf8mvdtm&st=bz0ja53n&dl=0
# - Flat File description: https://www.dropbox.com/scl/fi/s5mgdu69lw7ef5k6wfs00/Flat-File-description-FF_-prefix.pdf?rlkey=uywqpaw8wvpw26no3isomqn5y&st=h0waii0g&dl=0

# TODO:
#   - Create debug option in algorithm
#   - Add argument for number of days to fetch historical data


class NorthFieldDemoAlgorithm(QCAlgorithm):

    # This example simply reads information from the custom data files in
    # the Object Store and loads it into the algorithm.

    def initialize(self):
        # self.set_start_date(2017, 12, 20)
        self.set_start_date(2023, 8, 13)  # TEST: start date
        self.set_end_date(2024, 8, 15)
        self.set_cash(1_000_000)  # TEST: Set cash to $1,000,000 for testing purposes
        self.universe_settings.resolution = Resolution.DAILY

        # Add custom universe.
        # self._universe = self.add_universe(NorthFieldUniverse, 'NorthFieldUniverse', Resolution.DAILY, self._select_assets)
        # self.add_universe(NorthFieldInvestableUniverse, 'NorthFieldInvestableUniverse', Resolution.DAILY, self._select_assets)

        # Load investible universe
        self._universe = self.add_universe(
            NorthFieldInvestableUniverse,
            "NorthFieldInvestableUniverse",
            Resolution.DAILY,
            self._select_assets,
        )

        # Add custom datasets.
        NorthFieldFactorExposure.algorithm = self
        for dataset in [
            NorthFieldAlpha,
            NorthFieldFactorDefinition,
            NorthFieldFactorCorrelation,
            NorthFieldFactorExposure,
        ]:
            self.add_data(dataset, dataset.__name__, Resolution.DAILY)

        # Set the initial rebalance time to now + 30 days
        self.next_rebalance_time = self.Time  # + timedelta(days=30)

    def _select_assets(self, data):
        # Select a subset of all the assets in the universe file.
        symbols = [x.symbol for x in data]
        return symbols

    def _portfolio_value(self):
        # Get current portfolio value
        pv = self.portfolio.total_portfolio_value
        return pv

    def _calculate_transaction_costs(self):
        # Return total transaction costs incurred so far this day/step
        # If you want incremental costs per step, track changes in orders.
        # Here we just return the cumulative self.transaction_costs.
        return self.transaction_costs

    def _estimate_volatility_penalty(self, window=30):
        # Compute volatility based on last `window` portfolio values
        if len(self.portfolio_values) < window + 1:
            return 0.0  # Not enough data yet

        # Compute returns: (V_t - V_(t-1)) / V_(t-1)
        recent_values = self.portfolio_values[-(window + 1) :]
        daily_returns = []
        for i in range(1, len(recent_values)):
            ret = (recent_values[i] - recent_values[i - 1]) / (
                recent_values[i - 1] + 1e-6
            )
            daily_returns.append(ret)

        # Compute standard deviation of daily returns
        vol = np.std(daily_returns)
        # Example penalty: proportional to volatility
        penalty = vol
        return penalty

    def on_data(self, data):
        # # Rebalance monthly
        # if self.Time < self.next_rebalance_time:
        #     return

        # Once the code below runs, set the next rebalance time at the end.

        # Get alpha data.
        symbol_data_by_symbol = data.get(NorthFieldAlpha).items()

        # Select the subset of the alpha data objects that are in the universe.
        if self._universe.selected:
            symbol_data_by_symbol = {
                symbol: symbol_data
                for symbol, symbol_data in symbol_data_by_symbol
                if symbol in self._universe.selected
            }

        #
        # BEGIN: Get stock information: Ticker, alpha, historical price data
        #

        # Iterate over stock and alpha
        _alpha_info = data.get(NorthFieldAlpha)
        if not _alpha_info or len(_alpha_info) == 0:
            return

        alphas = [asset.alpha for asset in _alpha_info.values()]
        symbols = [asset.symbol for asset in _alpha_info.values()]
        market_caps = [asset.market_cap for asset in _alpha_info.values()]

        stocks_dict: Dict[Dict[int, str, str | float | List[float]]] = {}
        idx = 0

        for symbol, alpha in zip(symbols, alphas):
            # Get stock history
            _stock = self.AddEquity(symbol, Resolution.DAILY)
            history = self.History(_stock.Symbol, 30, Resolution.DAILY)

            try:
                closing_prices = history["close"].tolist()
                # closing_prices = history["close"].values # TODO: Try this
            except KeyboardInterrupt:
                return None
            except:
                # closing_prices = None
                closing_prices = []
                self.debug(f"Stock has no data: {symbol}")

            _tmp_dict = {idx: {"name": symbol, "alpha": alpha, "price": closing_prices}}

            # Update original dict
            stocks_dict.update(_tmp_dict)

            # self.debug(f"Size: {len(stocks_dict)}")

            # Increase index
            idx = idx + 1

        #
        # END: Get stock information: Ticker, alpha, historical price data
        #

        #
        # BEGIN: Compute (total) covariance matrix of stock returns
        #

        # 1. Construct the Factor Covariance Matrix

        # 1.1. correlation matrix (C_F)
        factor_corr = list(data.get(NorthFieldFactorCorrelation).values())
        if factor_corr:
            series = [f.series for f in factor_corr]
            C_F = pd.DataFrame(series, index=series[0].index)

            C_F.to_numpy()
            # self.debug(f"C_F: {C_F.shape}")
        else:
            # C_F = np.empty([2,2])
            C_F = pd.Series(dtype=int)
            self.debug(f"Factor correlation is empty.")

        # 1.2. Create diagonal matrix of factor volatilities (var_F)
        _factor_def = data.get(NorthFieldFactorDefinition)

        variance: list = [fd.variance for fd in _factor_def.values()]

        if variance:
            var_F: np.array = np.array(variance)
            # var_F: np.array = np.diag(variance)
            # self.debug(f"var_F: {var_F.shape}")
        else:
            var_F = np.empty([2, 2])
            self.debug(f"Volatility is empty.")

        # Diagonal matrix of factor volatilities
        if var_F.any():
            D_F = np.diag(var_F)

        # 1.3. Compute factor covariance matrix (cov_F)
        if D_F.any() and C_F.to_numpy().any():
            # self.debug(f"C_F: {C_F.shape}")
            # self.debug(f"D_F: {D_F.shape}")
            cov_F = D_F @ C_F.to_numpy() @ D_F
            # self.debug(f"cov_F: {cov_F.shape}")
        else:
            cov_F = np.empty([2, 2])

        # END: Step 1

        # 2. Compute the Stock Covariance Matrix
        _factor_exposures = data.get(NorthFieldFactorExposure)

        # 2.1. Beta values
        factor_exposures: list = [fe.betas for fe in _factor_exposures.values()]

        if factor_exposures:
            beta: np.array = np.array(factor_exposures)
            # self.debug(f"beta: {beta.shape}")
        else:
            beta = np.empty([2, 2])
            self.debug(f"beta is empty.")

        # 2.2. Compute Stock Covariance Matrix (Factor-Driven Component, cov_stock_factor)
        if beta.any() and cov_F.any():
            # self.debug(f"beta: {beta.shape}")
            # self.debug(f"cov_F: {cov_F.shape}")
            # self.debug(f"beta^T: {beta.T.shape}")
            cov_stock_factor = beta @ cov_F @ beta.T
        else:
            cov_stock_factor = np.empty([2, 2])
            self.debug(f"cov_stock_factor matrix is empty.")

        # END: Step 2

        # 3. Add the Idiosyncratic Volatility (Variance)

        # 3.1. Idiosyncratic Variance (D_e)
        # Access the NorthFieldFactorExposure data

        # Extract the monthly_residual_sd_pct values
        factor_volatilities: list = [
            fe.monthly_residual_sd_pct for fe in _factor_exposures.values()
        ]

        if factor_volatilities:
            var_e: np.array[float] = np.array(factor_volatilities)
            D_e: np.array = np.diag(var_e)
            # self.debug(f"D_e: {D_e.shape}")
        else:
            D_e = np.empty([2, 2])
            self.debug(f"Variance is empty.")

        # 4. Final Covariance Matrix
        if cov_stock_factor.any() and D_e.any():
            cov_total = cov_stock_factor + D_e
        else:
            cov_total = np.empty([2, 2])
            self.debug(f"Total covariance (cov_total) matrix is empty.")

        #
        # END: Compute (total) covariance matrix of stock returns
        #

        #
        # BEGIN: Build GCN model
        #

        # env = TradingGraphEnvironmentDDQN(
        #     stocks_dict=stocks_dict,
        #     cov_total=cov_total,
        #     calc_port_val=self._portfolio_value,
        #     calc_trans_cost=self._calculate_transaction_costs,
        #     calc_vol_penalty=self._estimate_volatility_penalty,
        # )
        # state = env.reset()

        env = TradingGraphEnvironmentPPO(
            stocks_dict=stocks_dict,
            cov_total=cov_total,
            calc_port_val=self._portfolio_value,
            calc_trans_cost=self._calculate_transaction_costs,
            calc_vol_penalty=self._estimate_volatility_penalty,
        )
        state = env.reset()

        # Instantiate the environment
        env = DummyVecEnv(
            [
                lambda: TradingGraphEnvironmentPPO(
                    stocks_dict=stocks_dict,
                    cov_total=cov_total,
                    calc_port_val=self._portfolio_value,
                    calc_trans_cost=self._calculate_transaction_costs,
                    calc_vol_penalty=self._estimate_volatility_penalty,
                )
            ]
        )

        # Validate the environment
        check_env(env, warn=True)

        # Instantiate the PPO agent with the custom policy
        model = PPO(
            # CustomPPOPolicy,
            CustomPPOPolicyFlat,
            env,
            verbose=1,
            tensorboard_log="./ppo_trading_tensorboard/",
            # Add other hyperparameters as needed
        )

        # Train the agent
        model.learn(total_timesteps=10)

        # Save the model
        model.save("ppo_trading_gcn")

        # pyg_graph = env._construct_graph()  # Build graph to get dimensions and features

        # # Number of input features per node, for example, 11 if 1 alpha + 10 time steps
        # input_dim = pyg_graph.x.shape[1]
        # hidden_dim = 64  # 64
        # output_dim = 32  # Adjust this based on the task
        # num_layers = 2

        # gcn_model = GCNModel(input_dim, hidden_dim, output_dim, num_layers)
        # gcn_model.train()  # or .eval() depending on usage

        #
        # END: Build GCN model
        #

        #
        # BEGIN: GraphRL training with DDQN
        #

        # Episode training loop (for DDQN)
        num_episodes = 100
        epsilon = 1.0
        epsilon_end = 0.1
        epsilon_decay = 0.995
        rebalance_interval = 30  # days

        # Initialize the Double DQN agent
        state_dim = output_dim  # from the GCN output_dim
        action_dim = 3  # example: buy, hold, sell

        agent = DoubleDQNAgent(
            # state_dim=embeddings.shape[1],
            state_dim=state_dim,
            action_dim=action_dim,
            device="cpu",
        )

        with torch.no_grad():
            embeddings = gcn_model(pyg_graph)

        # Training loop
        for episode in trange(num_episodes, desc="Training"):
            pyg_graph = env.reset()

            with torch.no_grad():
                embeddings = gcn_model(pyg_graph)

            state = embeddings.mean(dim=0).cpu().numpy()

            done = False
            episode_reward = 0.0
            step_count = 0

            while not done:
                # Choose action using epsilon-greedy
                action = agent.select_action(state, epsilon=epsilon)

                # Execute action in the environment
                next_pyg_graph, reward, done, info = env.step(action)
                episode_reward += reward
                step_count += 1

                with torch.no_grad():
                    next_embeddings = gcn_model(next_pyg_graph)
                next_state = next_embeddings.mean(dim=0).cpu().numpy()

                agent.store_transition(state, action, reward, next_state, done)
                agent.update()

                # Check if it's time to rebalance the portfolio
                # This is a simple check every `rebalance_interval` steps.
                # In a real scenario, you might use the environment's current date (info['current_date']).
                if step_count % rebalance_interval == 0 and not done:
                    # Perform the sorting and rebalancing logic

                    # Extract alpha and compute node strength for each stock:
                    # Here we assume:
                    #  - pyg_graph.x: node features with alpha at x[:,0] and price data at x[:,1:]
                    #  - pyg_graph.edge_index: edges with covariance as weights in pyg_graph.edge_attr (if stored)
                    # If covariance is not directly stored in edge_attr, you might reconstruct it or have it accessible from environment.

                    alpha_values = pyg_graph.x[:, 0].cpu().numpy()

                    # Compute node strength = sum of absolute edge weights
                    # Assuming edge weights stored in pyg_graph.edge_attr:
                    # edge_index: [2, E], edge_attr: [E, ...]
                    # If you have covariance in edge_attr[:,0], for example:
                    if (
                        hasattr(pyg_graph, "edge_attr")
                        and pyg_graph.edge_attr is not None
                    ):
                        edge_weights = pyg_graph.edge_attr[:, 0].cpu().numpy()
                    else:
                        # If not stored, fallback or skip
                        edge_weights = np.zeros(pyg_graph.edge_index.shape[1])

                    # Build adjacency to sum edge weights per node
                    num_nodes = pyg_graph.x.shape[0]
                    node_strength = np.zeros(num_nodes)

                    # pyg_graph.edge_index has shape [2, E], each column is an edge (src, dst)
                    src_nodes = pyg_graph.edge_index[0].cpu().numpy()
                    dst_nodes = pyg_graph.edge_index[1].cpu().numpy()

                    for e, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
                        w = edge_weights[e]
                        # Undirected graph: contribute to both u and v
                        node_strength[u] += abs(w)
                        node_strength[v] += abs(w)

                    # Compute the score = alpha * node_strength
                    scores = alpha_values * node_strength

                    # Sort stocks by their score
                    sorted_indices = np.argsort(scores)  # ascending order

                    # top 10% and bottom 10%
                    num_stocks = len(scores)
                    cutoff = max(1, num_stocks // 10)
                    bottom_indices = sorted_indices[:cutoff]
                    top_indices = sorted_indices[-cutoff:]

                    # Construct a portfolio action:
                    # For simplicity, assume we have a function environment.set_holdings(symbol, weight)
                    # or environment handles the actual rebalancing. Alternatively, store these actions
                    # as RL actions or send them through a broker interface in your environment.
                    # Here we just demonstrate the logic.

                    # First, liquidate everything not in top or bottom sets
                    current_invested = [
                        s
                        for s in self.Portfolio.Keys
                        if self.Portfolio[s].Quantity != 0
                    ]

                    num_nodes = len(stocks_dict)
                    node_symbols = [stocks_dict[i]["name"] for i in range(num_nodes)]

                    top_symbols = [node_symbols[i] for i in top_indices]
                    bottom_symbols = [node_symbols[i] for i in bottom_indices]

                    # Liquidate positions that are not in the top or bottom sets
                    for sym in current_invested:
                        if sym not in top_symbols and sym not in bottom_symbols:
                            self.Liquidate(sym)

                    # Go long the top 10%
                    if len(top_symbols) > 0:
                        long_weight = 1.0 / (2 * len(top_symbols))
                        for sym in top_symbols:
                            self.SetHoldings(sym, long_weight)

                    # Short the bottom 10%
                    if len(bottom_symbols) > 0:
                        short_weight = -1.0 / (2 * len(bottom_symbols))
                        for sym in bottom_symbols:
                            self.SetHoldings(sym, short_weight)

                state = next_state

            # End of episode
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            print(
                f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}"
            )

        #
        # END: GraphRL training with DDQN
        #

        self.debug(f"END.")

        # NOTE: Rebalance every 30 days [DO NOT REMOVE]
        # Set the next rebalance time (roughly 30 days later)
        self.next_rebalance_time = self.Time + timedelta(days=30)
