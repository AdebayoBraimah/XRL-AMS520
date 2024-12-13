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

from models import GCNModel, DoubleDQNAgent

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

        stocks_dict: Dict[Dict[str, str | float | List[float]]] = {}

        alphas = [asset.alpha for asset in _alpha_info.values()]
        symbols = [asset.symbol for asset in _alpha_info.values()]
        market_caps = [asset.market_cap for asset in _alpha_info.values()]

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
        # BEGIN: Build graph
        #

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
                # alpha_i = stock_i.get("alpha")
                # price_i = stock_i.get("price")

                # Access stock information for stock 'j'
                stock_j = stocks_dict[j]
                name_j = stock_j.get("name")
                # alpha_j = stock_j.get("alpha")
                # price_j = stock_j.get("price")

                # G.add_edge(name_i, name_j, weight=0)

                try:
                    if cov_total.any():
                        G.add_edge(name_i, name_j, weight=cov_total[i, j])
                    else:
                        G.add_edge(name_i, name_j, weight=0)
                except KeyboardInterrupt:
                    return None
                except:
                    G.add_edge(name_i, name_j, weight=0)

        # BEGIN: DEBUG
        for node in G.nodes:
            if len(G.nodes[node]["price"]) != 30:
                self.debug(f"Node {node} has length {len(G.nodes[node]['price'])}")
        # END: DEBUG

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

        #
        # END: Build graph
        #

        #
        # BEGIN: Build GCN model
        #

        # Number of input features per node, for example, 11 if 1 alpha + 10 time steps
        input_dim = pyg_graph.x.shape[1]
        hidden_dim = 64  # 64
        output_dim = 32  # Adjust this based on the task
        num_layers = 2

        gcn_model = GCNModel(input_dim, hidden_dim, output_dim, num_layers)
        gcn_model.train()  # or .eval() depending on usage

        # embeddings now contain enriched node representations that incorporate both node features and structural info
        with torch.no_grad():
            embeddings = gcn_model(pyg_graph)

        #
        # END: Build GCN model
        #

        # #
        # # BEGIN: Rank stocks by alpha and graph connection strength
        # #

        # # Compute node strength as sum of absolute edge weights
        # scores = []
        # for node in G.nodes:
        #     alpha_val = G.nodes[node]["alpha"]
        #     node_strength = sum(abs(G[node][nbr]["weight"]) for nbr in G[node])
        #     # Example metric: multiply alpha by node_strength
        #     score = alpha_val * node_strength
        #     scores.append((node, score))

        # # Sort by score
        # scores_sorted = sorted(scores, key=lambda x: x[1])

        # # Select top 10% and bottom 10%
        # long_count = max(1, int(0.1 * num_stocks))
        # short_count = long_count

        # short_stocks = [x[0] for x in scores_sorted[:short_count]]
        # long_stocks = [x[0] for x in scores_sorted[-long_count:]]

        # # Place trades: Long top 10%, short bottom 10%
        # # First, liquidate everything not in those sets
        # current_invested = [x.Key.Value for x in self.Portfolio if x.Value.Invested]
        # for symbol_name in current_invested:
        #     if symbol_name not in long_stocks and symbol_name not in short_stocks:
        #         self.Liquidate(symbol_name)

        # # Go long on top 10%
        # for sym in long_stocks:
        #     self.SetHoldings(
        #         sym, 1.0 / (2 * long_count)
        #     )  # allocate evenly among top 10%

        # # Go short on bottom 10%
        # for sym in short_stocks:
        #     self.SetHoldings(
        #         sym, -1.0 / (2 * short_count)
        #     )  # allocate evenly among bottom 10%

        # #
        # # END: Rank stocks by alpha and graph connection strength
        # #

        #
        # BEGIN: Rank stocks by alpha and graph connection strength
        #
        
        
        #
        # END: Rank stocks by alpha and graph connection strength
        #

        self.debug(f"END.")

        # NOTE: Rebalance every 30 days [DO NOT REMOVE]
        # Set the next rebalance time (roughly 30 days later)
        self.next_rebalance_time = self.Time + timedelta(days=30)
