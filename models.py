# region imports
from AlgorithmImports import *
# endregion
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GatedGraphConv


class TGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(TGCN, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # Step 1: Apply GCN layer to capture spatial dependencies
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn(x, edge_index))  # Graph convolution for spatial features

        # Step 2: Apply GRU to capture temporal dependencies (prices over time)
        # Reshape for GRU: (batch_size, sequence_length, feature_dim)
        x = x.view(x.size(0), -1, 1)  # Assuming temporal data is in the second feature column
        x, _ = self.gru(x)  # Apply GRU (recurrent unit) for temporal modeling

        # Step 3: Apply a fully connected layer to produce final output
        x = self.fc(x[:, -1, :])  # Take the output from the last time step
        return x


# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GRU

# class TGCN(torch.nn.Module):
#     def __init__(self, num_node_features, hidden_dim, num_time_steps):
#         super(TGCN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, hidden_dim)
#         self.gru = GRU(hidden_dim, hidden_dim)
#         self.fc = torch.nn.Linear(hidden_dim, 1)  # Predict stock returns or other outputs
#         self.num_time_steps = num_time_steps

#     def forward(self, data_list):
#         # data_list contains multiple graphs over time (list of graph data)
#         h_list = []
#         for data in data_list:
#             x, edge_index = data.x, data.edge_index
#             h = F.relu(self.conv1(x, edge_index))
#             h_list.append(h)

#         h_cat = torch.stack(h_list, dim=1)  # Stack all graph outputs along time axis
#         h, _ = self.gru(h_cat)  # Apply GRU to capture temporal dependencies
#         return self.fc(h[:, -1])  # Use the last time step's output
