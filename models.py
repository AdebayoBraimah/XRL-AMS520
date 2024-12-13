# region imports
# from AlgorithmImports import *

# endregion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import gym
import torch.optim as optim
import random
import numpy as np
from collections import deque

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FlatFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for flat vector observations.
    """
    def __init__(self, observation_space, features_dim=64):
        # Ensure observation_space is a Box
        assert isinstance(observation_space, gym.spaces.Box), "FlatFeatureExtractor only works with Box spaces."
        super(FlatFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Define the network layers
        self.linear1 = nn.Linear(observation_space.shape[0], 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, features_dim)

    def forward(self, observations):
        """
        Forward pass through the feature extractor.
        """
        x = self.linear1(observations)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class GCNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor integrating a Graph Convolutional Network (GCN).
    """

    def __init__(self, observation_space, features_dim=32):
        # Assuming observation_space is a gym.spaces.Dict with 'graph' key
        super(GCNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Define your GCN architecture
        self.gcn1 = GCNConv(
            in_channels=observation_space["graph"].shape[1], out_channels=64
        )
        self.gcn2 = GCNConv(in_channels=64, out_channels=features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        """
        Forward pass for the feature extractor.
        """
        graph = observations["graph"]  # Assuming 'graph' is a batch of PyG Data objects

        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = self.gcn1(x, edge_index)
        x = self.relu(x)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)

        # Global pooling to obtain a graph-level representation
        x = global_mean_pool(x, batch)

        return x


class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_values = self.critic(state)
        return action_probs, state_values


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GCNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.relu(x)
        x = self.convs[-1](x, edge_index)
        return x


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DoubleDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=10000,
        update_target_every=1000,
        device="cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.update_target_every = update_target_every
        self.device = device

        # Initialize the two Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.step_count = 0

    def select_action(self, state, epsilon=0.1):
        # Epsilon-greedy policy
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # Current Q-values
        current_q = self.q_network(states_t).gather(1, actions_t)

        # Double DQN target:
        # 1) Choose next action using q_network (online)
        next_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
        # 2) Evaluate chosen action using target_network
        next_q = self.target_network(next_states_t).gather(1, next_actions)
        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        # Compute loss
        loss = nn.MSELoss()(current_q, target_q.detach())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.q_network.state_dict())


class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Additional layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Final layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
            self.bns.append(nn.BatchNorm1d(output_dim))
        else:
            # If only 1 layer, output_dim = hidden_dim
            # and no further BatchNorm needed
            self.convs[0] = GCNConv(input_dim, output_dim)

        self.activation = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Pass data through each GCN layer
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Apply batch norm if this layer has it
            if i < len(self.bns):
                x = self.bns[i](x)
            # Apply activation except possibly after the last layer
            if i < len(self.convs) - 1:
                x = self.activation(x)

        return x


class TGCN(torch.nn.Module):
    """A Temporal Graph Convolutional Network (TGCN) model for time series forecasting."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """Initialize the TGCN model.

        NOTE: This is a work in progress and may require additional modifications.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            output_dim: Output feature dimension.
            num_layers: Number of layers in the model. Defaults to 2.
        """
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
        x = x.view(
            x.size(0), -1, 1
        )  # Assuming temporal data is in the second feature column
        x, _ = self.gru(x)  # Apply GRU (recurrent unit) for temporal modeling

        # Step 3: Apply a fully connected layer to produce final output
        x = self.fc(x[:, -1, :])  # Take the output from the last time step
        return x
