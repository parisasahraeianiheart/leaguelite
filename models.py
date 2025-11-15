import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Simple MLP Actor-Critic for PPO."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h)
        return logits, value


class BehaviorCloningPolicy(nn.Module):
    """Policy network trained via supervised learning on (state, action) pairs."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, x):
        return self.net(x)
