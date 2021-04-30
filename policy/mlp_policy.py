from .policy import Policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from utils.mlp_utils import build_mlp_with_relu
from typing import Union

class MLPPolicy(Policy):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.obs_dim = args.env_config.obs_dim
        self.act_dim = args.env_config.act_dim
        self.hidden_dims = args.hidden_dims
        self.policy = build_mlp_with_relu(self.obs_dim, self.act_dim, self.hidden_dims)

    def select_action(self, obs: torch.tensor) -> int:
        obs = obs.unsqueeze(0)
        action = self.get_policy_dist(obs).sample()
        return action.cpu().numpy().item()

    def get_policy_dist(self, obs: torch.tensor) -> D.Distribution:
        logits = self.forward(obs)
        return D.Categorical(logits=logits)

    def forward(self, obs: torch.tensor) -> torch.tensor:
        return self.policy(obs)
