import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from utils.mlp_utils import build_mlp_with_relu
from typing import Union

class Policy(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def select_action(self, obs: torch.tensor) -> int:
        raise NotImplementedError

    def get_policy_dist(self, obs: torch.tensor) -> D.Distribution:
        raise NotImplementedError

    def forward(self, obs: torch.tensor) -> torch.tensor:
        raise NotImplementedError
