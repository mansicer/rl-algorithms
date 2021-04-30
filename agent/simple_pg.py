import os
import torch
from policy import MLPPolicy
from component import PGFeeder
from utils.env_utils import sample_one_trajectory

class SimplePGAgent():
    def __init__(self, args) -> None:
        args.t_env = 0
        args.t_update = 0
        self.args = args
        self.env = args.env
        self.policy = MLPPolicy(args)
        self.feeder = PGFeeder(args)
        self.optimizer = torch.optim.Adam(self.policy.parameters())

    def update(self) -> dict:
        self.optimizer.zero_grad()
        obs, act, weight = self.feeder.sample(self.policy)
        prob = self.policy.get_policy_dist(obs).log_prob(act)
        loss = - (prob * weight).mean()
        loss.backward()
        self.optimizer.step()

        self.args.t_update += 1

        logs = {
            "loss": loss.item()
        }
        logs.update(self.feeder.last_log())

        return logs
        
    def save_model(self):
        torch.save(self.policy, open(os.path.join(self.args.model_save_path, self.t_env), 'wb'))
    
