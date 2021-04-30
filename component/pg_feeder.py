from .basic_feeder import DataFeeder
from utils.env_utils import sample_one_trajectory
import numpy as np
import torch

class PGFeeder(DataFeeder):
    def __init__(self, args) -> None:
        self.args = args
        self.batch_size = args.batch_size
        self.env = args.env
        self.returns = []

    def sample(self, policy) -> dict:
        obs_arr = []
        act_arr = []
        ret_arr = []

        total_steps = 0

        while True:
            trajectory = sample_one_trajectory(self.env, policy, self.args)
            
            obs_arr.extend(trajectory['observation'])
            act_arr.extend(trajectory['action'])
            ret = sum(trajectory['reward'])
            ret_arr.extend([ret] * trajectory['episode_length'])

            total_steps += trajectory['episode_length']

            if total_steps > self.batch_size:
                break
        
        self.args.t_env += total_steps
        self.returns.append([list(set(ret_arr))])

        obs_arr = torch.as_tensor(obs_arr[:self.batch_size]).float().to(self.args.device)
        act_arr = torch.as_tensor(act_arr[:self.batch_size]).int().to(self.args.device)
        ret_arr = torch.as_tensor(ret_arr[:self.batch_size]).float().to(self.args.device)

        return obs_arr, act_arr, ret_arr

    def last_log(self) -> dict:
        returns = np.array(self.returns[-1])
        return_mean = returns.mean()
        return_std = returns.std()
        logs = {
            "return_mean": return_mean,
            "return_std": return_std
        }
        return logs
