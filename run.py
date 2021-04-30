import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import gym
from argparse import ArgumentParser
from agent import SimplePGAgent
from utils.env_utils import *
from utils.config_utils import read_config


def config_check(config):
    if torch.cuda.is_available():
        config.device = torch.device('cuda')
        config.cuda = True
    else:
        config.device = torch.device('cpu')
        config.cuda = False
    return config


def run(args):
    env = gym.make("CartPole-v0")
    args.env = env

    agent = SimplePGAgent(args)
    while args.t_env <= args.t_max:
        logs = agent.update()
        print(logs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="default", help='which config to run')
    
    args = parser.parse_args()

    config = read_config(args.config, args)
    config = config_check(config)

    print(config)

    run(config)
