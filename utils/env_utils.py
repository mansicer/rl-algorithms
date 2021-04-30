import gym
import torch
from policy import Policy
from types import SimpleNamespace


def sample_one_trajectory(env: gym.Env, policy: Policy, args: SimpleNamespace) -> dict:
    obs_arr = []
    act_arr = []
    rew_arr = []
    done_arr = []
    info_arr = []

    while True:
        obs = env.reset()
        done = False
        while not done:
            obs_arr.append(obs)
            obs = torch.tensor(obs).to(args.device).float()
            action = policy.select_action(obs)
            obs, rew, done, info = env.step(action)
            act_arr.append(action)
            rew_arr.append(rew)
            done_arr.append(done)
            info_arr.append(info)
        
        if len(obs_arr) < args.trajectory_min_length:
            obs_arr.clear()
            act_arr.clear()
            rew_arr.clear()
            done_arr.clear()
            info_arr.clear()
        else:
            break

    return {
        "observation": obs_arr, 
        "action": act_arr, 
        "reward": rew_arr, 
        "done": done_arr, 
        "info": info_arr,
        "episode_length": len(obs_arr)
    }

def sample_trajectories(n_trajectories: int, env: gym.Env, policy: Policy, args: SimpleNamespace) -> dict:
    trajectories = [sample_one_trajectory(env, policy, args) for i in range(n_trajectories)]
    result = {
        "observation": [], 
        "action": [], 
        "reward": [], 
        "done": [], 
        "info": [],
        "episode_length": []
    }
    for trajectory in trajectories:
        for k, v in trajectory.items():
            result[k].append(v)
    
    
    return result

def clip_trajectories(trajectories: dict, args: SimpleNamespace) -> dict:
    trajectories_length = [ len(l) for l in trajectories['observation']]
    clip_length = min(min(trajectories_length), args.trajectory_max_length)

    for k in trajectories.keys():
        trajectories[k] = [l[:clip_length] for l in trajectories[k]]
    return trajectories