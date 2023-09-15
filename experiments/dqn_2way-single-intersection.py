import os
import sys

# import gym
import numpy as np
# from gym import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from sumo_rl import SumoEnvironment
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.dqn.dqn import DQN

# add parent path to sys.path
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from stable_baselines3.dqn.policies import DQNPolicy

import torch as th

if __name__ == "__main__":
    env = SumoEnvironment(
        net_file=parent_path + "/nets/2way-single-intersection/single-intersection.net.xml",
        route_file=parent_path + "/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=True,
        num_seconds=100000,
    )


    class MaskedDQNPolicy(DQNPolicy):

        def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
            q_values = self.q_net(obs)

            # Build action mask
            action_mask = th.ones(self.action_space.n, device=q_values.device)
            action_mask[0:7] = 0
            q_values[:, action_mask == 0] = float('-inf')

            action = q_values.argmax(dim=1).reshape(-1)
            print("action is ", action)
            return action


    model = DQN(
        env=env,
        policy=MaskedDQNPolicy,  # "MlpPolicy", MaskedDQNPolicy
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=100000)
