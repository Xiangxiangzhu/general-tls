import os
import sys

# import gym
import numpy as np
from gym import spaces
from sumo_rl import SumoEnvironment
import gymnasium as gym
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


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ActionMaskWrapper, self).__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, seed=None, **kwargs):
        obs = self.env.reset()
        action_mask = self.compute_action_mask()
        # return {'observation': obs, 'action_mask': action_mask}
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        action_mask = self.compute_action_mask()
        return {'observation': obs, 'action_mask': action_mask}, reward, done, info

    def compute_action_mask(self):
        # Implement your logic to compute the action mask based on the current state of the environment
        # For example:
        mask = np.ones(self.action_space.n)
        # mask[invalid_action_indices] = 0
        return mask


from stable_baselines3.dqn.policies import DQNPolicy

import torch as th


class MaskedDQNPolicy(DQNPolicy):
    def _predict(self, obs, deterministic=True) -> th.Tensor:
        # Assuming the last part of the observation tensor is the action mask
        # and the rest is the actual observation. Adjust as per your implementation.
        if not isinstance(obs, dict):
            return super()._predict(obs, deterministic=deterministic)

        action_mask, actual_obs = obs['action_mask'], obs['observation']
        # Get Q-values from the parent class
        q_values = super()._predict(actual_obs, deterministic=deterministic)

        # Mask invalid actions by setting their Q-values to a large negative number
        q_values[action_mask == 0] = float('-inf')

        return q_values


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file=parent_path + "/nets/2way-single-intersection/single-intersection.net.xml",
        route_file=parent_path + "/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=True,
        num_seconds=100000,
    )
    # env = ActionMaskWrapper(env)

    model = DQN(
        env=env,
        policy="MlpPolicy",  # "MlpPolicy", MaskedDQNPolicy
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=100000)
