from stable_baselines3.dqn.policies import DQNPolicy

import torch as th


class MaskedDQNPolicy(DQNPolicy):

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.q_net(obs)

        action_mask = self.observation_space.my_action_mask.to(q_values.device)
        q_values[:, action_mask == 0] = float('-inf')

        action = q_values.argmax(dim=1).reshape(-1)
        return action
