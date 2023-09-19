#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import parl
from parl.utils.scheduler import LinearDecayScheduler


class DDQN(parl.Algorithm):
    def __init__(self, model, config):

        self.model = model
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config['start_lr'])
        self.config = config
        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],
                                                 config['max_train_steps'])
        self.lr = config['start_lr']
        self.target_model = copy.deepcopy(model)
        self.device = torch.device("cuda:0" if torch.cuda.
                                   is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.train_count = 0

        self.epsilon = self.config['epsilon']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']

    def sample(self, obs):

        logits = self.model(obs)
        act_values = logits.cpu().detach().numpy()
        return logits

    def predict(self, obs):
        logits = self.model(obs)
        _, predict_actions = logits.max(-1)
        return predict_actions

    def sync_target(self, decay=0.995):
        # soft update
        self.model.sync_weights_to(self.target_model, decay)

    def learn(self, obs, actions, dones, rewards, next_obs):
        # Update the Q network from the memory buffer.
        if self.train_count > 0 and self.train_count % self.config[
                'lr_decay_interval'] == 0:
            self.lr = self.lr_scheduler.step(
                step_num=self.config['lr_decay_interval'])
        terminal = dones
        actions_onehot = F.one_hot(actions, self.model.act_dim)
        pred_values = torch.sum(self.model(obs) * actions_onehot, dim=-1)
        greedy_action = self.model(next_obs).max(dim=1, keepdim=True)[1]
        with torch.no_grad():
            # target_model for evaluation, using the double DQN
            max_v_show = self.target_model(next_obs).detach().max(1)[0]
            max_v = self.target_model(next_obs).gather(1, greedy_action)[:, 0]
            assert max_v.shape == rewards.shape
            target = rewards + (1 - terminal) * self.config['gamma'] * max_v
        Q_loss = 0.5 * F.mse_loss(pred_values, target)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        self.optimizer.zero_grad()
        Q_loss.backward()

        # clip the grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=40.0)
        self.optimizer.step()
        self.train_count += 1
        if self.epsilon > self.epsilon_min and self.train_count % self.config[
                'epsilon_decay_interval'] == 0:
            self.epsilon *= self.epsilon_decay
        return Q_loss, pred_values.mean(), target.mean(), max_v_show.mean(
        ), self.train_count, self.lr, self.epsilon
