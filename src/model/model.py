import torch
import torch.nn as nn
import torch.nn.functional as F

import parl


class TLSModel(parl.Model):
    def __init__(self, obs_dim, act_dim, algo='DQN'):
        super(TLSModel, self).__init__()

        hid1_size = 20
        hid2_size = 20
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # embedding_size = 10
        # self.current_phase_embedding = nn.Embedding(act_dim, embedding_size)

        self.algo = algo
        if self.algo == 'Dueling':
            self.fc1_adv = nn.Linear(obs_dim, hid1_size)
            self.fc1_val = nn.Linear(obs_dim, hid1_size)

            self.fc2_adv = nn.Linear(hid1_size, hid2_size)
            self.fc2_val = nn.Linear(hid1_size, hid2_size)

            self.fc3_adv = nn.Linear(hid2_size, self.act_dim)
            self.fc3_val = nn.Linear(hid2_size, 1)
        else:
            self.fc1 = nn.Linear(obs_dim, hid1_size)
            self.fc2 = nn.Linear(hid1_size, hid2_size)
            self.fc3 = nn.Linear(hid2_size, self.act_dim)

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # print("################################")
        # print("tl model is ", x)
        # print("################################")
        # cur_phase = x
        # cur_phase_em = self.current_phase_embedding(cur_phase)
        # x = x[:, :-1]
        if self.algo == 'Dueling':
            fc1_a = F.relu(self.fc1_adv(x))
            fc1_v = F.relu(self.fc1_val(x))

            fc2_a = F.relu(self.fc2_adv(fc1_a))
            fc2_v = F.relu(self.fc2_val(fc1_v))

            # fc2_a = torch.cat((fc2_a, cur_phase_em), dim=-1)
            # fc2_v = torch.cat((fc2_v, cur_phase_em), dim=-1)
            As = self.fc3_adv(fc2_a)
            V = self.fc3_val(fc2_v)
            Q = As + (V - As.mean(dim=1, keepdim=True))
        else:
            x1 = F.relu(self.fc1(x))
            x2 = F.relu(self.fc2(x1))
            # x2 = torch.cat((x2, cur_phase_em), dim=-1)
            Q = self.fc3(x2)
        return Q
