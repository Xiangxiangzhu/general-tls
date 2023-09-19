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

config = {

    # ==========  env config ==========
    'net_file': "/nets/4x4-Lucas/4x4.net.xml",
    'route_file': "/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
    'use_gui': True,
    'delta_time': 5,
    'metric_period': 3600,  # 3600
    'min_green': 5,

    # ==========  learner config ==========
    'gamma': 0.85,  # also can be set to 0.95
    'epsilon': 0.9,
    'epsilon_min': 0.2,
    'epsilon_decay': 0.99,
    'start_lr': 0.00025,
    'episodes': 5,
    'algo': 'DQN',  # DQN
    'max_train_steps': int(1e6),
    'lr_decay_interval': 100,
    'epsilon_decay_interval': 100,
    'sample_batch_size':
        2048,  # also can be set to 32, which doesn't matter much.
    'learn_freq': 2,  # update parameters every 2 or 5 steps
    'decay': 0.995,  # soft update of double DQN
    'reward_normal_factor': 4,  # rescale the rewards, also can be set to 20,
    'train_count_log': 5,  # add to the tensorboard
    'is_show_log': False,  # print in the screen
    'step_count_log': 1000,

    # save checkpoint frequent episode
    'save_rate': 50,
    'save_dir': './save_model/general_light',
    'train_log_dir': './train_log/general_light',

    # memory config
    'memory_size': 20000,
    'begin_train_mmeory_size': int(3000)
}
