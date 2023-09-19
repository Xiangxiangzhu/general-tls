import argparse
import os
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from parl.utils import logger
from collections import defaultdict

# add parent path to sys.path
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from src import SumoEnvironment
from src.algorithm.ddqn import DDQN
from src.agents.parl_agent import Agent
from src.algorithm.config import config
from src.model.model import TLSModel
from src.algorithm.replay_buffer import ReplayMemory


def log_metrics(summary, datas, buffer_total_size):
    """ Log metrics
        """
    Q_loss, pred_values, target_values, max_v_show_values, train_count, lr, epsilon = datas
    metric = {
        'q_loss': Q_loss,
        'pred_values': pred_values,
        'target_values': target_values,
        'max_v_show_values': max_v_show_values,
        'lr': lr,
        'epsilon': epsilon,
        'memory_size': buffer_total_size,
        'train_count': train_count
    }
    # logger.info(metric)
    for key in metric:
        if key != 'train_count':
            summary.add_scalar(key, metric[key], train_count)


if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 30
    episodes = 4

    env = SumoEnvironment(
        net_file=parent_path + "/nets/4x4-Lucas/4x4.net.xml",
        route_file=parent_path + "/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=True,
        num_seconds=80000,
        min_green=5,
        delta_time=5,
    )

    _, obs_dim = next(iter(env.traffic_signals.items()))
    obs_dim = obs_dim.observation_space.shape[0]

    models = [
        TLSModel(ts.observation_space.shape[0], ts.action_space.n, config['algo'])
        for ts_id, ts in env.traffic_signals.items()
    ]

    algorithms = [DDQN(model, config) for model in models]

    n_agents = len(algorithms)

    initial_states = env.reset()
    ql_agents = {
        ts: Agent(
            algorithm,
            config
        )
        for algorithm, ts in zip(algorithms, env.ts_ids)
    }

    episode_count = 0
    step_forward = 0
    episodes_rewards = {}
    summarys = [
        SummaryWriter(os.path.join(config['train_log_dir'], str(agent_id)))
        for agent_id in range(n_agents)
    ]

    replay_buffers = [
        ReplayMemory(config['memory_size'], obs_dim, 0)
        for i in range(n_agents)
    ]

    obs = env.reset()

    with tqdm(total=config['episodes'], desc='[Training Model]') as pbar:
        while episode_count <= config['episodes']:
            step_count = 0
            while step_count < config['metric_period']:
                actions = {}
                for agent_id, ob in obs.items():
                    ob = ob.reshape(1, -1)
                    action = ql_agents[agent_id].sample(ob)
                    actions[agent_id] = action[0]
                rewards_list = []
                for _ in range(config['action_interval']):
                    step_count += 1
                    next_obs, rewards, dones, _ = env.step(actions)
                    rewards_list.append(rewards)

                #######
                # Use defaultdict to accumulate sums
                sums = defaultdict(int)
                # Count the number of dictionaries for each key
                counts = defaultdict(int)

                # Iterate over each dictionary and accumulate sums
                for d in rewards_list:
                    for key, value in d.items():
                        sums[key] += value
                        counts[key] += 1
                # Calculate averages
                rewards = {key: total / (counts[key] * config['reward_normal_factor']) for key, total in sums.items()}

                # calc the episodes_rewards and will add it to the tensorboard
                # Update episodes_rewards with rewards
                for key, value in rewards.items():
                    if key in episodes_rewards:
                        episodes_rewards[key] += value
                    else:
                        episodes_rewards[key] = value

                for agent_id, replay_buffer in enumerate(replay_buffers):
                    agent_name = env.ts_ids[agent_id]
                    replay_buffers[agent_id].append(
                        obs[agent_name], actions[agent_name], rewards[agent_name],
                        next_obs[agent_name], dones[agent_name])
                step_forward += 1
                obs = next_obs
                if len(replay_buffers[0]) >= config[
                    'begin_train_mmeory_size'] and step_forward % config[
                    'learn_freq'] == 0:
                    for agent_id, agent in enumerate(ql_agents):
                        sample_data = replay_buffers[agent_id].sample_batch(
                            config['sample_batch_size'])
                        train_obs, train_actions, train_rewards, train_next_obs, train_terminals = sample_data

                        Q_loss, pred_values, target_values, max_v_show_values, train_count, lr, epsilon = \
                            agent.learn(train_obs, train_actions, train_terminals, train_rewards, train_next_obs)
                        datas = [
                            Q_loss, pred_values, target_values,
                            max_v_show_values, train_count, lr, epsilon
                        ]
                        # tensorboard
                        if train_count % config['train_count_log'] == 0:
                            log_metrics(summarys[agent_id], datas,
                                        step_forward)
                if step_count % config['step_count_log'] == 0 and config[
                    'is_show_log']:
                    logger.info('episode_count: {}, step_count: {}, buffer_size: {}, buffer_size_total_size: {}.' \
                                .format(episode_count, step_count, len(replay_buffers[0]), step_forward))

            episode_count += 1
            avg_travel_time = env.world.eng.get_average_travel_time()
            obs = env.reset()
            for agent_id, summary in enumerate(summarys):
                summary.add_scalar('episodes_reward',
                                   episodes_rewards[agent_id], episode_count)
                # the avg travel time is same for all agents.
                summary.add_scalar('average_travel_time', avg_travel_time,
                                   episode_count)
            logger.info('episode_count: {}, average_travel_time: {}.'.format(
                episode_count, avg_travel_time))
            # reset to zeros
            episodes_rewards = np.zeros(n_agents)
            # save the model
            if episode_count % config['save_rate'] == 0:
                for agent_id, agent in enumerate(ql_agents):
                    save_path = "{}/agentid{}_episode_count{}.ckpt".format(
                        config['save_dir'], agent_id, episode_count)
                    agent.save(save_path)
            pbar.update(1)

    env.close()
