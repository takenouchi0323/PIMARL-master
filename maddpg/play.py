import argparse
import copy
import json
import math
import os, sys
import random
import time
import torch
import gc
from collections import namedtuple
from collections import deque
from itertools import count

import numpy as np
import contextlib

from ddpg_vec import DDPG, hard_update
from eval import eval_buffer
from replay_memory import ReplayMemory, Transition
from util.utils import *
from util.utils import copy_actor_policy, n_actions as util_n_actions

from torch.optim import Adam

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_model(agent, memory, config, index=None):
    print('=================== start eval ===================')
    eval_env = make_env(config.env.scenario, config)
    eval_env.seed(config.main.seed + 10)
    eval_rewards = []
    if config.alg.eval_buffer:
        memory_eval=memory.pop(index)
    else:
        memory_eval=memory

    with temp_seed(config.main.seed):
        for n_eval in range(config.env.num_eval_runs):
            obs_n = eval_env.reset()
            episode_reward = 0
            episode_step = 0
            n_agents = eval_env.n
            if config.main.render_episodes != None:
                time.sleep(0.5)
            while True:
                if config.main.render_episodes != None:
                    eval_env.render()
                    time.sleep(0.05)
                action_n = agent.select_action(
                    torch.Tensor(obs_n), memory=memory_eval, train=False, action_noise=True).squeeze().cpu().numpy()
                next_obs_n, reward_n, done_n, _ = eval_env.step(action_n)
                episode_step += 1
                terminal = (episode_step >= config.env.num_steps)
                episode_reward += np.sum(reward_n)  # sum of all agents' rewards
                obs_n = next_obs_n
                if done_n[0] or terminal:
                    eval_rewards.append(episode_reward)
                    if n_eval % (config.env.num_eval_runs//10) == 0:
                        print('test reward', episode_reward)
                    break
        print("========================================================")
        print("Eval reward: avg {} std {}".format(
            np.mean(eval_rewards), np.std(eval_rewards)))
        eval_env.close()
    return np.mean(eval_rewards)


def play(config):

    torch.set_num_threads(1)
    #device = torch.device("cuda:0" if torch.cuda.is_available() and config.main.cuda else "cpu")
    #print(device)

    #ここでのenvは，action spaceなどを取得するためのもので実際には使わない
    env = make_env(config.env.scenario, None)
    n_agents = env.n
    env.seed(config.main.seed)
    random.seed(config.main.seed)
    np.random.seed(config.main.seed)
    torch.manual_seed(config.main.seed)
    exp_save_dir = os.path.join(config.main.save_dir, config.main.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)

    n_actions = util_n_actions(env.action_space)
    obs_dim=env.observation_space[0].shape[0]

    useless=0
    eval_agent = DDPG(config.alg.gamma, useless, config.nn.actor_hidden_size, config.nn.critic_hidden_size,
                        obs_dim, n_actions[0],
                        n_agents, useless,
                        useless, useless,
                        config.alg.critic_type, config.alg.actor_type,
                        useless, config.env.num_episodes,
                        config.env.num_steps,
                        useless, 'cpu', 
                        useless, config.alg.use_buffer)
    """
    eval_agent = DDPG(config.alg.gamma, config.alg.tau, config.nn.actor_hidden_size, config.nn.critic_hidden_size,
                        obs_dim, n_actions[0],
                        n_agents, config.alg.actor_lr,
                        config.alg.critic_lr, config.alg.fixed_lr,
                        config.alg.critic_type, config.alg.actor_type,
                        config.alg.train_noise, config.env.num_episodes,
                        config.env.num_steps,
                        config.alg.target_update_mode, 'cpu', 
                        config.alg.use_allobs, config.alg.use_buffer)
    """
    
    eval_agent.actor.load_state_dict(torch.load(os.path.join(exp_save_dir, 'agents_actor.ckpt')))
    eval_agent.critic.load_state_dict(torch.load(os.path.join(exp_save_dir, 'agents_critic.ckpt')))
    memory = ReplayMemory(config.alg.replay_size)
    memory.load(exp_save_dir)

    index=None
    if config.alg.eval_buffer:
        index=eval_buffer(eval_agent, memory, config)
    eval_model(eval_agent, memory, config, index)

    time.sleep(5)


if __name__ == '__main__':

    #parser = argparse.ArgumentParser()
    #parser.add_argument('alg', type=str, 
    #                    choices=['pic', 'deepset', 'deepset2', 'mlp'])
    #args = parser.parse_args()

    """
    if args.alg == 'pic':
        from configs import config_pic
        config = config_pic.get_config()
    elif args.alg =='deepset':
        from configs import config_deepset
        config = config_deepset.get_config()
    elif args.alg =='deepset2':
        from configs import config_deepset2
        config = config_deepset2.get_config()
    elif args.alg == 'mlp':
        from configs import config_mlp
        config = config_mlp.get_config()
    """
    from configs import config_play
    config = config_play.get_config()

    play(config)