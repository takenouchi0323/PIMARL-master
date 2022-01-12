import argparse
import copy
import json
import math
import os
import random
import time
import torch
import gc
from collections import namedtuple
from collections import deque
from itertools import count

import numpy as np

from ddpg_vec import DDPG, hard_update
from eval import eval_model, eval_buffer
from replay_memory import ReplayMemory, Transition
from util.utils import *
from util.utils import copy_actor_policy, n_actions as util_n_actions

def train(config):

    if config.main.exp_name is None:
        config.main.exp_name = (config.env.scenario + '_' + config.alg.critic_type +
                                '_' + config.alg.target_update_mode + '_hiddensize' +
                                str(config.nn.actor_hidden_size) + '_' + str(config.main.seed))

    torch.set_num_threads(1)
    device = torch.device(config.main.cuda_num if torch.cuda.is_available() and config.main.cuda else "cpu")
    print(device)
    env = make_env(config.env.scenario, None)
    n_agents = env.n
    env.seed(config.main.seed)
    random.seed(config.main.seed)
    np.random.seed(config.main.seed)
    torch.manual_seed(config.main.seed)

    n_actions = util_n_actions(env.action_space)
    obs_dim=env.observation_space[0].shape[0]

    agent = DDPG(config.alg.gamma, config.alg.tau, config.nn.actor_hidden_size, config.nn.critic_hidden_size,
                    obs_dim, n_actions[0],
                    n_agents, config.alg.actor_lr, config.alg.critic_lr,
                    config.alg.fixed_lr, config.alg.critic_type,
                    config.alg.actor_type, config.alg.train_noise, config.env.num_episodes,
                    config.env.num_steps,
                    config.alg.target_update_mode, device, 
                    config.alg.use_allobs, config.alg.use_buffer, config.alg.noise)
    eval_agent = DDPG(config.alg.gamma, config.alg.tau, config.nn.actor_hidden_size, config.nn.critic_hidden_size,
                        obs_dim, n_actions[0],
                        n_agents, config.alg.actor_lr,
                        config.alg.critic_lr, config.alg.fixed_lr,
                        config.alg.critic_type, config.alg.actor_type,
                        config.alg.train_noise, config.env.num_episodes,
                        config.env.num_steps,
                        config.alg.target_update_mode, 'cpu', 
                        config.alg.use_allobs, config.alg.use_buffer, config.alg.noise)
    memory = ReplayMemory(config.alg.replay_size)

    rewards = []
    total_numsteps = 0
    updates = 0
    exp_save_dir = os.path.join(config.main.save_dir, config.main.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)

    with open(os.path.join(exp_save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    header = 'episode,Steps,time,q_loss,p_loss,Average Reward,final,abs\n'
    with open(os.path.join(exp_save_dir, 'eval_log.csv'), 'w') as f:
        f.write(header)
    latest_eval_rewards = deque(maxlen=10)

    best_eval_reward = -np.inf
    start_time = time.time()
    copy_actor_policy(agent, eval_agent)
    #torch.save({'agents': eval_agent}, os.path.join(exp_save_dir, 'agents_best.ckpt'))
    torch.save(eval_agent.actor.state_dict(), os.path.join(exp_save_dir, 'agents_best_actor.ckpt'))
    torch.save(eval_agent.critic.state_dict(), os.path.join(exp_save_dir, 'agents_best_critic.ckpt'))
    best_eval_reward = -np.inf

    for i_episode in range(config.env.num_episodes):

        obs_n = env.reset()
        episode_reward = 0
        episode_step = 0
        if config.main.render_episodes != None:
            if i_episode>config.main.render_episodes:
                time.sleep(0.5)
        while True:
            if config.main.render_episodes != None:
                if i_episode>config.main.render_episodes:
                    env.render()
                    time.sleep(0.05)
            obs_tensor = torch.Tensor(obs_n).to(device)
            action_tensor = agent.select_action(obs_tensor, memory=memory, train=True, action_noise=True)
            action_n = action_tensor.squeeze().cpu().numpy()

            del obs_tensor, action_tensor
            
            next_obs_n, reward_n, done_n, info = env.step(action_n)
            total_numsteps += 1
            episode_step += 1
            terminal = (episode_step >= config.env.num_steps)

            action = torch.Tensor(action_n).view(1, -1)
            mask = torch.Tensor([[not done for done in done_n]])
            next_x = torch.Tensor(np.concatenate(next_obs_n, axis=0)).view(1, -1)
            reward = torch.Tensor([reward_n])
            x = torch.Tensor(np.concatenate(obs_n, axis=0)).view(1, -1)
            memory.push(x, action, mask, next_x, reward)
            episode_reward += np.sum(reward_n)
            obs_n = next_obs_n

            if len(memory) > config.alg.batch_size:
                if total_numsteps % config.alg.steps_per_actor_update == 0:
                    for _ in range(config.alg.actor_updates_per_step):
                        transitions = memory.sample(config.alg.batch_size)
                        batch = Transition(*zip(*transitions))
                        policy_loss = agent.update_actor_parameters(batch, memory)
                        updates += 1
                if total_numsteps % config.alg.steps_per_critic_update == 0:
                    value_losses = []
                    for _ in range(config.alg.critic_updates_per_step):
                        transitions = memory.sample(config.alg.batch_size)
                        batch = Transition(*zip(*transitions))
                        value_losses.append(
                            agent.update_critic_parameters(batch, memory)[0])
                        updates += 1
                    
                    value_loss = np.mean(value_losses)
                    if config.alg.target_update_mode == 'episodic':
                        hard_update(agent.critic_target, agent.critic)

            if done_n[0] or terminal:
                episode_step = 0
                break
        if not config.alg.fixed_lr:
            agent.adjust_lr(i_episode)
        rewards.append(episode_reward)
        if (i_episode + 1) % config.env.eval_freq == 0:
            tr_log = {'num_adversary': 0, 'exp_save_dir': exp_save_dir,
                      'total_numsteps': total_numsteps,
                      'value_loss': value_loss, 'policy_loss': policy_loss,
                      'i_episode': i_episode, 'start_time': start_time}
            copy_actor_policy(agent, eval_agent)
            
            index=None
            if config.alg.eval_buffer:
                index=eval_buffer(eval_agent, memory, config)
            latest_eval_rewards, best_eval_reward = eval_model(eval_agent, tr_log, config, latest_eval_rewards, best_eval_reward, memory, index)

    env.close()
    time.sleep(5)

    torch.save(eval_agent.actor.state_dict(), os.path.join(tr_log['exp_save_dir'], 'agents_actor.ckpt'))
    torch.save(eval_agent.critic.state_dict(), os.path.join(tr_log['exp_save_dir'], 'agents_critic.ckpt'))
    memory.save(exp_save_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('alg', type=str, 
                        choices=['gcn', 'deepset', 'deepset2'])
    args = parser.parse_args()

    if args.alg == 'gcn':
        from configs import config_gcn
        config = config_gcn.get_config()
    elif args.alg =='deepset':
        from configs import config_deepset
        config = config_deepset.get_config()
    elif args.alg =='deepset2':
        from configs import config_deepset2
        config = config_deepset2.get_config()

    train(config)