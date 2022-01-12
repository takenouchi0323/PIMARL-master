import sys
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable, grad
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from models import model_factory
from replay_memory import ReplayMemory, Transition
import random


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def adjust_lr(optimizer, init_lr, episode_i, num_episode, start_episode):
    if episode_i < start_episode:
        return init_lr
    lr = init_lr * (1 - (episode_i - start_episode) / (num_episode - start_episode))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, n_agents, use_allobs=False, noise=None):
        super(Actor, self).__init__()
        self.n_agents = n_agents
        self.use_allobs = use_allobs
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs, memory, train, use_buffer=None, noise=None):
        """
        inputs:[n_agents*batch_size, num_obs(simple_spreadでは26)]
        state_all:[n_agents*batch_size, n_agnets, num_obs]
        x:[n_agents*batch_size, num_actions(simple_spreadでは5)]
        """
        x = inputs
        if self.use_allobs:
            batch_size=inputs.size()[0]//self.n_agents
            state=inputs
            rolled=torch.roll(inputs, -1, dims=1)
            for _ in range(self.n_agents-1):
                state=torch.cat((state, rolled), 1)
                rolled=torch.roll(rolled, -1, dims=1)
            state=state.view(batch_size*self.n_agents, -1)
            x=state

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.mu(x)
        return x

class Actor_DeepSet(nn.Module):
    """A deep set."""
    def __init__(self, hidden_size, num_inputs, num_outputs, n_agents):
        super(Actor_DeepSet, self).__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.n_agents = n_agents

        self.h1 = nn.Linear(self.num_inputs//self.n_agents, self.hidden_size)
        self.h1_other = nn.Linear(self.num_inputs//self.n_agents, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, self.num_outputs)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1) 

    def forward(self, inputs, memory, train=False, use_buffer=None, noise=None):
        """
        inputs:[n_agents*batch_size, num_obs(simple_spreadでは26)]
        state:[n_agents*batch_size, num_obs]
        state_other:[n_agents*batch_size, n_agents-1, num_obs]
        V:[n_agents*batch_size, num_actions(simple_spreadでは5)]
        """

        batch_size=inputs.size()[0]//self.n_agents
        if use_buffer == None:
            print("select use_buffer option")
            sys.exit()
        if use_buffer:#評価時にバッファーからとるかどうか
            if not train:
                #memoryからstate_other:[n_agents*batch_size, n_agents-1, num_obs]
                #print(memory.__len__())
                if memory.__len__()==5:
                    inputs2=memory[0][0].view(batch_size*self.n_agents, -1)
                    #print(batch_size)
                    #print(inputs2)
                else:
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    inputs2=batch[0][0].view(batch_size*self.n_agents, -1)
            else:
                choice=random.randint(0, 100)
                if choice<noise and memory.__len__()>batch_size*self.n_agents:
                    #print(noise)
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    batch=torch.stack(batch[0])
                    batch=torch.Tensor(batch).to(inputs.device)
                    inputs2=batch.view(batch_size*self.n_agents, -1)
                else:
                   inputs2=inputs.view(batch_size*self.n_agents, -1)
        else:
            inputs2=inputs.view(batch_size*self.n_agents, -1)

        state_other=[]
        state_other=torch.Tensor(state_other).to(inputs.device)
        rolled=torch.roll(inputs2, -1, dims=1)
        for _ in range(self.n_agents):
            state_other=torch.cat((state_other, rolled.view(batch_size, self.n_agents, -1)[:,1:]), 1)
            rolled=torch.roll(rolled, -1, dims=1)
        state_other=state_other.view(batch_size*self.n_agents, self.n_agents-1, -1)
        
        state=inputs
        
        h1=self.h1(state)
        h1_other = self.h1_other(state_other)
        h1 = F.relu(h1)
        h1_other=F.relu(h1_other)
        h1_summed = torch.sum(h1_other, 1)
        h1_summed /= self.n_agents
        h1 = torch.cat((h1, h1_summed), 1)
        h2 = self.h2(h1)
        h2 = F.relu(h2)

        # Compute V
        V = self.V(h2)

        return V    


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, num_agents, critic_type='mlp'):
        super(Critic, self).__init__()

        self.num_agents = num_agents
        self.critic_type = critic_type
        sa_dim = int((num_inputs + num_outputs) / num_agents)
        self.net_fn = model_factory.get_model_fn(critic_type)
        self.net = self.net_fn(sa_dim, num_agents, hidden_size)

    def forward(self, inputs, actions):
        bz = inputs.size()[0]
        s_n = inputs.view(bz, self.num_agents, -1)
        a_n = actions.view(bz, self.num_agents, -1)
        x = torch.cat((s_n, a_n), dim=2)
        V = self.net(x)
        return V


class DDPG(object):
    def __init__(self, gamma, tau, actor_hidden_size, critic_hidden_size, obs_dim, n_action, n_agent, actor_lr, critic_lr,
                 fixed_lr, critic_type, actor_type, train_noise, num_episodes, num_steps,
                 target_update_mode='soft', device='cpu', use_allobs=False, use_buffer=None, noise=None):
        self.device = device
        self.obs_dim = obs_dim
        self.n_agent = n_agent
        self.n_action = n_action
        self.use_buffer = use_buffer
        self.noise = noise

        self.critic = Critic(critic_hidden_size, obs_dim*self.n_agent, n_action * n_agent, n_agent, critic_type).to(self.device)
        self.critic_target = Critic(critic_hidden_size, obs_dim*self.n_agent, n_action * n_agent, n_agent, critic_type).to(self.device)
        critic_n_params = sum(p.numel() for p in self.critic.parameters())

        if actor_type == 'deepset':
            self.actor = Actor_DeepSet(actor_hidden_size, obs_dim*n_agent, n_action, n_agent).to(self.device)
            self.actor_target = Actor_DeepSet(actor_hidden_size, obs_dim*n_agent, n_action, n_agent).to(self.device)
            self.actor_perturbed = Actor_DeepSet(actor_hidden_size, obs_dim*n_agent, n_action, n_agent)
        else:
            self.actor = Actor(actor_hidden_size, obs_dim*n_agent if use_allobs else obs_dim, n_action, n_agent, use_allobs).to(self.device)
            self.actor_target = Actor(actor_hidden_size, obs_dim*n_agent if use_allobs else obs_dim, n_action, n_agent, use_allobs).to(self.device)
            self.actor_perturbed = Actor(actor_hidden_size, obs_dim*n_agent if use_allobs else obs_dim, n_action, n_agent, use_allobs)
        
        actor_n_params = sum(p.numel() for p in self.actor.parameters())

        print('# of critic params', critic_n_params)
        print('# of actor  params', actor_n_params)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr, weight_decay=0)
        self.fixed_lr = fixed_lr
        self.init_act_lr = actor_lr
        self.init_critic_lr = critic_lr
        self.num_episodes = num_episodes
        self.start_episode = 0
        self.num_steps = num_steps
        self.actor_scheduler = LambdaLR(self.actor_optim, lr_lambda=self.lambda1)
        self.critic_scheduler = LambdaLR(self.critic_optim, lr_lambda=self.lambda1)
        self.gamma = gamma
        self.tau = tau
        self.train_noise = train_noise

        self.target_update_mode = target_update_mode
        self.actor_params = self.actor.parameters()
        self.critic_params = self.critic.parameters()
        # Make sure target is with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)



    def adjust_lr(self, i_episode):
        adjust_lr(self.actor_optim, self.init_act_lr, i_episode, self.num_episodes, self.start_episode)
        adjust_lr(self.critic_optim, self.init_critic_lr, i_episode, self.num_episodes, self.start_episode)

    def lambda1(self, step):
        start_decrease_step = ((self.num_episodes / 2)
                               * self.num_steps) / 100
        max_step = (self.num_episodes * self.num_steps) / 100
        return 1 - ((step - start_decrease_step) / (
                max_step - start_decrease_step)) if step > start_decrease_step else 1

    def select_action(self, state, memory, train, action_noise=None, grad=False):

        self.actor.eval()

        mu = self.actor((Variable(state)), memory, train, self.use_buffer, self.noise)

        self.actor.train()
        if not grad:
            mu = mu.data

        if action_noise:
            noise = np.log(-np.log(np.random.uniform(0, 1, mu.size())))
            try:
                mu -= torch.Tensor(noise).to(self.device)
            except (AttributeError, AssertionError):
                mu -= torch.Tensor(noise)

        action = F.softmax(mu, dim=1)
        if not grad:
            return action
        else:
            return action, mu
        
    def update_critic_parameters(self, batch, memory, shuffle=None):
        state_batch = Variable(torch.cat(batch.state)).to(self.device)
        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward)).to(self.device)
        mask_batch = Variable(torch.cat(batch.mask)).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        next_action_batch = self.select_action(next_state_batch.view(-1, self.obs_dim), memory, True, action_noise=self.train_noise)
        next_action_batch = next_action_batch.view(-1, self.n_action * self.n_agent)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch[:, 0].unsqueeze(1)
        mask_batch = mask_batch[:, 0].unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = ((state_action_batch - expected_state_action_batch) ** 2).mean()
        value_loss.backward()
        unclipped_norm = clip_grad_norm_(self.critic_params, 0.5) #もとは0.5
        self.critic_optim.step()

        if self.target_update_mode == 'soft':
            soft_update(self.critic_target, self.critic, self.tau)
        elif self.target_update_mode == 'hard':
            hard_update(self.critic_target, self.critic)

        loss = value_loss.item()

        del value_loss, state_batch, action_batch, reward_batch, mask_batch, next_state_batch, next_action_batch, next_state_action_values, expected_state_action_batch, state_action_batch
        return loss, unclipped_norm

    def update_actor_parameters(self, batch, memory):
        state_batch = Variable(torch.cat(batch.state)).to(self.device)

        self.actor_optim.zero_grad()
        action_batch_n, logit = self.select_action(state_batch.view(-1, self.obs_dim), memory, True, action_noise=self.train_noise, grad=True)
        action_batch_n = action_batch_n.view(-1, self.n_action * self.n_agent)

        policy_loss = -self.critic(state_batch, action_batch_n)
        policy_loss = policy_loss.mean() + 1e-3 * (logit ** 2).mean()
        
        policy_loss.backward()
        clip_grad_norm_(self.actor_params, 0.1)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        loss = policy_loss.item()

        del action_batch_n, state_batch, policy_loss, logit

        return loss

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    @property
    def actor_lr(self):
        return self.actor_optim.param_groups[0]['lr']