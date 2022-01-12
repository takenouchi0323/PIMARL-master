import csv
import torch

def adjust_learning_rate(optimizer, steps, max_steps, start_decrease_step, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if steps > start_decrease_step:
        lr = init_lr * (1 - ((steps - start_decrease_step) / (max_steps - start_decrease_step)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def dict2csv(output_dict, f_name):
    with open(f_name, mode='w') as f:
        writer = csv.writer(f, delimiter=",")
        for k, v in output_dict.items():
            v = [k] + v
            writer.writerow(v)


def n_actions(action_spaces):
    """
    :param action_space: list
    :return: n_action: list
    """
    n_actions = []
    from gym import spaces
    from multiagent.environment import MultiDiscrete
    for action_space in action_spaces:
        if isinstance(action_space, spaces.discrete.Discrete):
            n_actions.append(action_space.n)
        elif isinstance(action_space, MultiDiscrete):
            total_n_action = 0
            one_agent_n_action = 0
            for h, l in zip(action_space.high, action_space.low):
                total_n_action += int(h - l + 1)
                one_agent_n_action += int(h - l + 1)
            n_actions.append(one_agent_n_action)
        else:
            raise NotImplementedError
    return n_actions


def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    return env





def make_env_vec(scenario_name, arglist, benchmark=False):
    from multiagent.environment_vec import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius)
    return env


def copy_actor_policy(s_agent, t_agent):
    if hasattr(s_agent, 'actors'):
        for i in range(s_agent.n_group):
            state_dict = s_agent.actors[i].state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            t_agent.actors[i].load_state_dict(state_dict)
        t_agent.actors_params, t_agent.critic_params = None, None
    else:
        state_dict = s_agent.actor.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        t_agent.actor.load_state_dict(state_dict)
        t_agent.actor_params, t_agent.critic_params = None, None


def compute_mf_action(actions):
    """
    actions: [n_agents, dim_a]
    Returns
    [n_agents, dim_a]
    """
    n_agents = actions.shape[0]
    actions_mf = np.zeros_like(actions)

    for idx in range(n_agents):
        others = np.delete(actions, idx, 0)
        actions_mf[idx] = np.mean(others, axis=0)
    
    return actions_mf


def compute_mf_action_batch(actions, n_agents):
    """
    actions: [batch, n_agents * dim_a]
    Returns
    [batch, n_agents * dim_a]
    """
    batch = actions.shape[0]
    actions = actions.view(batch, n_agents, -1)  # [b, n, dim_a]
    actions_mf = torch.zeros_like(actions)  # [b, n, dim_a]

    for idx in range(n_agents):
        clone = actions.clone()
        clone[:, idx, :] = 0
        mean_others = torch.mean(clone, dim=1)
        actions_mf[:, idx, :] = mean_others
    
    actions_mf = actions_mf.view(batch, -1)

    return actions_mf
