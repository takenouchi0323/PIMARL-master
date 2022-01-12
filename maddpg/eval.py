from util.utils import make_env, dict2csv
import numpy as np
import contextlib
import torch
#from ckpt_plot.plot_curve import plot_result
import os
import time


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_buffer(agent, memory, config):

    print("Evaluating buffers")
    best_reward=-10000000000
    best_index=None

    for i in range(config.alg.replay_size//config.env.num_steps):
        #print(memory.pop(i))
        #print(eval_buffer(eval_agent, , exp_save_dir, config))
        memory_single=memory.pop(i*config.env.num_steps)

        eval_env = make_env(config.env.scenario, config)
        eval_env.seed(config.main.seed + 100)
        eval_rewards = []

        with temp_seed(config.main.seed):
            for n_eval in range(500):#(config.env.num_eval_runs):
                obs_n = eval_env.reset()
                episode_reward = 0
                episode_step = 0
                n_agents = eval_env.n
                while True:
                    action_n = agent.select_action(torch.Tensor(obs_n), memory=memory_single, train=False, action_noise=True).squeeze().cpu().numpy()
                    next_obs_n, reward_n, done_n, _ = eval_env.step(action_n)
                    episode_step += 1
                    terminal = (episode_step >= config.env.num_steps)
                    episode_reward += np.sum(reward_n)  # sum of all agents' rewards
                    obs_n = next_obs_n
                    if done_n[0] or terminal:
                        eval_rewards.append(episode_reward)
                        break

            eval_env.close()
        
        if np.mean(eval_rewards)>best_reward:
            best_reward=np.mean(eval_rewards)
            best_index=i*config.env.num_steps
        
        #print(np.mean(eval_rewards))

    return best_index#np.mean(eval_rewards)


def eval_model(agent, tr_log, config, latest_eval_rewards, best_eval_reward, memory, index=None):
    """
    A version that does not rely on multiprocessing.
    Saving of final model is done in main.py, not here.

    Args:
        agent: agent that takes actions
        tr_log: dict of information to log
        config: configdict
        latest_eval_rewards: a deque of previous average eval rewards
        best_eval_reward: max eval reward seen during training

    Returns:
        updated latest_eval_rewards, best_eval_reward
    """
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
            while True:
                action_n = agent.select_action(
                    torch.Tensor(obs_n), memory=memory_eval, train=False, action_noise=True).squeeze().cpu().numpy()
                next_obs_n, reward_n, done_n, _ = eval_env.step(action_n)
                episode_step += 1
                terminal = (episode_step >= config.env.num_steps)
                episode_reward += np.sum(reward_n)  # sum of all agents' rewards
                obs_n = next_obs_n
                if done_n[0] or terminal:
                    eval_rewards.append(episode_reward)
                    if n_eval % 100 == 0:
                        print('test reward', episode_reward)
                    break
        if np.mean(eval_rewards) > best_eval_reward:
            best_eval_reward = np.mean(eval_rewards)
            torch.save(agent.actor.state_dict(), os.path.join(tr_log['exp_save_dir'], 'agents_best_actor.ckpt'))
            torch.save(agent.critic.state_dict(), os.path.join(tr_log['exp_save_dir'], 'agents_best_critic.ckpt'))
            torch.save({'agents': agent},
                       os.path.join(tr_log['exp_save_dir'], 'agents_best.ckpt'))

        latest_eval_rewards.append(np.mean(eval_rewards))
        time_elapsed = time.time() - tr_log['start_time']
        print("========================================================")
        print(
            "Episode: {}, total numsteps: {}, {} eval runs, total time: {} s".
                format(tr_log['i_episode'], tr_log['total_numsteps'],
                       config.env.num_eval_runs, time_elapsed))
        print("Eval reward: avg {} std {} | Avg last 10 policies: {} | Best reward {}".format(
            np.mean(eval_rewards), np.std(eval_rewards),
            np.mean(latest_eval_rewards), best_eval_reward))
        s = '%d,%d,%d,' % (tr_log['i_episode'], tr_log['total_numsteps'],
                           int(time_elapsed))
        s += '{:.3e},{:.3e},'.format(tr_log['value_loss'], tr_log['policy_loss'])
        s += '{:.3e},{:.3e},{:.3e}, {:.3e}\n'.format(
            np.mean(eval_rewards), np.mean(latest_eval_rewards), best_eval_reward, np.std(eval_rewards))
        with open(os.path.join(tr_log['exp_save_dir'], 'eval_log.csv'), 'a') as f:
            f.write(s)

        eval_env.close()

    return latest_eval_rewards, best_eval_reward