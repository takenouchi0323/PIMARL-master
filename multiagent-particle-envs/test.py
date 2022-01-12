import make_env
import numpy as np
import json

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

np.set_printoptions(precision=2, suppress=True)

scenario = scenarios.load("simple_spread_n100.py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, cam_range=scenario.world_radius)

for episode in range(10):

    print("Episode", episode)
    obs_n = env.reset()
    env.render()
    print("Landmark locations")
    for i, landmark in enumerate(env.world.landmarks):
        print("Landmark", i, landmark.state.p_pos)
    print("Agent locations")
    for i, agent in enumerate(env.world.agents):
        print("Agent", i, agent.state.p_pos)
    for obs in obs_n:
        print(obs)
    
    done = False
    while not done:
    
        l = input("Enter actions as a comma-separated string: ")
        if l == '':
            actions = [np.array([np.random.uniform(-1.0,1.0) for __ in range(5)])
                       for _ in range(scenario.num_agents)]
        else:
            actions = list(map(int, l.split(',')))
        # print(actions)
        obs_n_next, reward_n, done_n, info_n = env.step(actions)
        env.render()
        print("Next obs")
        for obs in obs_n_next:
            print(obs)
        print("Reward", reward_n)
        print("Done", done_n)

    # print("Number of collisions", scenario.collisions/2)
