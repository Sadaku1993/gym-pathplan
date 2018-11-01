import gym
import gym_pathplan
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Simple-v0')

observation = env.reset()

for _ in range(10000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    obstacle = np.where(observation['map']==1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(obstacle[0], obstacle[1], "og")
    plt.pause(0.01)

    if done:
        break
