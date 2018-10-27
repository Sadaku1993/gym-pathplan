import gym
import gym_pathplan
import numpy as np

env = gym.make('Sample-v0')

observation = env.reset()

print('start', observation['state'])
print('goal', observation['goal'])

# env.show()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    # action = np.array([0.1, 0.])
    observation, reward, done, _  = env.step(action)
    # print('state',  observation['state'])
    # print('reward', reward)
    if done:
        print(observation['state'])
        print(reward)
        break
