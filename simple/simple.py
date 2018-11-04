#coding:utf-8

'''
simple.py

random action test in Original Environment

author : Yudai Sadakuni
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

    if done:
        break
