#coding:utf-8

'''
check.py

check environment action space and observation space

author : Yudai Sadakuni
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gym
import gym_pathplan
import numpy as np

env = gym.make('Simple-v0')

## action info
print("action space", env.action_space)
print("action space low",  env.action_space.low)
print("action space high", env.action_space.high)
print(" ")

## observation info
print("observation space", env.observation_space)

print("state low", env.observation_space.spaces['state'].low)
print("state high", env.observation_space.spaces['state'].high)

print("lidar low", env.observation_space.spaces['lidar'].low)
print("lidar high", env.observation_space.spaces['lidar'].high)

print("goal low", env.observation_space.spaces['goal'].low)
print("goal high", env.observation_space.spaces['goal'].high)
