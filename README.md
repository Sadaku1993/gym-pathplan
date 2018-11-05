# OpenAI Gym original environments

## Requirement 
- python2.7
- OpenAI Gym

## Building OpenAI Gym from source code

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

## Environment

```python
import gym
import gym-pathplan

env = gym.Make('Simple-v0')
env.reset()

for _ i in range(1000):
    env.render()
    observation, reward, done, _ = env.step(env.action_space.sample()) # take a random action

    if done:
        env.reset()
```

It should look someting like this

![demo](https://github.com/Sadaku1993/gym-pathplan/blob/master/sample.gif)


