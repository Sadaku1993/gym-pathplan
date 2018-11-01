from gym.envs.registration import register

register(
    id='Sample-v0',
    entry_point='gym_pathplan.envs:Sample',
)

register(
    id='Simple-v0',
    entry_point='gym_pathplan.envs:Simple',
)
