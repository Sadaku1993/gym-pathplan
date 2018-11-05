from gym.envs.registration import register

register(
    id='Simple-v0',
    entry_point='gym_pathplan.envs:Simple',
)
