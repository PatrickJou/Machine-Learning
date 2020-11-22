import gym
from gym.envs.registration import register


from .frozen_lake import *

__all__ = ['RewardingFrozenLakeEnv']

register(
    id='RewardingFrozenLakeWithRewards20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': True, 'is_slippery': False}
)


def get_rewarding_large_frozen_lake():
    return gym.make('RewardingFrozenLakeWithRewards20x20-v0')

