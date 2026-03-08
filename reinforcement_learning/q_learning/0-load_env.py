#!/usr/bin/env python3
"""Module that loads the FrozenLake environment from OpenAI Gym"""


import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the FrozenLakeEnv environment

    Args:
        desc (list of lists): custom map description
        map_name (str): name of the pre-made map
        is_slippery (bool): determines if the ice is slippery

    Returns:
        env: the FrozenLake environment
    """
    if desc is None and map_name is None:
        map_name = "8x8"

    env = gym.make(
        "FrozenLake-v0",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )

    return env
