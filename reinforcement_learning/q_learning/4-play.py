#!/usr/bin/env python3
"""Module that allows the trained agent to play FrozenLake using the Q-table"""


import numpy as np


def play(env, Q, max_steps=100):
    """Plays one episode using a trained Q-table.

    Args:
        env: FrozenLakeEnv instance
        Q (numpy.ndarray): trained Q-table
        max_steps (int): maximum steps for the episode

    Returns:
        float: total reward accumulated in the episode
    """
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Choose the best action (exploit)
        action = np.argmax(Q[state])

        # Take the action
        new_state, reward, done, _ = env.step(action)

        # Render the board in console
        desc_copy = np.array(env.desc, copy=True)
        row, col = divmod(state, env.desc.shape[1])
        desc_copy[row, col] = b'`' + desc_copy[row, col] + b'`'
        print(desc_copy.astype(str))
        print(f"  ({['Left','Down','Right','Up'][action]})")

        total_reward += reward
        state = new_state

        if done:
            # Mark final position
            row, col = divmod(state, env.desc.shape[1])
            desc_copy = np.array(env.desc, copy=True)
            desc_copy[row, col] = b'`' + desc_copy[row, col] + b'`'
            print(desc_copy.astype(str))
            break

    return total_reward
