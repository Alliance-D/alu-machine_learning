#!/usr/bin/env python3
"""Module that implements the epsilon-greedy action selection"""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Determines the next action using epsilon-greedy.

    Args:
        Q (numpy.ndarray): Q-table
        state (int): current state
        epsilon (float): probability to explore

    Returns:
        int: index of the next action
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])
