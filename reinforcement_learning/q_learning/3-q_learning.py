#!/usr/bin/env python3
"""Module that implements Q-learning for FrozenLakeEnv"""


import numpy as np
from 2-epsilon_greedy import epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs Q-learning on the given FrozenLake environment.

    Args:
        env: FrozenLakeEnv instance
        Q (numpy.ndarray): Q-table
        episodes (int): number of episodes to train
        max_steps (int): max steps per episode
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon (float): initial epsilon for epsilon-greedy
        min_epsilon (float): minimum epsilon
        epsilon_decay (float): epsilon decay rate per episode

    Returns:
        tuple: (Q, total_rewards)
            Q (numpy.ndarray): updated Q-table
            total_rewards (list): rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _ = env.step(action)

            # Adjust reward for holes
            if reward == 0 and env.desc.flatten()[new_state] == b'H':
                reward = -1

            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            state = new_state
            total_reward += reward

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
        total_rewards.append(total_reward)

    return Q, total_rewards
