#!/usr/bin/env python3
"""Train a DQN agent to play Breakout using Keras-RL"""

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


def build_model(input_shape, nb_actions):
    """Build the CNN model"""
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4),
                           activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2),
                           activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1),
                           activation='relu'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))

    return model


def main():
    """Train the agent"""
    env = gym.make('BreakoutDeterministic-v4')
    np.random.seed(0)
    env.seed(0)

    nb_actions = env.action_space.n

    model = build_model((4,) + env.observation_space.shape,
                        nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=50000,
                   target_model_update=10000,
                   policy=policy,
                   enable_dueling_network=True)

    dqn.compile(Adam(lr=0.00025), metrics=['mae'])

    dqn.fit(env,
            nb_steps=1750000,
            visualize=False,
            verbose=2)

    dqn.save_weights('policy.h5', overwrite=True)

    env.close()


if __name__ == '__main__':
    main()
