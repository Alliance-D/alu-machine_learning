#!/usr/bin/env python3
"""Module for the Baum-Welch algorithm for a hidden Markov model."""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Perform the Baum-Welch algorithm for a hidden Markov model.

    Args:
        Observations: numpy.ndarray of shape (T,) with observation indices
        Transition: numpy.ndarray of shape (M, M) with transition probabilities
        Emission: numpy.ndarray of shape (M, N) with emission probabilities
        Initial: numpy.ndarray of shape (M, 1) with initial state probabilities
        iterations: number of EM iterations to perform

    Returns:
        Transition: converged transition probability matrix
        Emission: converged emission probability matrix
        or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    T = Observations.shape[0]
    M = Transition.shape[0]
    N = Emission.shape[1]

    if Transition.shape != (M, M):
        return None, None
    if Emission.shape[0] != M:
        return None, None
    if Initial.shape != (M, 1):
        return None, None

    for _ in range(iterations):
        # Forward pass
        F = np.zeros((M, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            F[:, t] = (F[:, t - 1] @ Transition) * Emission[
                :, Observations[t]]

        # Backward pass
        B = np.zeros((M, T))
        B[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            B[:, t] = np.sum(
                Transition * Emission[:, Observations[t + 1]] * B[:, t + 1],
                axis=1
            )

        # Xi: (M, M, T-1) - joint probability of states at t and t+1
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denom = np.sum(F[:, t, np.newaxis] * Transition *
                           Emission[:, Observations[t + 1]] * B[:, t + 1])
            xi[:, :, t] = (F[:, t, np.newaxis] * Transition *
                           Emission[:, Observations[t + 1]] * B[:, t + 1]
                           ) / denom

        # Gamma: (M, T) - probability of being in state i at time t
        gamma = np.sum(xi, axis=1)
        gamma = np.concatenate(
            (gamma, np.sum(xi[:, :, T - 2], axis=0, keepdims=True).T),
            axis=1
        )

        # M-step: update Transition
        Transition = np.sum(xi, axis=2) / np.sum(
            gamma[:, :T - 1], axis=1, keepdims=True)

        # M-step: update Emission
        denom = np.sum(gamma, axis=1, keepdims=True)
        for k in range(N):
            Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
        Emission = Emission / denom

    return Transition, Emission
