#!/usr/bin/env python3
"""Module for computing Markov chain state probabilities."""
import numpy as np


def markov_chain(P, s, t=1):
    """Determine the probability of a Markov chain being in a state
	after t iterations.

    Args:
        P: numpy.ndarray of shape (n, n), the transition matrix
        s: numpy.ndarray of shape (1, n), starting state probabilities
        t: number of iterations

    Returns:
        numpy.ndarray of shape (1, n) with state probabilities after t
        iterations, or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t < 1:
        return None
    return np.matmul(s, np.linalg.matrix_power(P, t))
