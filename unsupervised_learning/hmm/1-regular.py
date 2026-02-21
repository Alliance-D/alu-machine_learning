#!/usr/bin/env python3
"""Module for computing steady state probabilities of a regular Markov chain."""
import numpy as np


def regular(P):
    """Determine the steady state probabilities of a regular Markov chain.

    Args:
        P: numpy.ndarray of shape (n, n), the transition matrix

    Returns:
        numpy.ndarray of shape (1, n) with steady state probabilities,
        or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not np.isclose(P.sum(axis=1), 1).all():
        return None

    n = P.shape[0]
    Pk = P.copy()
    for _ in range(1000):
        Pk = np.matmul(Pk, P)
        if (Pk > 0).all() and np.allclose(Pk, Pk[0]):
            return Pk[0:1]

    return None
