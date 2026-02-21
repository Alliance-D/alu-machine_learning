#!/usr/bin/env python3
"""Module for determining if a Markov chain is absorbing."""
import numpy as np


def absorbing(P):
    """Determine if a Markov chain is absorbing.

    Args:
        P: numpy.ndarray of shape (n, n), the transition matrix

    Returns:
        True if the chain is absorbing, False on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if not np.isclose(P.sum(axis=1), 1).all():
        return False

    n = P.shape[0]
    absorbing_states = np.isclose(np.diag(P), 1)
    if not absorbing_states.any():
        return False

    reachable = absorbing_states.copy()
    for _ in range(n):
        new_reachable = reachable.copy()
        for i in range(n):
            if not reachable[i]:
                if np.any(P[i, reachable] > 0):
                    new_reachable[i] = True
        if np.all(new_reachable == reachable):
            break
        reachable = new_reachable

    return bool(reachable.all())
