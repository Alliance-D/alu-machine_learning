#!/usr/bin/env python3
"""Module for steady state probabilities of a regular Markov chain."""
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

    # Check regularity: some power of P must have all positive entries
    Pk = P.copy()
    regular = False
    for _ in range(n * n):
        if (Pk > 0).all():
            regular = True
            break
        Pk = Pk @ P
    if not regular:
        return None

    # used sum(pi) = 1 using linear algebra
    # pi @ (P - I) = 0, add normalization constraint
    A = (P - np.eye(n)).T
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return pi.reshape(1, n)
