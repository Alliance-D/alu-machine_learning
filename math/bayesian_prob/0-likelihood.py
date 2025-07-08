#!/usr/bin/env python3
"""Likelihood module for calculating the likelihood of seeing x outcomes
from n trials."""

import numpy as np
from scipy.special import comb


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data x from n trials
    for each probability in P.

    Args:
        x (int): Number of patients with severe side effects.
        n (int): Total number of patients.
        P (np.ndarray): 1D array of hypothetical probabilities.

    Returns:
        np.ndarray: Likelihood values for each probability in P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal
to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    likelihoods = comb(n, x) * (P ** x) * ((1 - P) ** (n - x))
    return np.atleast_1d(likelihoods)
