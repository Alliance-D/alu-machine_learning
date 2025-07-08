#!/usr/bin/env python3
"""
0-likelihood.py
Calculates the likelihood of obtaining data for various hypothetical
probabilities of developing severe side effects using the binomial distribution.
"""

import numpy as np


def comb(n, k):
    """Calculate the binomial coefficient "n choose k" without importing math.comb."""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    c = 1
    for i in range(min(k, n - k)):
        c = c * (n - i) // (i + 1)
    return c


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining this data for various hypothetical
    probabilities.

    Args:
        x (int): Number of patients with severe side effects.
        n (int): Total number of patients.
        P (numpy.ndarray): 1D array of hypothetical probabilities.

    Returns:
        numpy.ndarray: Likelihoods for each probability in P.

    Raises:
        ValueError: If n is not a positive integer.
        ValueError: If x is not an integer >= 0.
        ValueError: If x > n.
        TypeError: If P is not a 1D numpy.ndarray.
        ValueError: If any values in P are not in [0, 1].
    """
    # Validations
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate likelihood for each p in P using binomial pmf formula
    c = comb(n, x)
    likelihoods = c * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
