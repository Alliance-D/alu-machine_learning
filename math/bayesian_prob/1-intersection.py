#!/usr/bin/env python3
import numpy as np
from math import comb

def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining the data with the various hypothetical probabilities.

    Parameters:
    - x: int, number of patients that develop severe side effects
    - n: int, total number of patients observed
    - P: 1D numpy.ndarray of hypothetical probabilities
    - Pr: 1D numpy.ndarray of prior beliefs for probabilities in P

    Returns:
    - 1D numpy.ndarray containing the intersection for each probability in P
    """
    # Validations (in the specified order)
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate likelihood for each probability in P (binomial PMF)
    comb_val = comb(n, x)
    likelihoods = comb_val * (P ** x) * ((1 - P) ** (n - x))

    # Intersection = likelihood * prior
    intersection = likelihoods * Pr

    return intersection

