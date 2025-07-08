#!/usr/bin/env python3
"""
This module provides a function to calculate the correlation matrix
from a given covariance matrix using NumPy, without relying on numpy.cov.
"""

import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix.

    Parameters:
    - C: numpy.ndarray of shape (d, d), the covariance matrix

    Returns:
    - numpy.ndarray of shape (d, d), the correlation matrix

    Raises:
    - TypeError: If C is not a numpy.ndarray
    - ValueError: If C is not a 2D square matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    stddev = np.sqrt(np.diag(C))
    stddev[stddev == 0] = 1e-8  # prevent division by zero
    denom = np.outer(stddev, stddev)
    corr = C / denom

    return corr
