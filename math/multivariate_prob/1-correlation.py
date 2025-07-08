#!/usr/bin/env python3
import numpy as np

def correlation(C):
    """
    Calculates a correlation matrix.

    Parameters:
    - C: numpy.ndarray of shape (d, d), the covariance matrix

    Returns:
    - numpy.ndarray of shape (d, d), the correlation matrix
    """
    # Validate input
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Standard deviations (sqrt of diagonal)
    stddev = np.sqrt(np.diag(C))  # shape (d,)

    # Avoid division by zero by replacing 0s with very small number (optional safeguard)
    stddev[stddev == 0] = 1e-8

    # Outer product of stddev to form the denominator matrix
    denom = np.outer(stddev, stddev)  # shape (d, d)

    # Compute correlation matrix
    corr = C / denom

    return corr
