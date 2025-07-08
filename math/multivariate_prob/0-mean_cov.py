#!/usr/bin/env python3
"""
This module provides a function to calculate the mean and covariance
of a given dataset using NumPy, without relying on numpy.cov.
"""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters:
    - X: numpy.ndarray of shape (n, d) where:
        - n: number of data points
        - d: number of dimensions

    Returns:
    - mean: numpy.ndarray of shape (1, d)
    - cov: numpy.ndarray of shape (d, d)

    Raises:
    - TypeError: If X is not a 2D numpy.ndarray
    - ValueError: If X contains fewer than 2 data points
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
