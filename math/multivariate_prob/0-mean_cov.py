#!/usr/bin/env python3
import numpy as np

def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters:
    - X: numpy.ndarray of shape (n, d), where:
        - n is the number of data points
        - d is the number of dimensions

    Returns:
    - mean: numpy.ndarray of shape (1, d)
    - cov: numpy.ndarray of shape (d, d)
    """
    # Validate input
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    
    # Calculate mean
    mean = np.mean(X, axis=0, keepdims=True)
    
    # Center the data
    X_centered = X - mean

    # Calculate covariance matrix
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
