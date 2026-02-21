#!/usr/bin/env python3
"""Module for the maximization step of the EM algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        g: numpy.ndarray of shape (k, n) containing posterior probabilities

    Returns:
        pi: numpy.ndarray of shape (k,) with updated priors
        m: numpy.ndarray of shape (k, d) with updated centroid means
        S: numpy.ndarray of shape (k, d, d) with updated covariance matrices
        or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    nk = np.sum(g, axis=1)
    pi = nk / n
    m = (g @ X) / nk[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = (g[i] * diff.T) @ diff / nk[i]

    return pi, m, S

