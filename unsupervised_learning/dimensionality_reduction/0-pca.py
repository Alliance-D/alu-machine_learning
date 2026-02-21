#!/usr/bin/env python3
"""Module for performing PCA on a dataset."""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset maintaining a fraction of variance.

    Args:
        X: numpy.ndarray of shape (n, d) with zero-mean data
        var: fraction of variance to maintain

    Returns:
        W: numpy.ndarray of shape (d, nd) weights matrix
    """
    _, s, Vt = np.linalg.svd(X)
    cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)
    nd = np.argmax(cumvar >= var) + 1
    return Vt[:nd].T
