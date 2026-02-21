#!/usr/bin/env python3
"""Module for the Expectation-Maximization algorithm for a GMM."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Perform the expectation maximization algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters
        iterations: positive integer, maximum number of iterations
        tol: non-negative float, log likelihood tolerance for early stopping
        verbose: boolean, whether to print log likelihood info

    Returns:
        pi: numpy.ndarray of shape (k,) with priors per cluster
        m: numpy.ndarray of shape (k, d) with centroid means per cluster
        S: numpy.ndarray of shape (k, d, d) with covariance matrices
        g: numpy.ndarray of shape (k, n) with posterior probabilities
        log_l: log likelihood of the model
        or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    log_l_prev = 0

    for i in range(iterations):
        g, log_l = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(log_l, 5)))
        if i > 0 and abs(log_l - log_l_prev) <= tol:
            if verbose and i % 10 != 0:
                print("Log Likelihood after {} iterations: {}".format(
                    i, round(log_l, 5)))
            return pi, m, S, g, log_l
        log_l_prev = log_l
        pi, m, S = maximization(X, g)

    g, log_l = expectation(X, pi, m, S)
    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            iterations, round(log_l, 5)))

    return pi, m, S, g, log_l
