#!/usr/bin/env python3
"""Module for Gaussian Mixture Model using sklearn."""
import sklearn.mixture


def gmm(X, k):
    """Calculate a GMM from a dataset using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) with cluster priors
        m: numpy.ndarray of shape (k, d) with centroid means
        S: numpy.ndarray of shape (k, d, d) with covariance matrices
        clss: numpy.ndarray of shape (n,) with cluster indices per data point
        bic: BIC value for the model
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)
    return pi, m, S, clss, bic
