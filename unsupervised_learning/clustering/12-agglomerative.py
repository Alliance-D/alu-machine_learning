#!/usr/bin/env python3
"""Module for agglomerative clustering with Ward linkage."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Perform agglomerative clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters

    Returns:
        clss: numpy.ndarray of shape (n,) with cluster indices per data point
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
    return clss
