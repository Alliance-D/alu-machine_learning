#!/usr/bin/env python3
"""Module for K-means clustering using sklearn."""
import sklearn.cluster


def kmeans(X, k):
    """Perform K-means on a dataset using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters

    Returns:
        C: numpy.ndarray of shape (k, d) with centroid means per cluster
        clss: numpy.ndarray of shape (n,) with cluster index per data point
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
