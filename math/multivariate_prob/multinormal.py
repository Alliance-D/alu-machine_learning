#!/usr/bin/env python3
import numpy as np

class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes the MultiNormal instance.

        Parameters:
        - data: numpy.ndarray of shape (d, n)
            d: number of dimensions
            n: number of data points
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = (data_centered @ data_centered.T) / (n - 1)
        self.d = d

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Parameters:
        - x: numpy.ndarray of shape (d, 1)

        Returns:
        - The value of the PDF at x
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        # Difference from mean
        x_m = x - self.mean

        # Determinant and inverse of covariance matrix
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)

        # PDF computation
        num = np.exp(-0.5 * (x_m.T @ inv_cov @ x_m))
        denom = np.sqrt(((2 * np.pi) ** self.d) * det_cov)

        return float(num / denom)

