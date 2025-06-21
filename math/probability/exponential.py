#!/usr/bin/env python3
"""
Exponential distribution class
"""


class Exponential:
    """
    Represents an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor

        Parameters:
        - data: list of data to estimate the distribution (optional)
        - lambtha: expected number of occurrences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        if x < 0:
            return 0
        return self.lambtha * (2.7182818285 ** (-self.lambtha * x))

    def cdf(self, x):
        if x < 0:
            return 0
        return 1 - (2.7182818285 ** (-self.lambtha * x))
