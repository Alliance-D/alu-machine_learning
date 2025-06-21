#!/usr/bin/env python3
"""
Poisson distribution class
"""


class Poisson:
    """
    Represents a Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor

        Parameters:
        - data: list of data to estimate the distribution (optional)
        - lambtha: expected number of occurrences (float)
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
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, k):
        result = 1
        for i in range(1, k + 1):
            result *= i
        return result

    def pmf(self, k):
        k = int(k)
        if k < 0:
            return 0
        return (self.lambtha ** k) * (2.7182818285 ** -self.lambtha) / self.factorial(k)

    def cdf(self, k):
        k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(i) for i in range(k + 1))
