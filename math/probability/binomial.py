#!/usr/bin/env python3
"""Binomial Distribution Class"""

import math


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1 - (variance / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est
            self.n = n_est
            self.p = float(p_est)

    def pmf(self, k):
        """Calculates the PMF for a given number of successes k"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        nCk = math.factorial(self.n) / (math.factorial(k) * math.factorial(self.n - k))
        return nCk * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the CDF for a given number of successes k"""
        k = int(k)
        if k < 0:
            return 0

        return sum(self.pmf(i) for i in range(0, k + 1))
