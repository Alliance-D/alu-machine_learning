#!/usr/bin/env python3
"""Binomial Distribution Class"""


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
            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n
            self.n = n
            self.p = float(p)

    def factorial(self, k):
        """Computes factorial without import"""
        result = 1
        for i in range(1, k + 1):
            result *= i
        return result

    def pmf(self, k):
        """Probability Mass Function"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        n_fac = self.factorial(self.n)
        k_fac = self.factorial(k)
        n_k_fac = self.factorial(self.n - k)

        comb = n_fac / (k_fac * n_k_fac)
        prob = comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return prob

    def cdf(self, k):
        """Cumulative Distribution Function"""
        k = int(k)
        if k < 0:
            return 0

        total = 0
        for i in range(k + 1):
            total += self.pmf(i)
        return total
