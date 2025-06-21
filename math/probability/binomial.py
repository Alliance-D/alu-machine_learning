#!/usr/bin/env python3
"""Binomial distribution module without using imports."""


class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution.

        Args:
            data (list): List of data to estimate parameters.
            n (int): Number of Bernoulli trials.
            p (float): Probability of success.

        Raises:
            TypeError: If data is not a list.
            ValueError: If n ≤ 0, p not in (0, 1),
                        or data has < 2 values.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1 - (var / mean)
            n = round(mean / p)
            self.n = n
            self.p = sum(data) / (len(data) * n)

    def factorial(self, k):
        """Calculates the factorial of k."""
        if k == 0 or k == 1:
            return 1
        result = 1
        for i in range(2, k + 1):
            result *= i
        return result

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: PMF value for k.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        comb = (self.factorial(self.n) //
                (self.factorial(k) * self.factorial(self.n - k)))
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: CDF value for k.
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
