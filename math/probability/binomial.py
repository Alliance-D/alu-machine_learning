#!/usr/bin/env python3
"""Binomial distribution class without any imports."""


class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the distribution with data or provided n and p.

        Args:
            data (list): Dataset used to estimate parameters.
            n (int): Number of trials (if no data).
            p (float): Probability of success (if no data).

        Raises:
            TypeError: If data is not a list.
            ValueError: If invalid data, n, or p values.
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
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1 - (variance / mean)
            n = round(mean / p)
            self.n = n
            self.p = sum(data) / (len(data) * self.n)

    def factorial(self, num):
        """
        Compute factorial of a non-negative integer.

        Args:
            num (int): Non-negative integer.

        Returns:
            int: Factorial of num.
        """
        if num == 0 or num == 1:
            return 1
        result = 1
        for i in range(2, num + 1):
            result *= i
        return result

    def pmf(self, k):
        """
        Calculate the Probability Mass Function (PMF) for k successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: PMF value.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        n_fact = self.factorial(self.n)
        k_fact = self.factorial(k)
        nk_fact = self.factorial(self.n - k)
        comb = n_fact / (k_fact * nk_fact)
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculate the Cumulative Distribution Function (CDF) for k.

        Args:
            k (int): Number of successes.

        Returns:
            float: CDF value.
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
