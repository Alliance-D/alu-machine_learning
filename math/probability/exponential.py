#!/usr/bin/env python3
"""Exponential distribution module without any imports."""


class Exponential:
    """Represents an exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Exponential distribution.

        Args:
            data (list): List of the data to estimate lambtha.
            lambtha (float): Expected number of occurrences.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has fewer than two values or
                        lambtha is not positive.
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
        """
        Calculates the value of the PDF (Probability Density Function)
        for a given time period x.

        Args:
            x (float): Time period.

        Returns:
            float: The PDF value for x. Returns 0 if x < 0.
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * e ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the value of the CDF (Cumulative Distribution Function)
        for a given time period x.

        Args:
            x (float): Time period.

        Returns:
            float: The CDF value for x. Returns 0 if x < 0.
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return 1 - e ** (-self.lambtha * x)
