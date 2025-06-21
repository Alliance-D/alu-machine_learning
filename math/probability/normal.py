#!/usr/bin/env python3
"""Normal distribution module without imports."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the Normal distribution.

        Args:
            data (list): List of data to estimate the distribution.
            mean (float): Mean of the distribution.
            stddev (float): Standard deviation of the distribution.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has fewer than two values or stddev ≤ 0.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (
                sum([(x - self.mean) ** 2 for x in data]) / len(data)
            ) ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The z-score.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Args:
            z (float): The z-score.

        Returns:
            float: The corresponding x-value.
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the PDF value for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The value of the PDF.
        """
        import math
        pi = math.pi
        e = math.e
        exp_part = e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        return (1 / (self.stddev * ((2 * pi) ** 0.5))) * exp_part

    def cdf(self, x):
        """
        Calculates the CDF value for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The value of the CDF.
        """
        return 0.5 * (1 + self.erf(
            (x - self.mean) / (self.stddev * (2 ** 0.5))
        ))

    def erf(self, x):
        """
        Computes the error function approximation.

        Args:
            x (float): The input value.

        Returns:
            float: Approximation of the error function at x.
        """
        # Approximation using Taylor Series expansion
        return (2 / (3.1415926536 ** 0.5)) * (
            x - (x ** 3) / 3 + (x ** 5) / 10
            - (x ** 7) / 42 + (x ** 9) / 216
        )
