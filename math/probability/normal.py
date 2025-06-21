#!/usr/bin/env python3
"""Normal distribution module without any imports."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution.

        Args:
            data (list): List of data to estimate parameters.
            mean (float): Mean of the distribution.
            stddev (float): Standard deviation of the distribution.

        Raises:
            TypeError: If data is not a list.
            ValueError: If stddev <= 0 or data has < 2 values.
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
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Args:
            x (float): x-value to evaluate.

        Returns:
            float: z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Args:
            z (float): z-score to convert.

        Returns:
            float: x-value corresponding to z.
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF (Probability Density Function)
        for a given x-value.

        Args:
            x (float): x-value to evaluate.

        Returns:
            float: The PDF value for x.
        """
        e = 2.7182818285
        pi = 3.1415926536
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        denom = self.stddev * (2 * pi) ** 0.5
        return (e ** exponent) / denom

    def cdf(self, x):
        """
        Calculates the value of the CDF (Cumulative Distribution Function)
        for a given x-value using a Taylor series approximation.

        Args:
            x (float): x-value to evaluate.

        Returns:
            float: The CDF value for x.
        """
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = (2 / (pi ** 0.5)) * (
            z - (z ** 3) / 3 + (z ** 5) / 10 -
            (z ** 7) / 42 + (z ** 9) / 216
        )
        return 0.5 * (1 + erf)
