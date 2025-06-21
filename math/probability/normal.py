#!/usr/bin/env python3
"""
Normal distribution class with CDF computation
"""


class Normal:
    """
    Represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize a Normal distribution.
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
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def erf(self, x):
        # Approximate erf using Maclaurin series (4 terms)
        pi = 3.1415926536
        return (2 / (pi ** 0.5)) * (x - (x ** 3) / 3 + (x ** 5) / 10 - (x ** 7) / 42)

    def pdf(self, x):
        pi = 3.1415926536
        e = 2.7182818285
        num = e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        den = self.stddev * ((2 * pi) ** 0.5)
        return num / den

    def cdf(self, x):
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + self.erf(z))
