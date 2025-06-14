#!/usr/bin/env python3
"""Module for computing sum of squares from 1 to n without loops"""

def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n using a closed-form formula:
    ∑ i^2 = n(n+1)(2n+1)/6

    Args:
        n (int): upper limit of summation

    Returns:
        int or None: sum of squares or None if invalid input
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
