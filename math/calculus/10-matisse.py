#!/usr/bin/env python3
"""Module for computing the derivative of a polynomial"""


def poly_derivative(poly):
    """
    Computes the derivative of a polynomial represented by a list of
coefficients.

    Args:
        poly (list): list of coefficients. Index = power of x.

    Returns:
        list: list of coefficients of the derivative,
              or [0] if derivative is zero,
              or None if input is invalid.
    """
    if (not isinstance(poly, list) or
            not all(isinstance(c, (int, float)) for c in poly)):
        return None

    if len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    # Compute derivative: skip constant term and multiply by index
    derivative = [coef * idx for idx, coef in enumerate(poly)][1:]

    return derivative if any(derivative) else [0]
