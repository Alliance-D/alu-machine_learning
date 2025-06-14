#!/usr/bin/env python3
"""Module to compute the integral of a polynomial"""

def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): List of coefficients where index = power of x
        C (int): Integration constant (default = 0)

    Returns:
        list: List of coefficients of the integrated polynomial
              or None if input is invalid
    """
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]
    for i, coef in enumerate(poly):
        val = coef / (i + 1)
        integral.append(int(val) if val.is_integer() else val)

    # Remove unnecessary trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
