#!/usr/bin/env python3
"""
a function def definiteness(matrix): that calculates the definiteness of a matrix
"""

import numpy as np

def definiteness(matrix):
    """
    Determine the definiteness of a square, symmetric matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        A 2D, non-empty, square NumPy array.

    Returns
    -------
    str or None
        One of:
          - "Positive definite"
          - "Positive semi-definite"
          - "Negative semi-definite"
          - "Negative definite"
          - "Indefinite"
        or None if `matrix` is not valid (non-symmetric, empty, non-square,
        or has significant imaginary eigenvalues).

    Raises
    ------
    TypeError
        If `matrix` is not a numpy.ndarray.
    """
    # 1. Type check
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # 2. Must be 2D and square and non-empty
    if matrix.ndim != 2:
        return None
    n, m = matrix.shape
    if n == 0 or n != m:
        return None

    if not np.allclose(matrix, matrix.T, atol=1e-8):
        
        return None

    # 4. Compute eigenvalues
    try:
        ev = np.linalg.eigvals(matrix)
    except Exception:
        return None

    # 5. Reject if any eigenvalue is noticeably complex
    tol = 1e-8
    if np.any(np.abs(ev.imag) > tol):
        return None

    real_ev = ev.real

    # 6. Classify based on the signs of the real parts
    if np.all(real_ev > tol):
        return "Positive definite"
    if np.all(real_ev >= -tol) and np.any(np.abs(real_ev) <= tol):
        return "Positive semi-definite"
    if np.all(real_ev < -tol):
        return "Negative definite"
    if np.all(real_ev <= tol) and np.any(np.abs(real_ev) <= tol):
        return "Negative semi-definite"
    if np.any(real_ev > tol) and np.any(real_ev < -tol):
        return "Indefinite"

    # If it doesn't fit any category, return None
    return None
