#!/usr/bin/env python3
import numpy as np
""" a function def definiteness(matrix): that calculates the definiteness
of a matrix"""


def definiteness(matrix):
    """
    Calculates the definiteness of a given square matrix.

    Args:
        matrix (numpy.ndarray): A 2D square NumPy array.

    Returns:
        A string-one of:
            - "Positive definite"
            - "Positive semi-definite"
            - "Negative semi-definite"
            - "Negative definite"
            - "Indefinite"
        or None if the input is not a valid square matrix for this test
        (e.g., empty, non-square, or has wild complex eigenvalues).

    Raises:
        TypeError: If `matrix` is not a numpy.ndarray.
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

    # 3. Compute eigenvalues (may be complex)
    try:
        ev = np.linalg.eigvals(matrix)
    except Exception:
        # If NumPy cannot compute eigenvalues, treat as invalid
        return None

    # 4. If any eigenvalue has a significant imaginary part, reject
    tol = 1e-8
    if np.any(np.abs(ev.imag) > tol):
        return None

    # 5. Work only with the real parts
    real_ev = ev.real

    # 6. Classify by checking signs (with tolerance)

    # Positive definite
    if np.all(real_ev > tol):
        return "Positive definite"

    # Positive semi-definite
    if np.all(real_ev >= -tol) and np.any(np.abs(real_ev) <= tol):
        return "Positive semi-definite"

    # Negative definite
    if np.all(real_ev < -tol):
        return "Negative definite"

    # Negative semi-definite
    if np.all(real_ev <= tol) and np.any(np.abs(real_ev) <= tol):
        return "Negative semi-definite"

    # Indefinite
    if np.any(real_ev > tol) and np.any(real_ev < -tol):
        return "Indefinite"

    # If it doesn't fit any category, return None
    return None
