#!/usr/bin/env python3
"""function that performs element-wise addition, subtraction, multiplication,
and division"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division on two numpy.ndarrays.

    Args:
        mat1 (numpy.ndarray or scalar): The first matrix or scalar.
        mat2 (numpy.ndarray or scalar): The second matrix or scalar.

    Returns:
        tuple: A tuple containing four numpy.ndarrays:
               (element-wise sum, difference, product, quotient)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
