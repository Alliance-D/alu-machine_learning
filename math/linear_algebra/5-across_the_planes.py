#!/usr/bin/env python3
"""function to add two 2d matrices elementwise"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Parameters:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.

    Returns:
        list of list of int/float: A new matrix representing the element-wise
sum of mat1 and mat2.
        If mat1 and mat2 do not have the same shape, returns None.

    Assumptions:
        - mat1 and mat2 are 2D matrices with consistent row and column lengths.
        - All elements are either integers or floats.
    """
    if len(mat1) != len(mat2) or any(len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)):
        return None
    return [[c1 + c2 for c1, c2 in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
