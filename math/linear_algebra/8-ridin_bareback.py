#!/usr/bin/env python3
"""function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication between two 2D matrices.

    Parameters:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.

    Returns:
        list of list of int/float: A new matrix resulting from multiplying
mat1 by mat2.
        If the matrices cannot be multiplied (incompatible dimensions),
returns None.

    Assumptions:
        - mat1 and mat2 are 2D matrices (lists of lists) with uniform row
lengths.
        - All elements are integers or floats.
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [
        [
            sum(a * b for a, b in zip(row, col))
            for col in zip(*mat2)
        ] for row in mat1]
