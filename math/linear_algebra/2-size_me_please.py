#!/usr/bin/env python3
"""function def matrix_shape(matrix): that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix (list): A nested list (matrix) where all elements in the same dimension
                       are of the same type and shape.

    Returns:
        list: A list of integers representing the shape of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
