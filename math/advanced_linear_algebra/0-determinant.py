#!/usr/bin/env python3
"""a function def determinant(matrix): that calculates the determinant of a
matrix"""
def determinant(matrix):
    """
    Calculates the determinant of a square matrix represented as a
list of lists.

    Args:
        matrix (list of lists): The input matrix. The special case [[]] is
treated as a 0x0 matrix.

    Returns:
        The determinant of the matrix (an integer or float).

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix.
    """
    # Check that 'matrix' is a non-empty list
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    #  Handle the 0x0-matrix case: matrix == [[]] -> det = 1
    if matrix == [[]]:
        return 1

    #  Verify that each row is itself a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    #  Verify that the matrix is square (each row has length n)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    #  Base cases for small n
    if n == 1:
        # 1x1 matrix: determinant is the single entry
        return matrix[0][0]
    if n == 2:
        # 2x2 matrix: ad - bc
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive expansion by the first row (Laplace expansion)
    det = 0
    for col_index in range(n):
        submatrix = []
        for i in range(1, n):
            # For row i, copy all columns except col_index
            row_copy = [matrix[i][j] for j in range(n) if j != col_index]
            submatrix.append(row_copy)

        # Cofactor sign = (-1)^(0 + col_index) = (-1)^col_index
        cofactor_sign = (-1) ** col_index
        cofactor_value = matrix[0][col_index]

        # Recursive call for the minor
        minor_det = determinant(submatrix)

        det += cofactor_sign * cofactor_value * minor_det

    return det
