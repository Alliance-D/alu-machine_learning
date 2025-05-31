#!/usr/bin/env python3
def adjugate(matrix):
    """
    Calculates the adjugate (classical adjoint) of a given square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Returns:
        The adjugate matrix (list of lists), which is the transpose of the cofactor matrix.

    Raises:
        TypeError: If `matrix` is not a list of lists.
        ValueError: If `matrix` is empty or not square.
    """
    # 1. Type check: matrix must be a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # 2. Non-empty check
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # 3. Each row must be a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Check square: every row length == n
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    # Helper to compute determinant of any square matrix
    def _det(mat):
        m = len(mat)
        if m == 1:
            return mat[0][0]
        if m == 2:
            return mat[0][0]

