#!/usr/bin/env python3
def cofactor(matrix):
    """
    Calculates the cofactor matrix of a given square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Returns:
        The cofactor matrix (list of lists) of `matrix`, where each entry [i][j]
        is (-1)^(i+j) times the determinant of the submatrix formed by removing
        row i and column j from the original.

    Raises:
        TypeError: If `matrix` is not a list of lists.
        ValueError: If `matrix` is empty or not square.
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    def _det(mat):
        m = len(mat)
        if m == 1:
            return mat[0][0]
        if m == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        total = 0
        for c in range(m):

            sub = []
            for i in range(1, m):
                row_copy = [mat[i][j] for j in range(m) if j != c]
                sub.append(row_copy)
            sign = -1 if (c % 2 == 1) else 1
            total += sign * mat[0][c] * _det(sub)
        return total

    cofactor_matrix = []
    for i in range(n):
        row_cof = []
        for j in range(n):
            submatrix = []
            for r in range(n):
                if r == i:
                    continue
                filtered_row = [matrix[r][c] for c in range(n) if c != j]
                submatrix.append(filtered_row)

            minor_det = _det(submatrix)
            sign = -1 if ((i + j) % 2 == 1) else 1
            row_cof.append(sign * minor_det)
        cofactor_matrix.append(row_cof)

    return cofactor_matrix

