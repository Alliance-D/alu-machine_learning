#!/usr/bin/env python3
"""a function def adjugate(matrix): that calculates the adjugate matrix of
a matrix"""


def adjugate(matrix):
    """
    Calculates the adjugate (classical adjoint) of a given square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Returns:
        The adjugate matrix (list of lists), which is the transpose of the
cofactor matrix.

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
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        # Recursive expansion along first row
        total = 0
        for c in range(m):
            sub = []
            for i in range(1, m):
                row_copy = [mat[i][j] for j in range(m) if j != c]
                sub.append(row_copy)

            sign = -1 if (c % 2 == 1) else 1
            total += sign * mat[0][c] * _det(sub)
        return total

    # 7. Build the cofactor matrix first
    cofactor_matrix = []
    for i in range(n):
        row_cof = []
        for j in range(n):
            # Form the submatrix by removing row i and column j
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

    # 8. Transpose the cofactor matrix to get the adjugate
    adjugate_matrix = []
    for col in range(n):
        transposed_row = [cofactor_matrix[row][col] for row in range(n)]
        adjugate_matrix.append(transposed_row)

    return adjugate_matrix
