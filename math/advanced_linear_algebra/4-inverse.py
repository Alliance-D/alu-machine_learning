#!/usr/bin/env python3
"""a function def inverse(matrix): that calculates the inverse of a matrix"""


def inverse(matrix):
    """
    Calculates the inverse of a given square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix to invert.

    Returns:
        A new matrix (list of lists) representing the inverse of `matrix`,
        or None if `matrix` is singular (determinant == 0).

    Raises:
        TypeError: If `matrix` is not a list of lists.
        ValueError: If `matrix` is empty or not square.
    """
    # 1. Ensure `matrix` is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # 2. Check for emptiness
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # 3. Each row must be a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # 4. Check that it is square: each row length == n
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # 5. Compute the determinant of an nxn matrix via recursive expansion
    def _det(mat):
        m = len(mat)
        # Base cases
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

    if n == 1:
        a = matrix[0][0]
        if a == 0:
            return None
        return [[1.0 / a]]

    det_full = _det(matrix)
    if det_full == 0:
        return None  # singular matrix

    cofactor_matrix = []
    for i in range(n):
        row_cof = []
        for j in range(n):
            # Build the submatrix by excluding row i and column j
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

    # Transpose the cofactor matrix to get the adjugate
    #    adjugate[i][j] = cofactor_matrix[j][i]
    adjugate_matrix = []
    for i in range(n):
        transposed_row = [cofactor_matrix[j][i] for j in range(n)]
        adjugate_matrix.append(transposed_row)

    # 10. Divide each entry of the adjugate by det_full to get the inverse
    inverse_matrix = []
    for i in range(n):
        inv_row = []
        for j in range(n):
            inv_row.append(adjugate_matrix[i][j] / det_full)
        inverse_matrix.append(inv_row)

    return inverse_matrix
