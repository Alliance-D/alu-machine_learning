#!/usr/bin/env python3
def minor(matrix):
    """
    Calculates the minor matrix of a given square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Returns:
        A new matrix (list of lists) where each element at [i][j] is the determinant
        of the submatrix formed by removing row i and column j from the original.

    Raises:
        TypeError: If `matrix` is not a list of lists.
        ValueError: If `matrix` is empty or not square.
    """
    # Type check: matrix must be a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # "Non-empty square matrix" check -> if matrix is empty, error
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # Each row must be a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Check that each row has the same length and equals n
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix -> its minor is defined as [[1]]
    if n == 1:
        return [[1]]

    # Helper to compute determinant of a square matrix (recursive)
    def _det(mat):
        m = len(mat)
        # 0x0 case should never happen here because we never pass mat=[[]].
        if m == 1:
            return mat[0][0]
        if m == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        total = 0
        for c in range(m):
            # Build submatrix by removing row 0 and column c
            sub = []
            for i in range(1, m):
                row_copy = [mat[i][j] for j in range(m) if j != c]
                sub.append(row_copy)
            # Cofactor sign = (-1)^(0+c) = (-1)^c
            sign = -1 if (c % 2 == 1) else 1
            total += sign * mat[0][c] * _det(sub)
        return total

    # 7. Build the minor matrix
    minor_matrix = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            # Build the (n-1)x(n-1) submatrix by excluding row i and col j
            submatrix = []
            for r in range(n):
                if r == i:
                    continue
                # keep all columns except j
                filtered_row = [matrix[r][c] for c in range(n) if c != j]
                submatrix.append(filtered_row)

            # Compute determinant of that submatrix
            row_minors.append(_det(submatrix))
        minor_matrix.append(row_minors)

    return minor_matrix
