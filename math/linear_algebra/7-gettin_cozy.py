#!/usr/bin/env python3
"""function for concatinating two 2d matrices along a given axis"""


def cat_matrices2D(mat1, mat2, axis=0):
     """
    Concatenates two 2D matrices along a specified axis.

    Parameters:
        mat1 (list of list of int/float): The first matrix.
        mat2 (list of list of int/float): The second matrix.
        axis (int): The axis along which to concatenate (0 for rows, 1 for columns).

    Returns:
        list of list of int/float: A new matrix resulting from concatenation.
        If the matrices cannot be concatenated along the specified axis, returns None.

    Assumptions:
        - mat1 and mat2 are 2D matrices (lists of lists) with uniform row lengths.
        - All elements are integers or floats.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]
    else:
        return None
