#!/usr/bin/env python3
"""funtion that concatenates two matrices along a specific axis"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy arrays along the specified axis.

    Args:
        mat1 (np.ndarray): First matrix.
        mat2 (np.ndarray): Second matrix.
        axis (int): Axis along which to concatenate.

    Returns:
        np.ndarray: The concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
