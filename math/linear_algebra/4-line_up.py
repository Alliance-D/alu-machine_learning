#!/usr/bin/env python3
"""function to add teo arrays elementwise"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Parameters:
    arr1 (list of int/float): The first input list.
    arr2 (list of int/float): The second input list.

    Returns:
    list: A new list containing the element-wise sums of arr1 and arr2.
          If the input lists are not the same length, returns None.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
