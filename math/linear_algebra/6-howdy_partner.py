#!/usr/bin/env python3
"""function to concatinate two arrays"""


def cat_arrays(arr1, arr2):
    """
    Concatenates two arrays (lists of ints or floats).

    Parameters:
        arr1 (list of int/float): The first array.
        arr2 (list of int/float): The second array.

    Returns:
        list of int/float: A new list containing all elements of arr1 followed by all elements of arr2.

    Assumptions:
        - arr1 and arr2 are lists containing only integers or floats.
        - The original lists are not modified.
    """
    return arr1 + arr2
