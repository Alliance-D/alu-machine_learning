#!/usr/bin/env python3

#function to add two 2d matrices elementwise
def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2) or any(len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)):
        return None
    return [[c1 + c2 for c1, c2 in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
