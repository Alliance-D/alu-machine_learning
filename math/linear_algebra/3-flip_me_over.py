#!/usr/bin/env python3

#function to transpose a 2d matrix
def matrix_transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
