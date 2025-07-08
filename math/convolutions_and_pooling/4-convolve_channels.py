#!/usr/bin/env python3
"""Module that performs a valid convolution on grayscale images."""


import numpy as np


def convolve_grayscale(images, kernel, padding='valid', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with optional padding and
stride.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w)
    - kernel (numpy.ndarray): shape (kh, kw)
    - padding (str or tuple): 'same', 'valid', or (ph, pw)
    - stride (tuple): (sh, sw)

    Returns:
    - numpy.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    # Only apply padding if needed
    if ph > 0 or pw > 0:
        images = np.pad(
            images, ((0, 0), (ph, ph), (pw, pw)),
            mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
