#!/usr/bin/env python3
"""Module that performs a convolution on RGB images (multi-channel)."""


import numpy as np


def convolve_channels(images, kernel, padding='valid', stride=(1, 1)):
    """
    Performs a convolution on RGB images.

    Parameters:
    - images (numpy.ndarray): shape (m, h, w, c), multiple RGB images
    - kernel (numpy.ndarray): shape (kh, kw, c), kernel for each channel
    - padding (str or tuple): 'same', 'valid', or (ph, pw)
    - stride (tuple): (sh, sw)

    Returns:
    - numpy.ndarray: shape (m, new_h, new_w), convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel and image channel dimensions must match.")

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = images_padded[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
            ]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
