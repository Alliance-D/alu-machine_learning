#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad images with 0s on height and width
    padded = np.pad(images,
                    pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant')

    # Initialize output
    output = np.zeros((m, h, w))

    # Perform convolution with two for-loops
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return output
