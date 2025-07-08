#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w) containing grayscale images
    - kernel: numpy.ndarray of shape (kh, kw) with the kernel for convolution
    - padding: tuple of (ph, pw), padding for height and width

    Returns:
    - convolved images as a numpy.ndarray
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1

    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            image_slice = padded_images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
