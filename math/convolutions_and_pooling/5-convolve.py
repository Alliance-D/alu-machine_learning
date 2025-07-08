#!/usr/bin/env python3
import numpy as np

def convolve(images, kernels, padding='same', stride=(1, 1)):
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if c != kc:
        raise ValueError("Number of channels in images and kernels must match")

    # Compute padding
    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1 if sh > 1 else (kh - 1) // 2
        pw = ((w - 1) * sw + kw - w) // 2 + 1 if sw > 1 else (kw - 1) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    # Pad the images
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w_
