#!/usr/bin/env python3
import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    assert c == kc, "The number of channels in the image and kernel must match"

    if padding == 'same':
        ph = max((h - 1) * sh + kh - h, 0)
        pw = max((w - 1) * sw + kw - w, 0)
        ph_top = ph // 2
        ph_bottom = ph - ph_top
        pw_left = pw // 2
        pw_right = pw - pw_left
    elif padding == 'valid':
        ph_top = ph_bottom = pw_left = pw_right = 0
    else:  # tuple
        ph_top, pw_left = padding
        ph_bottom, pw_right = padding

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph_top, ph_bottom), (pw_left, pw_right), (0, 0)),
        mode='constant',
        constant_values=0
    )

    new_h = (h + ph_top + ph_bottom - kh) // sh + 1
    new_w = (w + pw_left + pw_right - kw) // sw + 1
    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            region = padded_images[:, h_start:h_end, w_start:w_end, :]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
