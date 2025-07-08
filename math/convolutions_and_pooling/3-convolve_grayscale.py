def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        ph = max((h - 1) * sh + kh - h, 0)
        pw = max((w - 1) * sw + kw - w, 0)
        ph_top = ph // 2
        ph_bottom = ph - ph_top
        pw_left = pw // 2
        pw_right = pw - pw_left
    elif padding == 'valid':
        ph_top = ph_bottom = pw_left = pw_right = 0
    else:
        ph, pw = padding
        ph_top = ph_bottom = ph
        pw_left = pw_right = pw

    # Pad the images
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
        mode='constant'
    )

    # Output dimensions
    out_h = (padded_images.shape[1] - kh) // sh + 1
    out_w = (padded_images.shape[2] - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    # Perform convolution using only 2 for-loops
    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel,
                axis=(1, 2)
            )
    return output
