import numpy as np

def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images.

    Args:
        images: numpy.ndarray of shape (m, h, w, c)
        kernel_shape: tuple of (kh, kw)
        stride: tuple of (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        numpy.ndarray of pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate the output dimensions
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, out_h, out_w, c))

    # Loop through output height and width
    for i in range(out_h):
        for j in range(out_w):
            # Extract slice for pooling
            slice_img = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(slice_img, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(slice_img, axis=(1, 2))

    return output
