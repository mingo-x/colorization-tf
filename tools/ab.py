import os

import numpy as np
from skimage import io, color

_OUTPUT_DIR = '/srv/glusterfs/xieya/image/ab'

def draw_ab_space_given_l(l):
    cell_size = 5
    canvas = np.zeros((23 * cell_size, 23 * cell_size, 3))
    canvas[:, :, 0].fill(l)
    for i in xrange(23):
        for j in xrange(23):
            a = (i - 11) * 10
            b = (j - 11) * 10
            np[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 1].fill(a)
            np[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 1].fill(b)

    canvas = color.lab2rgb(canvas)
    io.imsave(os.path.join(_OUTPUT_DIR, '{}.jpg'.format(l)), canvas)


def draw_ab_space():
    if not os.path.exists(_OUTPUT_DIR):
        os.makedirs(_OUTPUT_DIR)
        
    for l in xrange(0, 101, 10):
        draw_ab_space_given_l(l)


if __name__ == "__main__":
    draw_ab_space()
