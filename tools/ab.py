import os

import numpy as np
from skimage import io, color

import utils

_OUTPUT_DIR = '/srv/glusterfs/xieya/image/ab'

def draw_ab_space_given_l(l, save=True):
    cell_size = 10
    canvas = np.zeros((23 * cell_size, 23 * cell_size, 3))
    canvas[:, :, 0].fill(l)
    for i in xrange(23):
        for j in xrange(23):
            a = (i - 11) * 10
            b = (j - 11) * 10
            canvas[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 1].fill(a)
            canvas[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 2].fill(b)

    canvas = color.lab2rgb(canvas)
    if save:
        io.imsave(os.path.join(_OUTPUT_DIR, '{}.jpg'.format(l)), canvas)
    else:
        return canvas


def draw_ab_space():
    if not os.path.exists(_OUTPUT_DIR):
        os.makedirs(_OUTPUT_DIR)
        print('Created dir {}.'.format(_OUTPUT_DIR))

    for l in xrange(0, 101, 10):
        draw_ab_space_given_l(l)
        print(l)


def _weights_to_image(weights, out_name="", save=True):
    weights /= np.max(weights)  # Rescaling.
    grid = np.load('resources/pts_in_hull.npy')

    cell_size = 10
    canvas = np.zeros((23 * cell_size, 23 * cell_size), dtype=np.float32)  # Grayscale canvas.
    # canvas.fill(0.5)

    for i in xrange(len(weights)):
        a, b = grid[i]
        x = int(a / 10) + 11
        y = int(b / 10) + 11
        canvas[x * cell_size: (x + 1) * cell_size, y * cell_size: (y + 1) * cell_size] = weights[i]

    canvas = color.gray2rgb(canvas)
    if save:
        io.imsave(os.path.join(_OUTPUT_DIR, '{}.jpg'.format(out_name)), canvas)
    else:
        return canvas


def hist_to_image(hist_path):
    hist = np.load(hist_path)
    out_name = os.path.splitext(os.path.split(hist_path)[1])[0]
    _weights_to_image(hist, out_name)


def prior_to_image(prior_path):
    prior_name = os.path.splitext(os.path.split(prior_path)[1])[0]
    distribution = np.load(prior_path)
    _weights_to_image(distribution, prior_name + "_distribution")
    prior_factor_0 = utils.PriorFactor(priorFile=prior_path, gamma=0)
    prior_factor_5 = utils.PriorFactor(priorFile=prior_path, gamma=.5)
    _weights_to_image(prior_factor_0.prior_factor, prior_name + "_weights_g0")
    _weights_to_image(prior_factor_5.prior_factor, prior_name + "_weights_g5")


def hist_to_image_with_ab(hist_path):
    hist = np.load(hist_path)
    out_name = os.path.splitext(os.path.split(hist_path)[1])[0]
    alpha = _weights_to_image(hist, save=False)
    alpha *= 255
    alpha = alpha.astype(np.uint8)
    rgb = draw_ab_space_given_l(0, False)
    rgba = np.concatenate((rgb, alpha), -1)
    io.imsave(os.path.join(_OUTPUT_DIR, '{}.png'.format(out_name)), rgba)


if __name__ == "__main__":
    # draw_ab_space()
    # hist_to_image('/srv/glusterfs/xieya/image/ab/tf_coco_5_38k_hist.npy')
    # prior_to_image('/srv/glusterfs/xieya/prior/coco_313_soft.npy')
    hist_to_image_with_ab('/srv/glusterfs/xieya/image/ab/tf_coco_5_38k_hist.npy')
