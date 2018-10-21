import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color

import utils

_OUTPUT_DIR = '/srv/glusterfs/xieya/image/ab'


def draw_ab_space_given_l(l, save=True, weights=None, name=''):
    cell_size = 10
    canvas = np.zeros((23 * cell_size, 23 * cell_size, 3))
    if l < 30:
        canvas[:, :, 0].fill(100)
    else:
        canvas[:, :, 0].fill(0)

    grid = np.load('resources/pts_in_hull.npy')
    for i in xrange(313):
        if weights is not None and weights[i] < 1e-8:
            continue
        a, b = grid[i]
        i = int(a / 10) + 11
        j = int(b / 10) + 11
        canvas[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 0] = l
        canvas[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 1].fill(a)
        canvas[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 2].fill(b)

    canvas = color.lab2rgb(canvas)
    if save:
        io.imsave(os.path.join(_OUTPUT_DIR, '{0}_{1}.jpg'.format(name, l)), canvas)
    else:
        return canvas


def draw_ab_space(weights_path=None):
    if not os.path.exists(_OUTPUT_DIR):
        os.makedirs(_OUTPUT_DIR)
        print('Created dir {}.'.format(_OUTPUT_DIR))

    if weights_path is None:
        weights = None
        name = ''
    else:
        weights = np.load(weights_path)
        name = os.path.splitext(os.path.split(weights_path)[1])[0]

    for l in xrange(0, 101, 10):
        draw_ab_space_given_l(l, weights=weights, name=name)
        print(l)


def _weights_to_image(weights, out_name="", save=True, fill=0.):
    weights /= np.sum(weights)  # Rescaling.
    grid = np.load('resources/pts_in_hull.npy')

    cell_size = 10
    canvas = np.zeros((23 * cell_size, 23 * cell_size), dtype=np.float32)  # Grayscale canvas.
    canvas.fill(fill)

    for i in xrange(len(weights)):
        if weights[i] < 1e-8:
            continue
        a, b = grid[i]
        x = int(a / 10) + 11
        y = int(b / 10) + 11
        canvas[x * cell_size: (x + 1) * cell_size, y * cell_size: (y + 1) * cell_size] = weights[i]

    if save:
        plt.imsave(os.path.join(_OUTPUT_DIR, '{}.jpg'.format(out_name)), canvas, vmin=0, vmax=1)
    else:
        return canvas


def hist_to_image(hist_path):
    hist = np.load(hist_path)
    out_name = os.path.splitext(os.path.split(hist_path)[1])[0]
    _weights_to_image(hist, out_name, fill=0.5)


def abl_hists_to_image(hist_path):
    hists = np.load(hist_path)
    out_name = os.path.splitext(os.path.split(hist_path)[1])[0]
    for l in xrange(0, 101, 10):
        hist = np.sum(hists[max(0, l - 5): min(101, l + 5)], axis=0)
        hist /= np.sum(hist)
        cell_size = 10
        canvas = np.zeros((23 * cell_size, 23 * cell_size, 3))
        if l < 30:
            canvas[:, :, 0].fill(100)
        else:
            canvas[:, :, 0].fill(0)
        grid = np.load('resources/pts_in_hull.npy')
        for c in xrange(313):
            if hist[c] < 1e-8:
                continue
            a, b = grid[c]
            i = int(a / 10) + 11
            j = int(b / 10) + 11
            canvas[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 0] = l
            canvas[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 1].fill(a)
            canvas[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size, 2].fill(b)

        canvas = color.lab2rgb(canvas)
        io.imsave(os.path.join(_OUTPUT_DIR, '{0}_{1}.jpg'.format(out_name, l)), canvas)

        _weights_to_image(hist, out_name + "_heatmap_{}".format(l), fill=0.5)


def prior_to_image(prior_path):
    prior_name = os.path.splitext(os.path.split(prior_path)[1])[0]
    distribution = np.load(prior_path)
    _weights_to_image(distribution, prior_name + "_distribution")
    prior_factor_0 = utils.PriorFactor(priorFile=prior_path, gamma=0)
    prior_factor_5 = utils.PriorFactor(priorFile=prior_path, gamma=.5)
    _weights_to_image(prior_factor_0.prior_factor, prior_name + "_weights_g0")
    _weights_to_image(prior_factor_5.prior_factor, prior_name + "_weights_g5")


def hist_to_image_as_alpha(hist_path):
    hist = np.load(hist_path)
    out_name = os.path.splitext(os.path.split(hist_path)[1])[0]
    alpha = _weights_to_image(hist, save=False)
    alpha = alpha[:, :, np.newaxis]
    # alpha *= 255
    # alpha = alpha.astype(np.uint8)
    for l in xrange(0, 101, 10):
        rgba = _add_alpha_to_ab_space(alpha, l)
        io.imsave(os.path.join(_OUTPUT_DIR, 'alpha_{0}_{1}.png'.format(out_name, l)), rgba)


def hist_to_image_as_mask(hist_path):
    hist = np.load(hist_path)
    threshold = np.mean(hist) / np.max(hist)
    out_name = os.path.splitext(os.path.split(hist_path)[1])[0]
    mask = _weights_to_image(hist, save=False)
    alpha = np.zeros_like(mask, dtype=np.float32)
    alpha[mask > threshold] = 1.
    alpha = alpha[:, :, np.newaxis]
    # alpha *= 255
    # alpha = alpha.astype(np.uint8)
    for l in xrange(0, 101, 10):
        rgba = _add_alpha_to_ab_space(alpha, l)
        io.imsave(os.path.join(_OUTPUT_DIR, 'mask_{0}_{1}.png'.format(out_name, l)), rgba)


def _add_alpha_to_ab_space(alpha, l):
    rgb = draw_ab_space_given_l(l, False)
    rgba = np.concatenate((rgb, alpha), -1)

    return rgba


def hist_of_img_list(img_list):
    lookup = utils.LookupEncode('resources/pts_in_hull.npy')
    hist = np.zeros((313,), dtype=np.float32)
    for img_id in img_list:
        # img_name = os.path.splitext(os.path.split(img_path)[1])[0]
        img_path = "/srv/glusterfs/xieya/image/color/tf_coco_5_38k_noeval/{0}.jpg".format(img_id)
        img_rgb = io.imread(img_path)
        img_lab = color.rgb2lab(img_rgb)
        img_ab = img_lab[:, :, 1:]
        ab_idx = lookup.encode_points(img_ab).flatten()
        for idx in xrange(313):
            hist[idx] += len(ab_idx[ab_idx == idx])
        print(img_id)

    _weights_to_image(hist, "img_list_ab_hist", fill=0.5)

    i_sorted = np.argsort(hist)
    print(i_sorted[::-1])


def compare_pred_with_gt(pred_hist_path, gt_hist_path, diff=1e-3):
    pred_hist = np.load(pred_hist_path)
    pred_hist /= np.sum(pred_hist)
    pred_hist_name = os.path.splitext(os.path.split(pred_hist_path)[1])[0]
    gt_hist = np.load(gt_hist_path)
    more_hist = np.zeros_like(gt_hist, dtype=np.float32)
    more_hist[pred_hist > gt_hist + diff] = 1.
    more_alpha = _weights_to_image(more_hist, save=False)
    more_alpha = more_alpha[:, :, np.newaxis]
    less_hist = np.zeros_like(gt_hist, dtype=np.float32)
    less_hist[pred_hist + diff < gt_hist] = 1.
    less_alpha = _weights_to_image(less_hist, save=False)
    less_alpha = less_alpha[:, :, np.newaxis]

    for l in xrange(0, 101, 10):
        rgba_more = _add_alpha_to_ab_space(more_alpha, l)
        rgba_less = _add_alpha_to_ab_space(less_alpha, l)
        io.imsave(os.path.join(_OUTPUT_DIR, 'comp_more_{0}_{1}.png'.format(pred_hist_name, l)), rgba_more)
        io.imsave(os.path.join(_OUTPUT_DIR, 'comp_less_{0}_{1}.png'.format(pred_hist_name, l)), rgba_less)
    

def merge(out_name):
    dir_path = '/srv/glusterfs/xieya/image/ab'
    patterns = ['313_ab_1_heatmap_{}.jpg', 'tf_coco_5_38k_abl_rgb_hist_heatmap_{}.jpg', 'tf_coco_5_38k_abl_hist_heatmap_{}.jpg']
    for l in xrange(0, 101, 10):
        imgs = []
        for p in patterns:
            img_path = os.path.join(dir_path, p.format(l))
            imgs.append(io.imread(img_path))
        imgs = np.asarray(imgs)
        canvas = np.hstack(imgs)
        io.imsave(os.path.join(dir_path, '{0}_{1}.jpg'.format(out_name, l)), canvas)


if __name__ == "__main__":
    # draw_ab_space('/srv/glusterfs/xieya/image/ab/tf_coco_5_38k_rgb_hist.npy')
    # hist_to_image('/srv/glusterfs/xieya/image/ab/tf_coco_5_38k_c313_hist.npy')
    # prior_to_image('/srv/glusterfs/xieya/prior/coco_313_soft.npy')
    # hist_to_image_as_alpha('/srv/glusterfs/xieya/image/ab/tf_coco_5_38k_c313_hist.npy')
    # hist_to_image_as_mask('/srv/glusterfs/xieya/image/ab/tf_coco_5_38k_c313_hist.npy')
    # redish_img_list = [
    #     294, 295, 296, 297, 301, 344, 347, 350, 358, 375, 380, 386, 388, 389, 390, 391, 394, 395, 397, 398
    # ]
    # hist_of_img_list(redish_img_list)
    # compare_pred_with_gt('/srv/glusterfs/xieya/image/ab/tf_coco_5_38k_hist.npy', '/srv/glusterfs/xieya/prior/coco_313_soft.npy')
    # abl_hists_to_image('/srv/glusterfs/xieya/image/ab/tf_coco_5_38k_abl_hist.npy')
    merge('heatmap_comp')
