import math
import os

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io, transform
from sklearn.metrics import auc

from ... import utils

_AUC_THRESHOLD = 150
_DIR = '/srv/glusterfs/xieya/image/color'
_PRIOR_PATH = '/srv/glusterfs/xieya/prior/coco_313_soft.npy'


def _get_img_name(model_name, idx):
    return os.path.join(model_name, "{}.jpg".format(idx))


def _l2_acc(gt_ab, pred_ab, prior_factor):
    '''
    L2 accuracy given different threshold.
    '''
    ab_idx = lookup.encode_points(gt_ab)
    prior = prior_factor.get_weights(ab_idx)

    l2_dist = np.sqrt(np.sum(np.square(gt_ab - pred_ab), axis=2))
    ones = np.ones_like(l2_dist)
    zeros = np.zeros_like(l2_dist)
    scores = []
    scores_rb = []
    total = np.sum(ones)
    prior_sum = np.sum(prior)
    for thr in range(0, _AUC_THRESHOLD + 1):
        score = np.sum(
            np.where(np.less_equal(l2_dist, thr), ones, zeros)) / total
        score_rb = np.sum(
            np.where(np.less_equal(l2_dist, thr), prior, zeros)) / prior_sum
        scores.append(score)
        scores_rb.append(score_rb)
    return scores, scores_rb, prior_sum


def _mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def _psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def evaluate_from_rgb(in_dir):
    '''
        AUC / image
        AUC / pixel
        PSNR of RGB / image
        RMSE of AB / pixel
    '''
    prior_factor = utils.PriorFactor(gamma=0., priorFile=_PRIOR_PATH, verbose=True)
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    gt_imgs = hf['val_ims']

    img_names = os.listdir(in_dir)
    img_names.sort()
    l2_accs = []
    l2_rb_accs = []
    prior_weights = []
    auc_scores = []
    auc_rb_scores = []
    psnr_rgb_scores = []
    mse_ab_scores = []
    x = [i for i in range(0, _AUC_THRESHOLD + 1)]
    fout = open(os.path.join(in_dir, 'metrics.txt'), 'w')

    for img_name in img_names:
        if not img_name.endswith('.jpg'):
            continue
        img_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(in_dir, img_name)
        gt_bgr = gt_imgs[img_id]
        img_rgb = io.imread(img_path)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        img_ab = color.rgb2lab(img_rgb)[:, :, 1:]
        gt_ab = color.rgb2lab(gt_rgb)[:, :, 1:]
        ab_ss = transform.downscale_local_mean(img_ab, (4, 4, 1))
        gt_ab_ss = transform.downscale_local_mean(gt_ab, (4, 4, 1))
        l2_acc, l2_rb_acc, prior_weight = _l2_acc(gt_ab_ss, ab_ss, prior_factor)
        l2_accs.append(l2_acc)
        l2_rb_accs.append(l2_rb_accs)
        prior_weights.append(prior_weight)
        auc_score = auc(x, l2_acc) / _AUC_THRESHOLD
        auc_rb_score = auc(x, l2_rb_acc) / _AUC_THRESHOLD
        auc_scores.append(auc_score)
        auc_rb_scores.append(auc_rb_score)
        psnr_rgb_score = _psnr(gt_rgb, img_rgb)
        psnr_rgb_scores.append(psnr_rgb_score)
        mse_ab_score = _mse(gt_ab_ss, ab_ss)
        mse_ab_scores.append(mse_ab_score)

        summary = '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(img_id, auc_score, auc_rb_score, psnr_rgb_score, np.sqrt(mse_ab_score))
        print(summary)
        fout.write(summary)

    # AUC / pix
    l2_acc_per_pix = np.average(l2_accs, weights=prior_weights, axis=0)
    l2_rb_acc_per_pix = np.average(l2_rb_accs, weights=prior_weights, axis=0)
    auc_per_pix = auc(x, l2_acc_per_pix) / _AUC_THRESHOLD
    auc_rb_per_pix = auc(x, l2_rb_acc_per_pix) / _AUC_THRESHOLD
    print("AUC per pix\t{0}".format(auc_per_pix))
    print("AUC rebalanced per pix\t{0}".format(auc_rb_per_pix))

    # AUC / img
    print("AUC per image\t{0}".format(np.mean(auc_scores)))
    print("AUC rebalanced per image\t{0}".format(np.mean(auc_rb_scores)))

    # PSNR RGB / img
    print("PSNR RGB per image\t{0}".format(np.mean(psnr_rgb_scores)))

    # RMSE AB / pix
    print("RMSE AB per pix\t{0}".format(np.sqrt(np.mean(mse_ab_scores))))

    fout.close()


def parse_metrics(model_name, metric_file_name):
    file_path = os.path.join(model_name, metric_file_name)
    metrics = []
    with open(file_path, 'r') as fin:
        for line in fin:
            items = line.strip().split('\t')
            if len(items) - 1 > len(metrics):
                for _ in xrange(len(items) - 1):
                    metrics.append([])
            for i in xrange(len(items) - 1):
                metrics[i].append(items[i + 1])

    return np.asarray(metrics)


def compose_imgs(model_names, idx, order):
    img = io.imread(os.path.join(_DIR, _get_img_name(model_names[order[0]], idx)))
    for i in xrange(1, len(order)):
        next_img = io.imread(os.path.join(_DIR, _get_img_name(model_names[order[i]], idx)))
        img = np.hstack((img, next_img))

    return img


def annotate(model_names):
    model_metrics = []
    for model_name in model_names:
        metrics = parse_metrics(model_name, 'ce.txt')
        metrics.extend(parse_metrics(model_name, 'metrics.txt'))
        model_metrics.append(metrics)

    model_metrics = np.asarray(model_metrics)
    n_models = len(model_names)
    n_metrics = model_metrics.shape[1]
    metrics_acc = np.zeros((n_metrics,))

    for i in xrange(100):
        metrics = model_metrics[:, :, i]
        rank_by_metrics = [
            np.argsort(metrics[:, 0]), 
            np.argsort(metrics[:, 1]), 
            np.argsort(metrics[:, 2])[::-1],
            np.argsort(metrics[:, 3])[::-1],
            np.argsort(metrics[:, 4])[::-1],
            np.argsort(metrics[:, 5])
        ]

        order = np.random.permutation(n_models)
        img = compose_imgs(model_names, i, order)
        io.imshow(img)
        plt.show()
        rank_input = raw_input('Please input your rank:')
        rank_input = rank_input.strip().split(',')
        rank = [-1] * n_models
        for j in xrange(n_models):
            rank[j] = int(rank_input[order[j]])

        for j in xrange(n_metrics):
            rank_by_m = rank_by_metrics[:, j]
            hit = np.sum(rank_by_m[order] == rank)
            metrics_acc[j] += hit
            print(j, metrics[:, j], rank_by_m)

    print(metrics_acc)


if __name__ == "__main__":
    model_names = ['', '', '']
    annotate(model_names)
