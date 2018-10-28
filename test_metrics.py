import math
import os

import cv2
import h5py
import numpy as np
from skimage import color, io, transform
io.use_plugin('matplotlib')
from sklearn.metrics import auc

import utils

_AUC_THRESHOLD = 150
_DIR = '/srv/glusterfs/xieya/image/color'
_LOG_FREQ = 32
_PRIOR_PATH = '/srv/glusterfs/xieya/prior/coco_313_soft.npy'


def _compose_imgs(model_names, idx, order):
    img = io.imread(os.path.join(_DIR, _get_img_name(model_names[order[0]], idx)))
    for i in xrange(1, len(order)):
        next_img = io.imread(os.path.join(_DIR, _get_img_name(model_names[order[i]], idx)))
        img = np.hstack((img, next_img))
    return img


def _get_img_name(model_name, idx):
    return os.path.join(model_name, "{}.jpg".format(idx))


def _get_pairwise_order(scores, sort_order):
    n_samples = len(scores)
    order = []
    for i in xrange(n_samples):
        for j in xrange(i + 1, n_samples):
            if sort_order == 1:
                order.append(scores[i] < scores[j])
            else:
                order.append(scores[i] > scores[j])

    return np.asarray(order)


def _l2_acc(gt_ab, pred_ab, prior_factor_0, prior_factor_5):
    '''
    L2 accuracy given different threshold.
    '''
    ab_idx = lookup.encode_points(gt_ab)
    prior_0 = prior_factor_0.get_weights(ab_idx)
    prior_5 = prior_factor_5.get_weights(ab_idx)

    l2_dist = np.sqrt(np.sum(np.square(gt_ab - pred_ab), axis=2))
    ones = np.ones_like(l2_dist)
    zeros = np.zeros_like(l2_dist)
    scores = []
    scores_rb_0 = []
    scores_rb_5 = []
    total = np.sum(ones)
    prior_0_sum = np.sum(prior_0)
    prior_5_sum = np.sum(prior_5)
    for thr in range(0, _AUC_THRESHOLD + 1):
        score = np.sum(
            np.where(np.less_equal(l2_dist, thr), ones, zeros)) / total
        score_rb_0 = np.sum(
            np.where(np.less_equal(l2_dist, thr), prior_0, zeros)) / prior_0_sum
        score_rb_5 = np.sum(
            np.where(np.less_equal(l2_dist, thr), prior_5, zeros)) / prior_5_sum
        scores.append(score)
        scores_rb_0.append(score_rb_0)
        scores_rb_5.append(score_rb_5)
    return scores, scores_rb_0, prior_0_sum, scores_rb_5, prior_5_sum


def _mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def _parse_annotation():
    annotation = []
    pairwise = []
    with open('/srv/glusterfs/xieya/rank.txt', 'r') as fin:
        for line in fin:
            items = line.strip().split('\t')
            rank = [int(items[i]) for i in xrange(1, len(items))]
            order = np.argsort(rank)
            annotation.append(order)
            pairwise_order = _get_pairwise_order(rank, 1)
            pairwise.append(pairwise_order)
    return np.asarray(annotation), np.asarray(pairwise)


def _parse_metrics(model_name, metric_file_name):
    file_path = os.path.join(_DIR, os.path.join(model_name, metric_file_name))
    metrics = []
    with open(file_path, 'r') as fin:
        for line in fin:
            items = line.strip().split('\t')
            scores = []
            for i in xrange(1, len(items)):
                scores.append(float(items[i]))
            metrics.append(scores)

    return np.asarray(metrics)


def _psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def annotate(model_names):
    n_models = len(model_names)
    with open(os.path.join(_DIR, model_names[0] + "/ce.txt")) as fin, open('/srv/glusterfs/xieya/rank.txt', 'a+') as fout:
        for i in xrange(9, 100):
            order = np.random.permutation(n_models)
            print(fin.readline().strip().split('\t')[0])
            img = _compose_imgs(model_names, i, order)
            io.imshow(img)
            io.show()
            rank_input = raw_input('Please input your rank:')
            rank_input = rank_input.strip().split(',')
            rank = ['-1'] * n_models
            for j in xrange(n_models):
                rank[j] = rank_input[order[j]]
            rank_str = "\t".join(rank)
            fout.write("{0}\t{1}\n".format(i, rank_str))
            fout.flush()


def compare_metrics(model_names, metric_file_name, sort_order):
    model_metrics = []  # n_models * n_samples * n_metrics
    for model_name in model_names:
        model_metrics.append(_parse_metrics(model_name, metric_file_name))

    model_metrics = np.asarray(model_metrics)
    n_models, _, n_metrics = model_metrics.shape

    annotation, pairwise_annotation = _parse_annotation()  # n_annotations * n_models, n_annotations * n_pairs
    n_annotations = annotation.shape[0]
    annotation = np.transpose(annotation, (1, 0))

    for i in xrange(n_metrics):
        scores = model_metrics[:, 0: n_annotations, i]  # n_models * n_annotations
        orders = np.argsort(scores, axis=0)[::sort_order[i]]
        match_total = np.sum(annotation == orders)
        match_top = np.sum(annotation[0] == orders[0])
        pairwise_orders = [_get_pairwise_order(scores[:, j], sort_order[i]) for j in xrange(n_annotations)]
        pairwise_orders = np.asarray(pairwise_orders)
        print(orders[:, 0], annotation[:, 0])
        print(pairwise_orders[0], pairwise_annotation[0])
        match_pairwise = np.sum(pairwise_annotation == pairwise_orders)

        print("Metrics {0} total match {1} top match {2} pairwise match {3}".format(i, match_total, match_top, match_pairwise))


def evaluate_from_rgb(in_dir):
    '''
        AUC / image
        AUC / pixel
        PSNR of RGB / image
        RMSE of AB / pixel
    '''
    prior_factor_0 = utils.PriorFactor(gamma=0., priorFile=_PRIOR_PATH, verbose=True)
    prior_factor_5 = utils.PriorFactor(gamma=0.5, priorFile=_PRIOR_PATH, verbose=True)
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    gt_imgs = hf['val_ims']

    img_names = os.listdir(in_dir)
    img_names.sort()
    l2_accs = []
    l2_rb_0_accs = []
    l2_rb_5_accs = []
    prior_0_weights = []
    prior_5_weights = []
    auc_scores = []
    auc_rb_0_scores = []
    auc_rb_5_scores = []
    psnr_rgb_scores = []
    mse_ab_scores = []
    x = [i for i in range(0, _AUC_THRESHOLD + 1)]
    fout = open(os.path.join(in_dir, 'metrics.txt'), 'w')
    img_count = 0

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
        l2_acc, l2_rb_0_acc, prior_0_weight, l2_rb_5_acc, prior_5_weight = _l2_acc(gt_ab_ss, ab_ss, prior_factor_0, prior_factor_5)
        l2_accs.append(l2_acc)
        l2_rb_0_accs.append(l2_rb_0_acc)
        l2_rb_5_accs.append(l2_rb_5_acc)
        prior_0_weights.append(prior_0_weight)
        prior_5_weights.append(prior_5_weight)
        auc_score = auc(x, l2_acc) / _AUC_THRESHOLD
        auc_rb_0_score = auc(x, l2_rb_0_acc) / _AUC_THRESHOLD
        auc_rb_5_score = auc(x, l2_rb_5_acc) / _AUC_THRESHOLD
        auc_scores.append(auc_score)
        auc_rb_0_scores.append(auc_rb_0_score)
        auc_rb_5_scores.append(auc_rb_5_score)
        psnr_rgb_score = _psnr(gt_rgb, img_rgb)
        psnr_rgb_scores.append(psnr_rgb_score)
        mse_ab_score = _mse(gt_ab_ss, ab_ss)
        mse_ab_scores.append(mse_ab_score)

        summary = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(img_id, auc_score, auc_rb_0_score, auc_rb_5_score, psnr_rgb_score, np.sqrt(mse_ab_score))
        # print(summary)
        fout.write(summary)
        img_count += 1
        if img_count % _LOG_FREQ == 0:
            print(img_count)
            fout.flush()

    l2_accs = np.asarray(l2_accs)
    prior_0_weights = np.asarray(prior_0_weights)
    prior_5_weights = np.asarray(prior_5_weights)
    l2_rb_0_accs = np.asarray(l2_rb_0_accs)
    l2_rb_5_accs = np.asarray(l2_rb_5_accs)
    
    # AUC / pix
    l2_acc_per_pix = np.mean(l2_accs, axis=0)
    l2_rb_0_acc_per_pix = np.average(l2_rb_0_accs, weights=prior_0_weights, axis=0)
    l2_rb_5_acc_per_pix = np.average(l2_rb_5_accs, weights=prior_5_weights, axis=0)
    auc_per_pix = auc(x, l2_acc_per_pix) / _AUC_THRESHOLD
    auc_rb_0_per_pix = auc(x, l2_rb_0_acc_per_pix) / _AUC_THRESHOLD
    auc_rb_5_per_pix = auc(x, l2_rb_5_acc_per_pix) / _AUC_THRESHOLD
    print("AUC per pix\t{0}".format(auc_per_pix))
    print("AUC rebalanced gamma 0. per pix\t{0}".format(auc_rb_0_per_pix))
    print("AUC rebalanced gamma 0.5 per pix\t{0}".format(auc_rb_5_per_pix))

    # AUC / img
    print("AUC per image\t{0}".format(np.mean(auc_scores)))
    print("AUC rebalanced gamma 0. per image\t{0}".format(np.mean(auc_rb_0_scores)))
    print("AUC rebalanced gamma 0.5 per image\t{0}".format(np.mean(auc_rb_5_scores)))

    # PSNR RGB / img
    print("PSNR RGB per image\t{0}".format(np.mean(psnr_rgb_scores)))

    # RMSE AB / pix
    print("RMSE AB per pix\t{0}".format(np.sqrt(np.mean(mse_ab_scores))))

    fout.close()
    print(in_dir)


if __name__ == "__main__":
    model_names = ['tf_224_1_476k', 'tf_coco_24k', 'language_2_18k']
    lookup = utils.LookupEncode('resources/pts_in_hull.npy')
    # annotate(model_names)
    # compare_metrics(model_names, 'ce.txt', [1, 1])
    evaluate_from_rgb('/srv/glusterfs/xieya/image/color/language_5_31k')

