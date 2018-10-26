import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

_DIR = '/srv/glusterfs/xieya/image/color'


def _get_img_name(model_name, idx):
    return os.path.join(model_name, "{}.jpg".format(idx))


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
