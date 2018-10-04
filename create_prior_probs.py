#!/srv/glusterfs/xieya/anaconda2/bin/python

# ----- Parameters passed to the cluster -------
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue

#$ -t 1:100

#$ -S /srv/glusterfs/xieya/anaconda2/bin/python

#$ -l h_rt=5:59:59

#$ -l h_vmem=8G

#$ -o /srv/glusterfs/xieya/log

#$ -e /srv/glusterfs/xieya/log

#$ -j y

import h5py
import numpy as np
from skimage.io import imread
from skimage import color
from skimage.transform import resize
# import tensorflow as tf 

import os
# import random
import sys


_GRID_PATH = ''
_LOG_FREQ = 100
_N_CLASSES = 313
_TASK_NUM = 100
_TASK_ID = os.environ.get('SGE_TASK_ID')
if _TASK_ID is not None:
    print("Task id: {}".format(_TASK_ID))
    _TASK_ID = int(_TASK_ID) - 1
else:
    _TASK_ID = 0


def get_index(ab):
    ab = ab[:, np.newaxis, :]
    distance = np.sum(np.square(ab - points), axis=2)
    index = np.argmin(distance, axis=1)
    return index


def get_file_list():
    filename_lists = []
    img_idx = 0
    for img_f in lists_f:
        if img_idx % _TASK_NUM == _TASK_ID:
            img_f = img_f.strip()
            filename_lists.append(img_f)
        img_idx += 1
    return filename_lists


def cal_prob():
    out_path = '/srv/glusterfs/xieya/prior/{0}_onehot_{1}.npy'.format(_N_CLASSES, _TASK_ID)
    if os.path.isfile(out_path):
        print('Done.')
        return

    filename_lists = get_file_list()
    counter = 0
    # random.shuffle(filename_lists)

    # construct graph
    # in_data = tf.placeholder(tf.float64, [None, 2])
    # expand_in_data = tf.expand_dims(in_data, axis=1)

    # distance = tf.reduce_sum(tf.square(expand_in_data - points), axis=2)
    # index = tf.argmin(distance, axis=1)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    for img_f in filename_lists:
        img_f = img_f.strip()
        if not os.path.isfile(img_f):
            print(img_f)
            continue
        img = imread(img_f)
        img = resize(img, (224, 224), preserve_range=True)
        if len(img.shape) != 3 or img.shape[2] != 3:
            continue
        img_lab = color.rgb2lab(img)
        img_lab = img_lab.reshape((-1, 3))
        img_ab = img_lab[:, 1:]
        # nd_index = sess.run(index, feed_dict={in_data: img_ab})
        nd_index = get_index(img_ab)
        for i in nd_index:
            i = int(i)
            probs[i] += 1

        if counter % _LOG_FREQ == 0:
            print(counter)
            sys.stdout.flush()
        counter += 1

    # sess.close()
    # probs = probs / np.sum(probs)
    np.save(out_path, probs)


def cal_prob_coco():
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    train_origs = hf['train_ims']  # BGR format
    counter = 0

    for i in xrange(len(train_origs)):
        if i % _TASK_NUM != _TASK_ID:
            continue
        img_bgr = train_origs[i]
        img_rgb = img_bgr[:, :, ::-1]
        img_lab = color.rgb2lab(img_rgb)
        img_lab = img_lab.reshape((-1, 3))
        img_ab = img_lab[:, 1:]
        nd_index = get_index(img_ab)
        for i in nd_index:
            i = int(i)
            probs[i] += 1

        if counter % _LOG_FREQ == 0:
            print(counter)
            sys.stdout.flush()
        counter += 1

    np.save('/srv/glusterfs/xieya/prior/coco_{0}_onehot_{1}'.format(_N_CLASSES, _TASK_ID), probs)


def merge():
    print("Merging...")
    probs = np.zeros((_N_CLASSES), dtype=np.float64)
    path_pattern = '/srv/glusterfs/xieya/prior/{0}_onehot_{1}.npy'
    for i in xrange(_TASK_NUM):
        file_path = path_pattern.format(_N_CLASSES, i)
        p = np.load(file_path)
        probs += p
        print(i)
    probs = probs / np.sum(probs)
    np.save('/srv/glusterfs/xieya/prior/{}_onehot'.format(_N_CLASSES), probs)


if __name__ == "__main__":
    lists_f = open('/srv/glusterfs/xieya/data/imagenet1k_uncompressed/train.txt')
    if _N_CLASSES == 313:
        _GRID_PATH = '/home/xieya/colorization-tf/resources/pts_in_hull.npy'
    else:
        _GRID_PATH = '/home/xieya/colorfromlanguage/priors/full_lab_grid_10.npy'
    points = np.load(_GRID_PATH)
    points = points.astype(np.float64)
    points = points[None, :, :]
    probs = np.zeros((_N_CLASSES), dtype=np.float64)
    print("Number of classes: {}.".format(_N_CLASSES))
    #print("Imagenet.")
    #cal_prob()
    # print("Coco.")
    # cal_prob_coco()
    merge()
