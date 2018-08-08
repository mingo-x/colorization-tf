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

import os
import sys

import cv2
import monotonic
import numpy as np
from skimage import color, io
import tensorflow as tf

import demo
import utils

_TASK_ID = os.environ.get('SGE_TASK_ID')
if _TASK_ID is not None:
    print("Task id: {}".format(_TASK_ID))
    _TASK_ID = int(_TASK_ID) - 1

_CKPT_PATH = '/srv/glusterfs/xieya/colorization-gan/models/model.ckpt-499000'
_COLOR_DIR = '/srv/glusterfs/xieya/data/imagenet_colorized/'
_GRAY_DIR = '/srv/glusterfs/xieya/data/imagenet_gray/'
_IMG_LIST_PATH = '/home/xieya/colorization-tf/data/train.txt'
_LOG_FREQ = 100
_VAL_DIR = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val'
_TASK_NUM = 100
_BATCH_SIZE = 32
_INPUT_SIZE = 224
_T = 2.63


def _colorize(img_paths_batch, out_dir, model, input_tensor, sess):
    img_l_batch = []
    img_l_rs_batch = []
    for img_path in img_paths_batch:
        img = cv2.imread(img_path)
        print img_path
        img_rs = cv2.resize(img, (_INPUT_SIZE, _INPUT_SIZE))

        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_l = img_l[:, :, None]
        img_l_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
        img_l_rs = img_l_rs[:, :, None]

        img_l = (img_l.astype(dtype=np.float32)) / 255. * 100 - 50
        img_l_rs = (img_l_rs.astype(dtype=np.float32)) / 255.0 * 100 - 50

        img_l_batch.append(img_l)
        img_l_rs_batch.append(img_l_rs)
    img_l_batch = np.asarray(img_l_batch)
    img_l_rs_batch = np.asarray(img_l_rs_batch)
    print(img_l_batch.shape, img_l_rs_batch.shape)
    exit()

    img_313_rs_batch = sess.run(model, feed_dict={input_tensor: img_l_rs_batch})

    for i in xrange(len(img_paths_batch)):
        img_rgb, _ = utils.decode(img_l_batch[i: i + 1], img_313_rs_batch[i: i + 1], _T)
        img_name = os.path.split(img_paths_batch[i])[1]
        io.imsave(os.path.join(out_dir, img_name), img_rgb)


def _colorize_data_wrapper(phase):
    print("Phase: {}".format(phase))
    in_dir = _GRAY_DIR + phase
    out_dir = _COLOR_DIR + phase
    img_names = os.listdir(in_dir)

    input_tensor = tf.placeholder(
        tf.float32, shape=(_BATCH_SIZE, _INPUT_SIZE, _INPUT_SIZE, 1))
    model = demo._get_model(input_tensor)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, _CKPT_PATH)

        start_time = monotonic.monotonic()
        for i in xrange(0, len(img_names), _BATCH_SIZE):
            if i % (_BATCH_SIZE * _LOG_FREQ) == 0:
                print("Image count: {0} Time: {1}".format(i, monotonic.monotonic() - start_time))
                start_time = monotonic.monotonic()

            img_names_batch = img_names[i * _BATCH_SIZE: min(len(img_names), (i + 1) * _BATCH_SIZE)]
            img_paths_batch = map(lambda x: os.path.join(in_dir, x), img_names_batch)
            _colorize(img_paths_batch, out_dir, model, input_tensor, sess)

def _log(curr_idx):
    if (curr_idx / _TASK_NUM) % _LOG_FREQ == 0:
        print(curr_idx / _TASK_NUM)
        sys.stdout.flush()


def _to_gray(img_path, out_dir):
    img = io.imread(img_path)
    img_gray = color.rgb2gray(img)
    img_name = os.path.split(img_path)[1]
    io.imsave(os.path.join(out_dir, img_name), img_gray)


def _training_data(func):
    print("Training started...")
    sys.stdout.flush()
    with open(_IMG_LIST_PATH, 'r') as fin:
        line_idx = 0
        for line in fin:
            if line_idx % _TASK_NUM == _TASK_ID:
                img_path = line.strip()
                func(img_path, _GRAY_DIR + 'train')
                _log(line_idx)
            line_idx += 1


def _validation_data(func):
    img_names = os.listdir(_VAL_DIR)
    img_names = filter(lambda img_name: img_name.endswith('.JPEG'), img_names)
    img_names.sort()
    print("Validation total: {}".format(len(img_names)))
    sys.stdout.flush()

    for img_idx in range(len(img_names)):
        if img_idx % _TASK_NUM == _TASK_ID:
            func(os.path.join(_VAL_DIR, img_names[img_idx]), _GRAY_DIR + 'val')
            _log(img_idx)


def main():
    # _validation_data(_to_gray)
    # _training_data(_to_gray)
    _colorize_data_wrapper('val')
    _colorize_data_wrapper('train')

if __name__ == "__main__":
    main()
