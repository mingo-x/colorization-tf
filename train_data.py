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
import random
import subprocess
import sys

import numpy as np
from skimage import color, io

_TASK_ID = os.environ.get('SGE_TASK_ID')
if _TASK_ID is not None:
    print("Task id: {}".format(_TASK_ID))
    _TASK_ID = int(_TASK_ID) - 1
else:
    _TASK_ID = 0

_AB_TRAIN_SS_DIR = '/srv/glusterfs/xieya/data/imagenet_ab_ss/train'
_AB_VAL_SS_DIR = '/srv/glusterfs/xieya/data/imagenet_ab_ss/val'
_COLOR_DIR = '/srv/glusterfs/xieya/data/imagenet_colorized/train'
_COLOR_TRAIN_SS_DIR = '/srv/glusterfs/xieya/data/imagenet_colorized_ss/train'
_GRAY_TRAIN_DIR = '/srv/glusterfs/xieya/data/imagenet_gray/train'
_GRAY_TRAIN_SS_DIR = '/srv/glusterfs/xieya/data/imagenet_gray_ss/train'
_LOG_FREQ = 100
_ORIGINAL_TRAIN_DIR = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/train'
_ORIGINAL_TRAIN_SS_DIR = '/srv/glusterfs/xieya/data/imagenet_true_ss/train'
_ORIGINAL_VAL_DIR = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val'
_SS_RATE = 3
_TASK_NUM = 100
_TRAIN_DATA_LIST = '/home/xieya/colorization-tf/data/train.txt'


def count_img():
    line_idx = 0
    count = 0
    with open(_TRAIN_DATA_LIST, 'r') as fin:
        for line in fin:
            if line_idx % _TASK_NUM == _TASK_ID:
                img_path = line.strip()
                img_name = os.path.split(img_path)[1]
                color_path = os.path.join(_COLOR_DIR, img_name)
                if os.path.isfile(color_path):
                    count += 1
                    if count % _LOG_FREQ == 0:
                        print(count)
                        sys.stdout.flush()
                else:
                    print(img_name)
            line_idx += 1
    return count


def structure(data_list, out_dir):
    line_idx = 0
    count = 0
    with open(data_list, 'r') as fin:
        for line in fin:
            if line_idx % _TASK_NUM == _TASK_ID:
                img_path = line.strip()
                img_name = os.path.split(img_path)[1]
                class_name = img_name.split('_')[0]
                class_path = os.path.join(out_dir, class_name)
                in_path = os.path.join(out_dir, img_name)
                subprocess.check_call(['mv', in_path, class_path + '/'])
                count += 1
                if count % _LOG_FREQ == 0:
                    print(count)
                    sys.stdout.flush()
            line_idx += 1


def subsample(in_dir, out_dir):
    classes = os.listdir(_ORIGINAL_TRAIN_DIR)
    class_idx = 0
    for c in classes:
        if class_idx % _TASK_NUM == _TASK_ID:
            class_in_path = os.path.join(in_dir, c)
            class_out_path = os.path.join(out_dir, c)
            subprocess.check_call(['mkdir', '-p', class_out_path])
            kept_count = 0
            for img_name in os.listdir(class_in_path):
                if random.randint(1, _SS_RATE) == 1:
                    # Keep.
                    in_path = os.path.join(class_in_path, img_name)
                    out_path = os.path.join(class_out_path, img_name)
                    subprocess.check_call(['ln', '-s', in_path, out_path])
                    kept_count += 1

            print('Class {0} Kept {1}'.format(c, kept_count))
            sys.stdout.flush()
        class_idx += 1


def subsample_by_list(keep_list, in_dir, out_dir):
    line_idx = 0
    count = 0
    with open(keep_list, 'r') as fin:
        for line in fin:
            if line_idx % _TASK_NUM == _TASK_ID:
                img_name = line.strip().split()[0]
                in_path = os.path.join(in_dir, img_name)
                out_path = os.path.join(out_dir, img_name)
                subprocess.check_call(['ln', '-sf', in_path, out_path])
                count += 1
                if count % _LOG_FREQ == 0:
                    print(count)
                    sys.stdout.flush()
            line_idx += 1
    print('Kept {}'.format(count))


def check_zero(keep_list, in_dir):
    line_idx = 0
    count = 0
    zero_count = 0
    with open(keep_list, 'r') as fin:
        for line in fin:
            if line_idx % _TASK_NUM == _TASK_ID:
                img_name = line.strip().split()[0]
                in_path = os.path.join(in_dir, img_name)
                s = os.path.getsize(in_path)
                if s == 0:
                    print(img_name)
                    zero_count += 1

                count += 1
                if count % _LOG_FREQ == 0:
                    print(count)
                    sys.stdout.flush()
            line_idx += 1
    print('Zero {}'.format(zero_count))


def merge(out_file):
    prefix = '/srv/glusterfs/xieya/log/train_data.py.o3861047.'

    total_count = 0
    with open(out_file, 'w') as fout:
        for i in range(_TASK_NUM):
            fname = prefix + str(i+1)
            un_count = 0
            with open(fname, 'r') as fin:
                for line in fin:
                    line = line.strip()
                    if line.endswith('.JPEG'):
                        un_count += 1
                        fout.write(os.path.join(_ORIGINAL_TRAIN_DIR, line) + '\n')
            print('Task {0} Count {1}'.format(i+1, un_count))
            total_count += un_count


def merge_l():
    prefix = '/srv/glusterfs/xieya/log/train_data.py.o3862485.'

    mean_list = []
    count_list = []
    gray_count = 0
    for i in range(_TASK_NUM):
        fname = prefix + str(i + 1)
        with open(fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line.startswith('('):
                    mean, count = line[1: -1].split(', ')
                    mean = float(mean)
                    count = int(count)
                    mean_list.append(mean)
                    count_list.append(count)
                elif line.startswith('Gray'):
                    gray = int(line.split()[1])
                    gray_count += gray
            print('Task {0} finished.'.format(i + 1))

    mean_l = np.average(mean_list, weights=count_list)
    print("MeanL {0}, GrayCount {1}".format(mean_l, gray_count))


def keep_ab(keep_list, in_dir, out_dir, mean_l):
    line_idx = 0
    count = 0
    gray_count = 0
    with open(keep_list, 'r') as fin:
        for line in fin:
            if line_idx % _TASK_NUM == _TASK_ID:
                img_name = line.strip().split()[0]
                in_path = os.path.join(in_dir, img_name)
                out_path = os.path.join(out_dir, img_name)

                img_rgb = io.imread(in_path)

                if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
                    img_lab = color.rgb2lab(img_rgb)
                    img_lab[:, :, 0] = mean_l  # Remove l.
                    img_rgb = color.lab2rgb(img_lab)
                    io.imsave(out_path, img_rgb)
                else:
                    print(img_name)
                    gray_count += 1

                count += 1
                if count % _LOG_FREQ == 0:
                    print(count)
                    sys.stdout.flush()
            line_idx += 1

    print("Gray {}".format(gray_count))


def get_mean_l(keep_list, in_dir):
    line_idx = 0
    mean_l = []
    gray_count = 0
    class_graycount_dict = {}
    with open(keep_list, 'r') as fin:
        for line in fin:
            if line_idx % _TASK_NUM == _TASK_ID:
                img_name = line.strip().split()[0]
                class_name = img_name.split('/')[0]
                in_path = os.path.join(in_dir, img_name)

                img_rgb = io.imread(in_path)
                if len(img_rgb.shape) != 3 or img_rgb.shape[2] != 3:
                    gray_count += 1
                    if class_name not in class_graycount_dict:
                        class_graycount_dict[class_name] = 0
                    class_graycount_dict[class_name] += 1
                else:
                    img_lab = color.rgb2lab(img_rgb)
                    mean_l.append(np.mean(img_lab[:, :, 0]))  # Get l data.

                    if len(mean_l) % _LOG_FREQ == 0:
                        print(len(mean_l))
                        sys.stdout.flush()

            line_idx += 1

    mean = np.mean(mean_l)
    print("({0}, {1})".format(mean, len(mean_l)))
    print('Gray {}'.format(gray_count))
    for class_name in class_graycount_dict:
        print("{0} {1}".format(class_name, class_graycount_dict[class_name]))


def get_nongray_list(in_list, out_file):
    line_count = 0
    with open(in_list, 'r') as fin, open(out_file, 'w') as fout:
        for line in fin:
            img_name = line.strip().split()[0]
            in_path = os.path.join(_ORIGINAL_TRAIN_DIR, img_name)
            img = io.imread(in_path)
            if (len(img.shape) == 3 and img.shape[2] == 3):  # Non-gray.
                fout.write(line)
            line_count += 1
            if line_count % _LOG_FREQ == 0:
                print(line_count)


if __name__ == "__main__":
    # count =  count_img()
    # print('Total: {}'.format(count))
    # structure('data/uncolor_train.txt', _COLOR_DIR)
    # print("<<<<<<<<<<<<<<<")
    # subsample_by_list('/home/xieya/train.txt', _COLOR_DIR, _COLOR_TRAIN_SS_DIR)
    # subsample(_GRAY_TRAIN_DIR, _GRAY_TRAIN_SS_DIR)
    # merge('/home/xieya/zero.txt')
    # check_zero('/home/xieya/train.txt', _COLOR_DIR)
    # get_mean_l('/home/xieya/train.txt', _ORIGINAL_TRAIN_DIR)
    # merge_l()
    # keep_ab('/home/xieya/colorization-tf/resources/val.txt', _ORIGINAL_VAL_DIR, _AB_VAL_SS_DIR, 48.5744)
    get_nongray_list('/home/xieya/train.txt', '/home/xieya/colorization-tf/resources/train_nongray.txt')