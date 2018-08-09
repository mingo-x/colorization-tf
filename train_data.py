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
import subprocess
import sys

_TASK_ID = os.environ.get('SGE_TASK_ID')
if _TASK_ID is not None:
    print("Task id: {}".format(_TASK_ID))
    _TASK_ID = int(_TASK_ID) - 1

_COLOR_DIR = '/srv/glusterfs/xieya/data/imagenet_colorized/train'
_GRAY_TRAIN_DIR = '/srv/glusterfs/xieya/data/imagenet_gray/train'
_LOG_FREQ = 100
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


def structure(out_dir):
    line_idx = 0
    count = 0
    with open(_TRAIN_DATA_LIST, 'r') as fin:
        for line in fin:
            if line_idx % _TASK_NUM == _TASK_ID:
                img_path = line.strip()
                img_name = os.path.split(img_path)[1]
                class_name = img_name.split('_')[0]
                class_path = os.path.join(out_dir, class_name)
                in_path = os.path.join(out_dir, img_name)
                if os.path.isfile(in_path):
                    subprocess.check_call(['mv', in_path, class_path + '/'])
                    count += 1
                    if count % _LOG_FREQ == 0:
                        print(count)
                        sys.stdout.flush()
                else:
                    print(img_name)
            line_idx += 1


if __name__ == "__main__":
    # count =  count_img()
    # print('Total: {}'.format(count))
    structure(_COLOR_DIR)
