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

from skimage import color, io

_TASK_ID = os.environ.get('SGE_TASK_ID')
if _TASK_ID is not None:
    print("Task id: {}".format(_TASK_ID))
    _TASK_ID = int(_TASK_ID) - 1

_COLOR_DIR = '/srv/glusterfs/xieya/data/imagenet_colorized/'
_GRAY_DIR = '/srv/glusterfs/xieya/data/imagenet_gray/'
_IMG_LIST_PATH = '/home/xieya/colorization-tf/data/train.txt'
_LOG_FREQ = 100
_VAL_DIR = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val'
_TASK_NUM = 100
_BATCH_SIZE = 32
_INPUT_SIZE = 224


def _colorize(img_paths):
    img_l_batch = []
    img_l_rs_batch = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_rs = cv2.resize(img, (_INPUT_SIZE, _INPUT_SIZE))

        # Input gray image.
        img_l = img[:, :, None]
        img_l_rs = img_rs[:, :, None]
        img_l = (img_l.astype(dtype=np.float32)) / 255. * 100 - 50
        img_l_rs = (img_l_rs.astype(dtype=np.float32)) / 255.0 * 100 - 50

        img_l_batch.append(img_l)
        img_l_rs_batch.append(img_l_rs)
    img_l_batch = np.asarray(img_l_batch)
    img_l_rs_batch = np.asarray(img_l_rs_batch)

    img_313_rs_batch = sess.run(model, feed_dict={input_tensor: img_l_rs_batch})
    for i in xrange(_BATCH_SIZE):
        img_rgb, _ = decode(img_l_batch[i: i + 1], img_313_rs_batch[i: i + 1], T)
        img_name = os.path.split(img_paths[i])[1]
        imsave(os.path.join(OUTPUT_DIR, img_name), img_rgb)


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
    _validation_data(_to_gray)
    _training_data(_to_gray)


if __name__ == "__main__":
    main()
