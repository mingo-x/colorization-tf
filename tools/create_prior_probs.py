#!/srv/glusterfs/xieya/anaconda2/bin/python

# ----- Parameters passed to the cluster ------
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queu

#$ -t 1:2

#$ -S /srv/glusterfs/xieya/anaconda2/bin/python

#$ -l h_rt=0:59:59

#$ -l h_vmem=4G

#$ -o /srv/glusterfs/xieya/log

#$ -e /srv/glusterfs/xieya/log

#$ -j y


import os
# import functools
# import monotonic
'''
import numpy as np
from skimage.io import imread
from skimage import color
from skimage.transform import resize
from multiprocessing import Pool
'''

_NUM_TASKS = 100
_IMG_PATHS = 'data/train.txt'
_POINTS_PATH = 'resources/pts_in_hull.npy'
_PRINT_FREQ = 10
_TASK_ID = int(os.environ.get('SGE_TASK_ID')) - 1
print _TASK_ID


# def _get_img_list():
#   img_list = []  
#   img_count = 0
#   with open(_IMG_PATHS, 'r') as fin:
#     print(_IMG_PATHS)
#     for img_path in fin:
#       if img_count % _NUM_TASKS == _TASK_ID:
#         img_path = img_path.strip()
#         img_list.append(img_path)
#       img_count += 1
#   return img_list


# def _get_index(in_data, points):
#   '''
#   Args:
#     in_data: [None, 2]
#     points:
#   '''
#   expand_in_data = in_data[:, np.newaxis, :]
#   distance = np.sum(np.square(expand_in_data - points), axis=2)
#   index = np.argmin(distance, axis=1)
#   return index


# def _calculate_prior(img_path, points, probs):
#   img = imread(img_path)
#   img = resize(img, (224, 224), preserve_range=True)
#   if len(img.shape)!=3 or img.shape[2]!=3:
#     return probs
#   img_lab = color.rgb2lab(img)
#   img_lab = img_lab.reshape((-1, 3))
#   img_ab = img_lab[:, 1:]
#   nd_index = _get_index(img_ab, points)
#   for i in nd_index:
#     probs[i] += 1


# def main():
#   points = np.load(_POINTS_PATH)
#   print(_POINTS_PATH)
#   points = points.astype(np.float64)
#   points = points[None, :, :]
  
#   img_list = _get_img_list()
#   probs = np.zeros((313), dtype=np.float64)   
#   img_count = 0
#   start_time = monotonic.monotonic()

#   for img_path in img_list:
#     _calculate_prior(img_path, points, probs)
#     img_count += 1
#     if img_count % _PRINT_FREQ == 0:
#       print(img_count, monotonic.monotonic()-start_time)
#       start_time = monotonic.monotonic()

#   probs = probs / np.sum(probs)
#   np.save('probs_{}'.format(_TASK_ID), probs)


# if __name__ == "__main__":
#   main()
