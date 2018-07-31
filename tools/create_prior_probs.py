import numpy as np
from skimage.io import imread
from skimage import color
from skimage.transform import resize
from multiprocessing import Pool


_NUM_PROCESSES = 2
_IMG_PATHS = 'data/train.txt'
_POINTS_PATH = 'resources/pts_in_hull.npy'
_PRINT_FREQ = 100


def _get_img_list():
  img_list = []  
  with open(_IMG_PATHS, 'r') as fin:
    for img_path in fin:
      img_path = img_path.strip()
      img_list.append(img_path)
  return img_list


def _get_index(in_data, points):
  '''
  Args:
    in_data: [None, 2]
    points:
  '''
  expand_in_data = in_data[:, np.newaxis, :]
  distance = np.sum(np.square(expand_in_data - points), axis=2)
  index = np.argmin(distance, axis=1)
  return index


def _calculate_prior(img_path, points):
  probs = np.zeros((_NUM_PROCESSES, 313), dtype=np.float64)
  img = imread(img_path)
  img = resize(img, (224, 224), preserve_range=True)
  if len(img.shape)!=3 or img.shape[2]!=3:
    return probs
  img_lab = color.rgb2lab(img)
  img_lab = img_lab.reshape((-1, 3))
  img_ab = img_lab[:, 1:]
  nd_index = _get_index(img_ab, points)
  for i in nd_index:
    probs[i] += 1
   
  probs = probs / np.sum(probs)
  return probs


def main():
  points = np.load(_POINTS_PATH)
  points = points.astype(np.float64)
  points = points[None, :, :]
  
  img_list = _get_img_list()
  probs = np.zeros((_NUM_PROCESSES, 313), dtype=np.float64)
  pool = Pool(processes=_NUM_PROCESSES)   
  calculate_prior = lambda x: _calculate_prior(x, points)
  img_count = 0

  for prior in pool.imap_unordered(calculate_prior, img_list):
    probs += prior
    img_count += 1
    if img_count % _PRINT_FREQ == 0:
      print(img_count)

  probs = probs / np.sum(probs)
  np.save('probs', probs)


if __name__ == "__main__":
  main()
