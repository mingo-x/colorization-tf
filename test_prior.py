import cv2
import numpy as np
from skimage.transform import resize

import utils

_IMG_PATH = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val/ILSVRC2012_val_00050000.JPEG'
img = cv2.imread(_IMG_PATH)
img = resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = [img]
img = np.asarray(img, dtype=np.uint8)
_, data_ab = utils.preprocess(img, training=False)
prior = utils.get_prior(data_ab)
prior = prior[0, :, :, 0]
print(prior)
e = np.sum(prior) / (prior.shape[0] * prior.shape[1])
print(e, prior.shape)