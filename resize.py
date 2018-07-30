import os
import cv2
from skimage.io import imsave

IMG_SIZE = 256
IMG_NAME = 'high_gray.jpg'

def _resize(img):
    h = img.shape[0]
    w = img.shape[1]

    if w > h:
        img = cv2.resize(img, (int(IMG_SIZE * w / h), IMG_SIZE))
    else:
        img = cv2.resize(img, (IMG_SIZE, int(IMG_SIZE * h / w)))

    return img

img = cv2.imread(IMG_NAME)
img = _resize(img)
img_base= os.path.splitext(IMG_NAME)[0]
imsave('{0}_{1}.jpg'.format(img_base, IMG_SIZE), img)