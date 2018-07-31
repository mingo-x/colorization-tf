import tensorflow as tf
from utils import *
from net import Net
from skimage.io import imsave
import cv2

IMG_SIZE = 256

def _resize(img):
    h = img.shape[0]
    w = img.shape[1]

    if w > h:
        img = cv2.resize(img, (int(IMG_SIZE * w / h), IMG_SIZE))
    else:
        img = cv2.resize(img, (IMG_SIZE, int(IMG_SIZE * h / w)))

    return img

img = cv2.imread('high_gray.jpg')
if len(img.shape) == 3:
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = _resize(img)

img = img[None, :, :, None]
data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

#data_l = tf.placeholder(tf.float32, shape=(None, None, None, 1))
autocolor = Net(train=False)

conv8_313 = autocolor.inference(data_l)

saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, '/srv/glusterfs/xieya/colorization-tf/models/model.ckpt-499000')
  conv8_313 = sess.run(conv8_313)

img_rgb, _ = decode(data_l, conv8_313, 2.63)
imsave('color.jpg', img_rgb)
