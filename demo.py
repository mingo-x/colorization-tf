import tensorflow as tf
from utils import *
from net import Net
from skimage.io import imsave
import cv2

INPUT_SIZE = 224
IMG_DIR = '/srv/glusterfs/xieya/video/dog_frames/'
OUTPUT_DIR = '/srv/glusterfs/xieya/video/dog_frames_color/'

def _resize(img):
    h = img.shape[0]
    w = img.shape[1]

    if w > h:
        img = cv2.resize(img, (int(IMG_SIZE * w / h), IMG_SIZE))
    else:
        img = cv2.resize(img, (IMG_SIZE, int(IMG_SIZE * h / w)))

    return img


def _get_model(input_tensor):
    autocolor = Net(train=False)
    conv8_313 = autocolor.inference(input_tensor)
    return conv8_313


def _colorize_single_img(img_name, model, input_tensor, sess):
    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    img_rs = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    if len(img.shape) == 3:
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_l = img_l[None, :, :, None]
        img_l_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
        img_l_rs = img_l_rs[None, :, :, None]
    else:
        img_l = img[None, :, :, None]
        img_l_rs = img_rs[None, :, :, None]

    # img = _resize(img)
    
    img_l_rs = (img_l_rs.astype(dtype=np.float32)) / 255.0 * 100 - 50
    img_313_rs = sess.run(model,  feed_dict={input_tensor: img_l_rs})
    img_rgb, _ = decode(img_l, img_313_rs, 2.63)
    imsave(os.path.join(OUTPUT_DIR, img_name), img_rgb)


def main():
    input_tensor = tf.placeholder(tf.float32, shape=(1, INPUT_SIZE, INPUT_SIZE, 1))
    model = _get_model(input_tensor)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, '/srv/glusterfs/xieya/colorization-tf/models/model.ckpt-499000')
        for img_name in os.listdir(IMG_DIR):
            print(img_name)
            _colorize_single_img(img_name, model, input_tensor, sess)


    
