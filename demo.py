import subprocess

import pickle
import tensorflow as tf
from utils import *
from net import Net
from skimage.io import imsave
from skimage import color, transform
import cv2

INPUT_SIZE = 224
_RESIZE_SIZE = 0
_CIFAR_IMG_SIZE = 32
_CIFAR_BATCH_SIZE = 20
_CIFAR_COUNT = 0
_CKPT_PATH = '/srv/glusterfs/xieya/colorization-test-34/models/model.ckpt-15000'
IMG_DIR = '/srv/glusterfs/xieya/image/grayscale/colorization_test'
OUTPUT_DIR = '/srv/glusterfs/xieya/image/color/colorization_test'
_IMG_NAME = '/srv/glusterfs/xieya/image/grayscale/cow_gray.jpg'
#T = 2.63
T = 2.63


def _resize(image):
    h = image.shape[0]
    w = image.shape[1]
    resize_size = min(h, w)

    if w > h:
      image = cv2.resize(image, (int(resize_size * w / h), resize_size))

      crop_start = np.random.randint(0, int(resize_size * w / h) - resize_size + 1)
      image = image[:, crop_start:crop_start + resize_size, :]
    else:
      image = cv2.resize(image, (resize_size, int(resize_size * h / w)))

      crop_start = np.random.randint(0, int(resize_size * h / w) - resize_size + 1)
      image = image[crop_start:crop_start + resize_size, :, :]
    return image


def _get_model(input_tensor):
    autocolor = Net(train=False)
    conv8_313 = autocolor.inference(input_tensor)
    return conv8_313


def _colorize_single_img(img_name, model, input_tensor, sess):
    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    img = _resize(img)
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

    img_l = (img_l.astype(dtype=np.float32)) / 255.0 * 100 - 50
    img_l_rs = (img_l_rs.astype(dtype=np.float32)) / 255.0 * 100 - 50
    img_313_rs = sess.run(model, feed_dict={input_tensor: img_l_rs})
    img_rgb, _ = decode(img_l_rs, img_313_rs, T)
    imsave(os.path.join(OUTPUT_DIR, img_name), img_rgb)


def _get_cifar_data(training=True):
    data = []
    if training:
        for i in range(1, 6):
            filename = '{}/data_batch_{}'.format(IMG_DIR, i)
            batch_data = pickle.load(open(filename, 'rb'))
            if len(data) > 0:
                data = np.vstack((data, batch_data[b'data']))
            else:
                data = batch_data[b'data']

    else:
        filename = '{}/test_batch'.format(IMG_DIR)
        batch_data = unpickle(filename)
        data = batch_data[b'data']

    w = 32
    h = 32
    s = w * h
    data = np.array(data)
    data = np.dstack((data[:, :s], data[:, s:2 * s], data[:, 2 * s:]))
    data = data.reshape((-1, w, h, 3))
    print('Cifar data size: {}'.format(data.shape))

    return data


def _colorize_cifar_batch(img_batch, model, input_tensor, sess):
    global _CIFAR_COUNT

    img_lab_batch = color.rgb2lab(img_batch)
    img_l_batch = img_lab_batch[:, :, :, 0:1]
    img_l_batch = img_l_batch - 50.

    # Upscale.
    img_batch_rs = map(
        lambda x: transform.resize(x, (INPUT_SIZE, INPUT_SIZE)), img_batch)
    img_lab_batch_rs = color.rgb2lab(img_batch_rs)
    img_l_batch_rs = img_lab_batch_rs[:, :, :, 0:1]
    img_l_batch_rs = img_l_batch_rs - 50.

    img_313_batch_rs = sess.run(
        model, feed_dict={input_tensor: img_l_batch_rs})

    for i in range(_CIFAR_BATCH_SIZE):
        img_313_rs = img_313_batch_rs[i]
        img_313_rs = img_313_rs[None, :, :, :]
        img_l = img_l_batch[i]
        img_l = img_l[None, :, :, :]
        img_rgb, _ = decode(img_l, img_313_rs, T)
        imsave(
            os.path.join(OUTPUT_DIR, str(_CIFAR_COUNT).zfill(5) + '.jpg'),
            img_rgb)
        _CIFAR_COUNT += 1
    print('Progress: {}'.format(_CIFAR_COUNT))


def cifar():
    cifar_data = _get_cifar_data(True)  # True for training.
    cifar_data_size = cifar_data.shape[0]

    input_tensor = tf.placeholder(
        tf.float32, shape=(_CIFAR_BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 1))
    model = _get_model(input_tensor)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, _CKPT_PATH)

        for i in range(cifar_data_size / _CIFAR_BATCH_SIZE):
            cifar_batch = cifar_data[i * _CIFAR_BATCH_SIZE: (i + 1) * _CIFAR_BATCH_SIZE, :, :, :]
            _colorize_cifar_batch(cifar_batch, model, input_tensor, sess)


def _colorize_high_res_img(img_name):
    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    if len(img.shape) == 3:
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_l = img_l[None, :, :, None]
    else:
        img_l = img[None, :, :, None]

        img_l = (img_l.astype(dtype=np.float32)) / 255.0 * 100 - 50
        autocolor = Net(train=False)
        conv8_313 = autocolor.inference(img_l)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, _CKPT_PATH)
            img_313 = sess.run(conv8_313)

        img_rgb, _ = decode(img_l, img_313, T)
        imsave(os.path.join(OUTPUT_DIR, img_name), img_rgb)
    

def main():
    input_tensor = tf.placeholder(
        tf.float32, shape=(1, INPUT_SIZE, INPUT_SIZE, 1))
    model = _get_model(input_tensor)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, _CKPT_PATH)
        for img_name in os.listdir(IMG_DIR):
            if img_name.endswith('.jpg') or img_name.endswith('.JPEG'):
                print(img_name)
                _colorize_single_img(img_name, model, input_tensor, sess)


if __name__ == "__main__":
    subprocess.check_call(['mkdir', '-p', OUTPUT_DIR])
    main()
    # _colorize_high_res_img(_IMG_NAME)
    # cifar()
