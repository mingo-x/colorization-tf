import os

import numpy as np
import tensorflow as tf
from net import Net
from skimage.io import imsave
from skimage.transform import resize
import cv2
import utils

IMG_SIZE = 224
IMG_DIR = ''
OUT_DIR = ''
LABEL_PATH = ''
VGG16 = tf.keras.applications.vgg16.VGG16()


def _predict_single_image(img_path, model, input_tensor, sess):
    img_name = os.path.split(img_path)[1]
    img = cv2.imread(img_path)
    img = img[None, :, :, None]
    data_l, _, _ = utils.preprocess(img)
    prediction = sess.run(model, feed_dict={input_tensor: data_l})
    img_rgb = utils.decode(data_l, prediction, 2.63)
    imsave(os.path.join(OUT_DIR, img_name), img_rgb)
    return img_rgb


def _get_model():
    input_tensor = tf.placeholder(tf.float32, (1, IMG_SIZE, IMG_SIZE, 1))
    autocolor = Net(train=False)
    conv8_313 = autocolor.inference(input_tensor)

    return conv8_313, input_tensor


def _vgg_loss(img, label):
    img = img[np.newaxis, :, :, :]
    img = tf.keras.applications.vgg16.preprocess_input(img)
    classification = VGG16.predict(img)
    accuracy = tf.keras.metrics.top_k_categorical_accuracy(label, classification, k=1)
    return tf.reduce_mean(accuracy)


def main():
    img_list = os.listdir(IMG_DIR).sort()
    saver = tf.train.Saver()
    model, input_tensor = _get_model()
    label_tensor = tf.placeholder(tf.float32, (1))
    colored_img_tensor = tf.placeholder(tf)
    
    vgg16_loss = _vgg_loss()
    with tf.Session() as sess, open(LABEL_PATH, 'r') as label_file:
        saver.restore(sess, 'models/model.ckpt')
        for img_path in img_list:
            img_label = int(label_file.readline())

            img_rgb = _predict_single_image(img_path, model, input_tensor, sess)


img_rgb = decode(data_l, conv8_313, 2.63)
imsave('color.jpg', img_rgb)
