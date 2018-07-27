import os

import numpy as np
import tensorflow as tf
from net import Net
from skimage.io import imsave
import cv2
import utils

IMG_SIZE = 224
IMG_DIR = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val'
OUT_DIR = '/srv/glusterfs/xieya/colorization-tf/prediction'
LABEL_PATH = '/home/xieya/colorization-tf/resources/ILSVRC2012_validation_ground_truth.txt'
LOG_PATH = '/home/xieya/metrics.txt'
MODEL_CHECKPOINT = '/srv/glusterfs/xieya/colorization-tf/pretrained/color_model.ckpt'
NUM_IMGS = 100


def _predict_single_image(img_name, model, input_tensor, sess):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    img = _image_process(img)
    img = np.asarray(img, dtype=np.uint8)
    # img_true = img
    img = img[np.newaxis, :, :, :]
    data_l, data_ab = utils.preprocess(img, training=False)
    prediction = sess.run(model, feed_dict={input_tensor: data_l})
    img_rgb, img_ab = utils.decode(data_l, prediction, 2.63)
    imsave(os.path.join(OUT_DIR, img_name), img_rgb)
    return img_rgb, img_ab, data_ab


def _get_model():
    input_tensor = tf.placeholder(tf.float32, (1, IMG_SIZE, IMG_SIZE, 1))
    autocolor = Net(train=False)
    conv8_313 = autocolor.inference(input_tensor)

    return conv8_313, input_tensor


def _l2_loss(img_true, img_pred):
    print(img_true.shape, img_pred.shape)
    l2_dist = np.sqrt(np.sum(np.square(img_true - img_pred), axis=2))
    ones = np.ones_like(l2_dist)
    zeros = np.zeros_like(l2_dist)
    scores = []
    for thr in range(0, 151):
        score = np.mean(np.where(np.less_equal(l2_dist, thr), ones, zeros))
        scores.append(score)
    return scores


def _vgg_loss(img, label, model):
    img = img[np.newaxis, :, :, :]
    img = tf.keras.applications.vgg16.preprocess_input(img)
    prediction = model.predict(img)
    class_name = tf.keras.applications.vgg16.decode_predictions(prediction, top=1)[0][0][1]
    print(class_name)
    prediction = prediction[0]
    prediction = np.argmax(prediction)
    return float(prediction == label)


def _image_process(image):
    h = image.shape[0]
    w = image.shape[1]

    if w > h:
      image = cv2.resize(image, (int(IMG_SIZE * w / h), IMG_SIZE))

      mirror = np.random.randint(0, 2)
      if mirror:
        image = np.fliplr(image)
      crop_start = np.random.randint(0, int(IMG_SIZE * w / h) - IMG_SIZE + 1)
      image = image[:, crop_start:crop_start + IMG_SIZE, :]
    else:
      image = cv2.resize(image, (IMG_SIZE, int(IMG_SIZE * h / w)))
      mirror = np.random.randint(0, 2)
      if mirror:
        image = np.fliplr(image)
      crop_start = np.random.randint(0, int(IMG_SIZE * h / w) - IMG_SIZE + 1)
      image = image[crop_start:crop_start + IMG_SIZE, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def main():
    img_list = os.listdir(IMG_DIR)
    img_list.sort()
    model, input_tensor = _get_model()
    print("Model got.")
    saver = tf.train.Saver()

    vgg16_losses = []
    l2_losses = []
    img_count = 0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess, open(LABEL_PATH, 'r') as label_file:
        saver.restore(sess, MODEL_CHECKPOINT)
        print('Checkpoint restored.')
        vgg16 = tf.keras.applications.vgg16.VGG16()
        for img_name in img_list:
            if not img_name.endswith('.JPEG'):
                continue
            print(img_name)
            img_count += 1
            img_label = int(label_file.readline())
            img_rgb, img_ab, data_ab = _predict_single_image(img_name, model, input_tensor, sess)
            vgg16_loss = _vgg_loss(img_rgb, img_label, vgg16)
            vgg16_losses.append(vgg16_loss)
            l2_loss = _l2_loss(data_ab, img_ab)
            l2_losses.append(l2_loss)

            if img_count == NUM_IMGS:
                break

    vgg16_acc = np.mean(vgg16_losses)
    print("VGG16 acc", ",", vgg16_acc)
    l2_accs = np.mean(l2_losses, axis=0)
    for i in range(0, 151):
        print("L2 acc", ",", i, ",", l2_accs[i])
    

if __name__ == "__main__":
    main()
    
