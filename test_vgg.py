import os

import numpy as np
import tensorflow as tf
from net import Net
from skimage.io import imsave
from skimage.transform import resize
from skimage import color
import cv2
import utils
from sklearn.metrics import auc

IMG_SIZE = 256
IMG_DIR = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val'
OUT_DIR = '/srv/glusterfs/xieya/colorization-tf/prediction'
LABEL_PATH = '/home/xieya/colorization-tf/resources/val.txt'
LOG_PATH = '/home/xieya/metrics.txt'
MODEL_CHECKPOINT = '/srv/glusterfs/xieya/colorization-tf/pretrained/color_model.ckpt'
#CLASS_ID_DICT_PATH = '/srv/glusterfs/xieya/colorization-tf/resources/class_index_dict.pkl'
NUM_IMGS = 10000
#CLASS_ID_DICT = pickle.load(open(CLASS_ID_DICT_PATH, 'rb'))
THRESHOLD = 50


def _predict_single_image(img_name, model, input_tensor, sess):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    img= _image_process(img)
    img = [img]
    img = np.asarray(img, dtype=np.uint8)
    _, data_ab = utils.preprocess(img, training=False)
    prior = utils.get_prior(data_ab)
    prior = prior[0, :, :, 0]
    return prior


def _get_model():
    input_tensor = tf.placeholder(tf.float32, (1, IMG_SIZE, IMG_SIZE, 1))
    autocolor = Net(train=False)
    conv8_313 = autocolor.inference(input_tensor)

    return conv8_313, input_tensor


def _image_process(image):
    h = image.shape[0]
    w = image.shape[1]

    if w > h:
      image = cv2.resize(image, (int(IMG_SIZE * w / h), IMG_SIZE))

      crop_start = (int(IMG_SIZE * w / h) - IMG_SIZE) / 2
      image = image[:, crop_start:crop_start + IMG_SIZE, :]
    else:
      image = cv2.resize(image, (IMG_SIZE, int(IMG_SIZE * h / w)))

      crop_start = (int(IMG_SIZE * h / w) - IMG_SIZE) / 2
      image = image[crop_start:crop_start + IMG_SIZE, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def main():
    img_list = os.listdir(IMG_DIR)
    img_list.sort()
    model, input_tensor = _get_model()
    print("Model got.")
    saver = tf.train.Saver()

    prior_means = []
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver.restore(sess, MODEL_CHECKPOINT)
        print('Checkpoint restored.')
        img_count = 0
        for img_name in img_list:
            if not img_name.endswith('.JPEG'):
                continue
            img_count += 1
            print(img_name)
            prior = _predict_single_image(img_name, model, input_tensor, sess)
            prior_mean = np.mean(prior)
            prior_means.append(prior_mean)
            print(prior_mean)

            if img_count >= NUM_IMGS:
                break
    
    print(np.mean(prior_means))
    

if __name__ == "__main__":
    main()
    
