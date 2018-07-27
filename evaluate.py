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
NUM_IMGS = 100
#CLASS_ID_DICT = pickle.load(open(CLASS_ID_DICT_PATH, 'rb'))


def _predict_single_image(img_name, model, input_tensor, sess):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    img= _image_process(img)
    img = [img]
    img = np.asarray(img, dtype=np.uint8)
    data_l, data_ab, prior = utils.preprocess(img, training=False, )
    prediction = sess.run(model, feed_dict={input_tensor: data_l})
    # prior = utils._prior_boost(gt_ab_313, gamma=0.)
    prior = prior[0, :, :, 0]
    prior = resize(prior, (IMG_SIZE, IMG_SIZE))
    img_rgb, img_ab = utils.decode(data_l, prediction, 2.63)

    # data_l = data_l[0, :, :, :] + 50
    # gray_ab= np.zeros((data_l.shape[0], data_l.shape[1], 2))
    # img_gray = np.concatenate((data_l, gray_ab), axis=-1)
    # img_gray = color.lab2rgb(img_gray)

    imsave(os.path.join(OUT_DIR, img_name), img_rgb)
    return img_ab, data_ab[0, :, :, :], prior


def _get_model():
    input_tensor = tf.placeholder(tf.float32, (1, IMG_SIZE, IMG_SIZE, 1))
    autocolor = Net(train=False)
    conv8_313 = autocolor.inference(input_tensor)

    return conv8_313, input_tensor


def _l2_loss(img_true, img_pred, prior=None):
    # print(img_true.shape, img_pred.shape)
    l2_dist = np.sqrt(np.sum(np.square(img_true - img_pred), axis=2))
    ones = np.ones_like(l2_dist)
    zeros = np.zeros_like(l2_dist)
    scores = []
    for thr in range(0, 151):
        score = np.average(np.where(np.less_equal(l2_dist, thr), ones, zeros), weights=prior)
        scores.append(score)
    return scores


def _vgg_loss(img, label, model):
    img = tf.keras.applications.vgg16.preprocess_input(img)
    prediction = model.predict(img)
    # decoded_prediction = tf.keras.applications.vgg16.decode_predictions(prediction, top=1)[0][0]
    # print(decoded_prediction)
    prediction = prediction[0]
    prediction = np.argmax(prediction)
    # print(prediction)
    return 1. if prediction == label else 0.


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

    vgg16_losses = []
    l2_losses = []
    l2_losses_re = []
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
            img_label = int(label_file.readline().split(' ')[1])
            img_ab, data_ab, prior = _predict_single_image(img_name, model, input_tensor, sess)
            img_rgb = tf.keras.preprocessing.image.load_img(os.path.join(OUT_DIR, img_name), target_size=(224, 224))
            img_rgb = tf.keras.preprocessing.image.img_to_array(img_rgb)
            img_rgb = img_rgb.reshape((1, img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]))
            vgg16_loss = _vgg_loss(img_rgb, img_label, vgg16)
            vgg16_losses.append(vgg16_loss)
            l2_loss = _l2_loss(data_ab, img_ab)
            l2_losses.append(l2_loss)
            l2_loss_re = _l2_loss(data_ab, img_ab, prior=prior)
            l2_losses_re.append(l2_loss_re)

            if img_count == NUM_IMGS:
                break

    vgg16_acc = np.mean(vgg16_losses)
    print("VGG16 acc, {}".format(vgg16_acc))
    l2_accs = np.mean(l2_losses, axis=0)
    l2_accs_re = np.mean(l2_losses_re, axis=0)
    x = [i for i in range(0, 151)]
    auc_score = auc(x, l2_accs)
    auc_score_re = auc(x, l2_accs_re)
    print("L2 auc, {0}, {1}, {2}, {3}".format(auc_score, auc_score / 150., auc_score_re, auc_score_re / 150.))
    for i in range(0, 151):
        print("L2 acc, {0}, {1}, {2}".format(i, l2_accs[i], l2_accs_re[i]))
    

if __name__ == "__main__":
    main()
    
