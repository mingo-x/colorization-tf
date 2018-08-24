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

IMG_SIZE = 224
IMG_DIR = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val'
OUT_DIR = '/srv/glusterfs/xieya/colorization-tf/prediction'
LABEL_PATH = '/home/xieya/colorization-tf/resources/val.txt'
# LOG_PATH = '/home/xieya/metrics.txt'
MODEL_CHECKPOINT = '/srv/glusterfs/xieya/colorization-gan/models/model.ckpt-499000'
# CLASS_ID_DICT_PATH = '/srv/glusterfs/xieya/colorization-tf/resources/class_index_dict.pkl'
NUM_IMGS = 10000
# CLASS_ID_DICT = pickle.load(open(CLASS_ID_DICT_PATH, 'rb'))
THRESHOLD = 150
_BATCH_SIZE = 40


def _predict_single_image(img_name, model, input_tensor, sess):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    img = _image_process(img)
    img = [img]
    img = np.asarray(img, dtype=np.uint8)
    data_l, data_ab = utils.preprocess(img, training=False)
    prediction = sess.run(model, feed_dict={input_tensor: data_l})
    # prior = utils._prior_boost(gt_ab_313, gamma=0.)
    prior = utils.get_prior(data_ab)
    prior = prior[0, :, :, 0]
    # prior = resize(prior, (IMG_SIZE, IMG_SIZE))
    # print(np.min(prior), np.max(prior), np.mean(prior))
    img_rgb, img_ab = utils.decode(data_l, prediction, 2.63)

    # data_l = data_l[0, :, :, :] + 50
    # gray_ab= np.zeros((data_l.shape[0], data_l.shape[1], 2))
    # img_gray = np.concatenate((data_l, gray_ab), axis=-1)
    # img_gray = color.lab2rgb(img_gray)

    imsave(os.path.join(OUT_DIR, img_name), img_rgb)
    return img_ab, data_ab[0, :, :, :], prior


def _predict_image_batch(img_batch, model, input_tensor, sess):
    data_l_batch, data_ab_batch = utils.preprocess(img_batch, training=False)
    predicted_313_batch = sess.run(
        model, feed_dict={input_tensor: data_l_batch})
    # prior = utils._prior_boost(gt_ab_313, gamma=0.)
    prior_batch = utils.get_prior(data_ab_batch)
    prior_batch = prior_batch[:, :, :, 0]
    # prior = resize(prior, (IMG_SIZE, IMG_SIZE))
    # print(np.min(prior), np.max(prior), np.mean(prior))
    predicted_rgb_batch = []
    predicted_ab_batch = []
    for i in range(_BATCH_SIZE):
        predicted_rgb, predicted_ab = utils.decode(
            data_l_batch[i: i + 1], predicted_313_batch[i: i + 1], 2.63)
        predicted_rgb_batch.append(predicted_rgb)
        predicted_ab_batch.append(predicted_ab)
    predicted_rgb_batch = np.asarray(predicted_rgb_batch)
    predicted_ab_batch = np.asarray(predicted_ab_batch)

    return predicted_rgb_batch, predicted_ab_batch, data_ab_batch, prior_batch


def _get_model():
    input_tensor = tf.placeholder(
        tf.float32, (_BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1))
    autocolor = Net(train=False)
    conv8_313 = autocolor.inference(input_tensor)

    return conv8_313, input_tensor


def _l2_loss(real_batch, pred_batch, prior=None):
    l2_dist = np.sqrt(np.sum(np.square(real_batch - pred_batch), axis=3))
    if prior is None:
        ones = np.ones_like(l2_dist)
    else:
        ones = prior
    ones_sum = np.sum(ones)
    zeros = np.zeros_like(l2_dist)
    scores = []
    for thr in range(0, THRESHOLD + 1):
        score = np.sum(
            np.where(np.less_equal(l2_dist, thr), ones, zeros)) / ones_sum
        scores.append(score)
    return scores


def _vgg_loss(img_batch, label_batch, model):
    img_batch = tf.keras.applications.vgg16.preprocess_input(img_batch)
    prediction_batch = model.predict(img_batch)
    prediction_batch = np.argmax(prediction_batch, axis=-1)
    avg_loss = np.mean(
        np.where(np.equal(prediction_batch, label_batch), 1., 0.))
    return avg_loss


def _image_process(image):
    h = image.shape[0]
    w = image.shape[1]

    # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
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


def _batch_process(img_name_list):
    img_batch = []
    for img_name in img_name_list:
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        img = _image_process(img)  # Resized image.
        img_batch.append(img)

    img_batch = np.asarray(img_batch, dtype=np.uint8)
    return img_batch


def _save_batch(predicted_rgb_batch, img_name_batch, save=True):
    if save:
        for i in range(_BATCH_SIZE):
            imsave(
                os.path.join(
                    OUT_DIR, img_name_batch[i]), predicted_rgb_batch[i])


def main():
    img_list = os.listdir(IMG_DIR)
    img_list.sort()
    img_list = filter(lambda img_name: img_name.endswith('.JPEG'), img_list)

    model, input_tensor = _get_model()
    print("Model got.")
    saver = tf.train.Saver()

    vgg16_losses = []
    l2_losses = []
    l2_losses_re = []
    prior_sums = []
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess, \
            open(LABEL_PATH, 'r') as label_file:
        saver.restore(sess, MODEL_CHECKPOINT)
        print('Checkpoint restored.')

        vgg16 = tf.keras.applications.vgg16.VGG16()
        img_idx = 0

        while img_idx < 2 * NUM_IMGS:
            # Prepare image batch.
            img_name_batch = img_list[img_idx: img_idx + _BATCH_SIZE]
            img_idx += _BATCH_SIZE

            # Prepare img lables.
            img_label_batch = []
            for i in range(_BATCH_SIZE):
                img_label = int(label_file.readline().split(' ')[1])
                img_label_batch.append(img_label)
            img_label_batch = np.asarray(img_label_batch)

            if img_idx <= NUM_IMGS:
                continue

            for img_name in img_name_batch:
                print(img_name)

            # Preprocess image batch.
            img_batch = _batch_process(img_name_batch)
            predicted_rgb_batch, predicted_ab_batch, data_ab_batch, prior_batch = _predict_image_batch(
                img_batch, model, input_tensor, sess)
            _save_batch(predicted_rgb_batch, img_name_batch, False)
            # img_ab, data_ab, prior = _predict_single_image(img_name, model, input_tensor, sess)

            # img_rgb = tf.keras.preprocessing.image.load_img(os.path.join(OUT_DIR, img_name), target_size=(224, 224))
            # img_rgb = tf.keras.preprocessing.image.img_to_array(img_rgb)
            # img_rgb = img_rgb.reshape((1, img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]))
            predicted_rgb_batch *= 255.  # [0, 1] -> [0, 255]
            vgg16_loss = _vgg_loss(predicted_rgb_batch, img_label_batch, vgg16)
            vgg16_losses.append(vgg16_loss)

            l2_loss = _l2_loss(data_ab_batch, predicted_ab_batch)
            l2_losses.append(l2_loss)
            l2_loss_re = _l2_loss(
                data_ab_batch, predicted_ab_batch, prior=prior_batch)
            l2_losses_re.append(l2_loss_re)
            prior_sums.append(np.sum(prior_batch))

    # print("Prior mean, {}".format(np.mean(prior_sums) / (IMG_SIZE * IMG_SIZE)))
    vgg16_acc = np.mean(vgg16_losses)
    print("VGG16 acc, {}".format(vgg16_acc))
    l2_accs = np.mean(l2_losses, axis=0)
    l2_accs_re = np.average(l2_losses_re, axis=0, weights=prior_sums)
    l2_accs_re_1 = np.average(l2_losses_re, axis=0)
    x = [i for i in range(0, THRESHOLD + 1)]
    auc_score = auc(x, l2_accs)
    auc_score_re = auc(x, l2_accs_re)
    auc_score_re_1 = auc(x, l2_accs_re_1)
    print("L2 auc, {0}, {1}, {2}, {3}, {4}, {5}".format(
        auc_score, auc_score / THRESHOLD,
        auc_score_re, auc_score_re / THRESHOLD,
        auc_score_re_1, auc_score_re_1 / THRESHOLD))
    for i in range(0, THRESHOLD + 1):
        print("L2 acc, {0}, {1}, {2}, {3}".format(i, l2_accs[i], l2_accs_re[i], l2_accs_re_1[i]))


if __name__ == "__main__":
    main()
