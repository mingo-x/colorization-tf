from glob import glob
import os
import random
import subprocess

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from net import Net
from skimage import io, color, transform
import cv2

from data import DataSet
from data_coco import DataSet as DataSetCOCO
import utils

_AUC_THRESHOLD = 150
_BATCH_SIZE = 1
_CAP_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7]
# _CAP_LAYERS = [6]
_COCO_PATH = '/srv/glusterfs/xieya/data/coco_colors.h5'
_INPUT_SIZE = 224
_RESIZE_SIZE = 0
_CIFAR_IMG_SIZE = 32
_CIFAR_BATCH_SIZE = 20
_CIFAR_COUNT = 0
_G_VERSION = 1
_CKPT_PATH = '/srv/glusterfs/xieya/language_2/models/model.ckpt-18000'
# IMG_DIR = '/srv/glusterfs/xieya/image/grayscale/colorization_test'
IMG_DIR = '/srv/glusterfs/xieya/data/coco_seg'
_JBU_K = 10
_OUTPUT_DIR = '/srv/glusterfs/xieya/image/color/language_2_new'
_PRIOR_PATH = '/srv/glusterfs/xieya/prior/coco_313_soft.npy'
#_PRIOR_PATH = 'resources/prior_probs_smoothed.npy'
_IMG_NAME = '/srv/glusterfs/xieya/image/grayscale/cow_gray.jpg'
_VIDEO_IN_DIR = '/srv/glusterfs/xieya/data/DAVIS/JPEGImages/Full-Resolution/bus'
_VIDEO_OUT_DIR = '/srv/glusterfs/xieya/video/bus/vgg_4'
_NEW_CAPTION = True
# T = 2.63
T = 2.63

    
def _resize(image, resize_size=None):
    h = image.shape[0]
    w = image.shape[1]
    if resize_size is None:
        resize_size = min(h, w)

    if w > h:
        image = cv2.resize(image, (int(resize_size * w / h), resize_size))

        # crop_start = np.random.randint(0, int(resize_size * w / h) - resize_size + 1)
        crop_start = (int(resize_size * w / h) - resize_size + 1) / 2
        image = image[:, crop_start:crop_start + resize_size, :]
    else:
        image = cv2.resize(image, (resize_size, int(resize_size * h / w)))

        # crop_start = np.random.randint(0, int(resize_size * h / w) - resize_size + 1)
        crop_start = (int(resize_size * h / w) - resize_size + 1) / 2
        image = image[crop_start:crop_start + resize_size, :, :]
    return image


def _get_model(input_tensor):
    autocolor = Net(train=False, g_version=_G_VERSION)
    conv8_313 = autocolor.inference(input_tensor)
    #if len(conv8_313) == 2:
    #    conv8_313 = conv8_313[0]
    return conv8_313


def _b_dist(a, b):
    return -np.log(np.sum(np.sqrt(a * b)))


def _cosine(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))


def _cross_entropy(a, b):
    return np.sum(-a * np.log(b)), np.sum(-b * np.log(a))


def _kl_dist(a, b):
    return np.sum(a * np.log(a / b)), np.sum(b * np.log(b / a))


def compare_c313_pixelwise():
    input_tensor = tf.placeholder(
        tf.float32, shape=(1, _INPUT_SIZE, _INPUT_SIZE, 1))
    model = _get_model(input_tensor)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, _CKPT_PATH)

    img_name = 'ILSVRC2012_val_00049823.JPEG'  # 823, 815, 923
    img_prefix = os.path.splitext(img_name)[0]
    pos = [(10, 45), (45, 10), (26, 24)]

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    img = _resize(img)
    img_rs = cv2.resize(img, (_INPUT_SIZE, _INPUT_SIZE))
    if len(img.shape) == 3:
        img_l_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
        img_l_rs = img_l_rs[None, :, :, None]
    else:
        img_l_rs = img_rs[None, :, :, None]

    img_l_rs = (img_l_rs.astype(dtype=np.float32)) / 255.0 * 2 - 1
    img_l_rs_rs = transform.downscale_local_mean(img_l_rs, (1, 4, 4, 1))
    img_313_rs = sess.run(model, feed_dict={input_tensor: img_l_rs})
    img_rgb, _, c313_rb, c313 = utils.decode(img_l_rs_rs, img_313_rs, T, return_313=True)
    io.imsave(os.path.join(_OUTPUT_DIR, os.path.split(img_name)[1]), img_rgb)
    for p in pos[0: -1]:
        color = img_rgb[p]
        x = c313[p]
        plt.plot(x, c=color)
        # ab_tools.weights_to_image(x, '{0}_{1}_{2}'.format(img_prefix, p[0], p[1]), fill=0.5, out_dir=_OUTPUT_DIR)
    plt.savefig(os.path.join(_OUTPUT_DIR, '{0}.jpg'.format(img_prefix)))
    plt.clf()
    for p in pos[: -1]:
        color = img_rgb[p]
        x_rb = c313_rb[p]
        plt.plot(x_rb, c=color)
        # ab_tools.weights_to_image(x_rb, '{0}_{1}_{2}_rb'.format(img_prefix, p[0], p[1]), fill=0.5, out_dir=_OUTPUT_DIR)
    plt.savefig(os.path.join(_OUTPUT_DIR, '{0}_rb.jpg'.format(img_prefix)))

    for i in xrange(len(pos)):
        for j in xrange(i + 1, len(pos)):
            print('bhatta dist', i, j, _b_dist(c313[pos[i]], c313[pos[j]]), _b_dist(c313_rb[pos[i]], c313_rb[pos[j]]))
            print('cosine', i, j, _cosine(c313[pos[i]], c313[pos[j]]), _cosine(c313_rb[pos[i]], c313_rb[pos[j]]))
            print('cross entropy', i, j, _cross_entropy(c313[pos[i]], c313[pos[j]]), _cross_entropy(c313_rb[pos[i]], c313_rb[pos[j]]))
            print('kl', i, j, _kl_dist(c313[pos[i]], c313[pos[j]]), _kl_dist(c313_rb[pos[i]], c313_rb[pos[j]]))
            print('overlap', i, j, _intersection_of_hist(c313[pos[i]], c313[pos[j]]))


def _intersection_of_hist(a, b):
    return np.sum(np.minimum(a, b))


def _compare_c313_single_image(img_name, model, input_tensor, sess, r_num=5):
    # Randomly sample points
    pos = [(random.randint(0, 55), random.randint(0, 55)) for _ in xrange(r_num)]

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    img = _resize(img)
    img_rs = cv2.resize(img, (_INPUT_SIZE, _INPUT_SIZE))
    if len(img.shape) == 3:
        img_l_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
        img_l_rs = img_l_rs[None, :, :, None]
    else:
        img_l_rs = img_rs[None, :, :, None]

    img_l_rs = (img_l_rs.astype(dtype=np.float32)) / 255.0 * 2 - 1
    img_l_rs_rs = transform.downscale_local_mean(img_l_rs, (1, 4, 4, 1))
    img_313_rs = sess.run(model, feed_dict={input_tensor: img_l_rs})
    img_rgb, _, c313_rb, c313 = utils.decode(img_l_rs_rs, img_313_rs, T, return_313=True)
    io.imsave(os.path.join(_OUTPUT_DIR, os.path.split(img_name)[1]), img_rgb)

    scores, scores_rb = [], []
    for p in pos:
        # if random.random() < 0.5:
        #     q = (p[0] + 1 if p[0] < 55 else p[0] - 1, p[1])
        # else:
        #     q = (p[0], p[1] + 1 if p[1] < 55 else p[1] - 1)
        q = (random.randint(0, 55), random.randint(0, 55))
        cp, cq = c313[p], c313[q]
        scores.append(_intersection_of_hist(cp, cq))
        cp_rb, cq_rb = c313_rb[p], c313_rb[q]
        scores_rb.append(_intersection_of_hist(cp_rb, cq_rb))
    s = np.mean(scores)
    s_rb = np.mean(scores_rb)
    print(s, s_rb)
    return (s, s_rb)


def compare_c313():
    input_tensor = tf.placeholder(
        tf.float32, shape=(1, _INPUT_SIZE, _INPUT_SIZE, 1))
    model = _get_model(input_tensor)
    saver = tf.train.Saver()
    
    sess = tf.Session()
    saver.restore(sess, _CKPT_PATH)
    scores, scores_rb = [], []
    img_name_p = 'ILSVRC2012_val_0004{:04}.JPEG'
    for i in xrange(0, 10000):
        img_name = img_name_p.format(i)
        if os.path.exists(os.path.join(IMG_DIR, img_name)):
            print(img_name)
            s, s_rb = _compare_c313_single_image(img_name, model, input_tensor, sess)
            scores.append(s)
            scores_rb.append(s_rb)
    sess.close()

    score = np.mean(scores)
    score_rb = np.mean(scores_rb)

    print('average:', score, score_rb)
    r = np.random.normal(size=(313))
    r /= np.sum(r)
    print('Sanity check: ', _intersection_of_hist(r, r))


def _colorize_single_img(img_name, model, input_tensor, sess, jbu=False):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    img = _resize(img)
    img_rs = cv2.resize(img, (_INPUT_SIZE, _INPUT_SIZE))
    if len(img.shape) == 3:
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_l = img_l[None, :, :, None]
        img_l_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
        img_l_rs = img_l_rs[None, :, :, None]
    else:
        img_l = img[None, :, :, None]
        img_l_rs = img_rs[None, :, :, None]

    # img = _resize(img)
    # img_rgb_sk = io.imread(os.path.join("/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val", img_name))
    # if len(img_rgb_sk.shape) < 3 or img_rgb_sk.shape[2] != 3:
    #     return
    # img_rgb_sk = cv2.resize(img_rgb_sk, (_INPUT_SIZE, _INPUT_SIZE))
    # img_lab = color.rgb2lab(img_rgb_sk)
    # img_lab_rs = transform.downscale_local_mean(img_lab, (4, 4, 1))
    # img_lab_rs[:, :, 0] = 50
    # img_rgb_rs = color.lab2rgb(img_lab_rs)
    # io.imsave(os.path.join(_OUTPUT_DIR, "test_" + img_name), img_rgb_rs)

    img_l = (img_l.astype(dtype=np.float32)) / 255.0 * 2 - 1
    img_l_rs = (img_l_rs.astype(dtype=np.float32)) / 255.0 * 2 - 1
    img_313_rs = sess.run(model, feed_dict={input_tensor: img_l_rs})
    # img_l_rs_rs = np.zeros((1, 56, 56, 1))
    img_rgb, _ = utils.decode(img_l_rs, img_313_rs, T, jbu=jbu, jbu_k=_JBU_K)
    io.imsave(os.path.join(_OUTPUT_DIR, os.path.split(img_name)[1]), img_rgb)


def _reconstruct_single_img(img_name, jbu=False):
    img_path = os.path.join(IMG_DIR, img_name)
    img_id = os.path.splitext(img_name)[0]
    img_rgb = io.imread(img_path)
    if len(img_rgb.shape) != 3 or img_rgb.shape[2] != 3:
        return
    img_rgb = _resize(img_rgb, 224)
    img_lab = color.rgb2lab(img_rgb)
    img_l = img_lab[None, :, :, 0: 1]
    # img_rgb_rs = cv2.resize(img_rgb, (_INPUT_SIZE, _INPUT_SIZE))
    # img_lab_rs = color.rgb2lab(img_rgb_rs)
    # img_ab_rs = img_lab_rs[None, :, :, 1:]
    img_ab = img_lab[None, :, :, 1:]
    img_ab_ss = transform.downscale_local_mean(img_ab, (1, 4, 4, 1))
    gt_313 = utils._nnencode(img_ab_ss)

    img_l = (img_l.astype(dtype=np.float32)) / 50. - 1
    img_dec, _ = utils.decode(img_l, gt_313, T, sfm=False, jbu=jbu, jbu_k=_JBU_K)
    io.imsave(os.path.join(_OUTPUT_DIR, img_id + '{}.jpg'.format('_jbu' if jbu else '')), img_dec)


def _colorize_ab_canvas(model, input_tensor, sess):
    gt_canvas = np.zeros((8 * 64, 8 * 64, 3))
    pr_canvas = np.zeros((8 * 64, 8 * 64, 3))
    cnt = 0
    for img_name in os.listdir(IMG_DIR):
        if cnt >= 64:
            break
        if img_name.endswith('.jpg') or img_name.endswith('.JPEG'):
            img_rgb_sk = io.imread(os.path.join("/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val", img_name))
            if len(img_rgb_sk.shape) < 3 or img_rgb_sk.shape[2] != 3:
                continue
            i = cnt / 8
            j = cnt % 8
            cnt += 1

            img_rgb_sk = cv2.resize(img_rgb_sk, (256, 256))
            img_lab = color.rgb2lab(img_rgb_sk)
            img_lab_rs = transform.downscale_local_mean(img_lab, (4, 4, 1))
            img_lab_rs[:, :, 0] = 50
            img_rgb_rs = color.lab2rgb(img_lab_rs)
            gt_canvas[i * 64: (i + 1) * 64, j * 64: (j + 1) * 64, :] = img_rgb_rs
                
            img = cv2.imread(os.path.join(IMG_DIR, img_name))
            img_rs = cv2.resize(img, (256, 256))
            if len(img.shape) == 3:
                img_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_l = img_l[None, :, :, None]
                img_l_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
                img_l_rs = img_l_rs[None, :, :, None]
            else:
                img_l = img[None, :, :, None]
                img_l_rs = img_rs[None, :, :, None]

            img_l = (img_l.astype(dtype=np.float32)) / 255.0 * 2 - 1
            img_l_rs = (img_l_rs.astype(dtype=np.float32)) / 255.0 * 2 - 1
            img_313_rs = sess.run(model, feed_dict={input_tensor: img_l_rs})
            img_l_rs_rs = np.zeros((1, 64, 64, 1))
            img_rgb, _ = decode(img_l_rs_rs, img_313_rs, T)
            pr_canvas[i * 64: (i + 1) * 64, j * 64: (j + 1) * 64, :] = img_rgb

    io.imsave(os.path.join(_OUTPUT_DIR, "gt_ab.jpg"), gt_canvas)
    io.imsave(os.path.join(_OUTPUT_DIR, "pr_ab.jpg"), pr_canvas)


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
        lambda x: transform.resize(x, (_INPUT_SIZE, _INPUT_SIZE)), img_batch)
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
            os.path.join(_OUTPUT_DIR, str(_CIFAR_COUNT).zfill(5) + '.jpg'),
            img_rgb)
        _CIFAR_COUNT += 1
    print('Progress: {}'.format(_CIFAR_COUNT))


def cifar():
    cifar_data = _get_cifar_data(True)  # True for training.
    cifar_data_size = cifar_data.shape[0]

    input_tensor = tf.placeholder(
        tf.float32, shape=(_CIFAR_BATCH_SIZE, _INPUT_SIZE, _INPUT_SIZE, 1))
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
        imsave(os.path.join(_OUTPUT_DIR, img_name), img_rgb)
    

def main(jbu=False):
    input_tensor = tf.placeholder(
        tf.float32, shape=(1, _INPUT_SIZE, _INPUT_SIZE, 1))
    model = _get_model(input_tensor)
    saver = tf.train.Saver()
    
    sess = tf.Session()
    saver.restore(sess, _CKPT_PATH)
    # img_list = ['ILSVRC2012_val_00049923.JPEG']
    for img_name in os.listdir(IMG_DIR):
    # for img_name in img_list:
        if img_name.endswith('.jpg') or img_name.endswith('.JPEG'):
            print(img_name)
            _colorize_single_img(img_name, model, input_tensor, sess, jbu=jbu)
     
    sess.close()


def colorize_segcap(jbu=False):
    file_list = sorted(glob("/srv/glusterfs/xieya/data/coco_seg/images_224/val2017/*.jpg"))
    file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
    print('Before: {}'.format(len(file_list)))
    im2cap = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/im2cap_comb.p', 'rb'))
    file_list = list(filter(lambda x: x in im2cap, file_list))
    print('After: {}'.format(len(file_list)))

    input_tensor = tf.placeholder(
        tf.float32, shape=(1, _INPUT_SIZE, _INPUT_SIZE, 1))
    model = _get_model(input_tensor)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, _CKPT_PATH)

    for img_id in file_list:
        print(img_id)
        img_name = os.path.join("images_224/val2017", img_id + ".jpg")
        _colorize_single_img(img_name, model, input_tensor, sess, jbu=jbu)

    sess.close()


def reconstruct(jbu=False):
    for i in xrange(49800, 49850):
        img_name = 'ILSVRC2012_val_000{}.JPEG'.format(i)
        print(img_name)
        _reconstruct_single_img(img_name, jbu=jbu)


def demo_wgan_ab():
    noise = tf.constant(np.random.normal(size=(64, 128)).astype('float32'))
    model = Net(train=False)
    model.output_dim = 2
    colorized = model.GAN_G(noise)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, _CKPT_PATH)
        ab = sess.run(colorized)  # [-1, 1]
        ab *= 110.
        l = np.full((64, 64, 64, 1), 50)
        lab = np.concatenate((l, ab), axis=-1)
        rgbs = []
        for i in xrange(64):
            rgb = color.lab2rgb(lab[i, :, :, :])
            rgbs.append(rgb)
        rgbs = np.array(rgbs)
        save_images(rgbs, '/srv/glusterfs/xieya/image/color/samples_ab.png')


def demo_wgan_rgb():
    noise = tf.constant(np.random.normal(size=(64, 128)).astype('float32'))
    model = Net(train=False)
    model.output_dim = 3
    colorized = model.GAN_G(noise)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, _CKPT_PATH)
        rgb = sess.run(colorized) # [-1, 1]
        rgb = ((rgb+1.)*(255.99/2)).astype('uint8')
        save_images(rgb, '/srv/glusterfs/xieya/image/color/samples_rgb.png')
        rgb_new = []
        for i in xrange(64):
            lab = color.rgb2lab(rgb[i, :, :, :])
            lab[:, :, 0] = 50.  # Remove l.
            rgb_new.append(color.lab2rgb(lab))
        rgb_new = np.array(rgb_new)
        save_images(rgb_new, '/srv/glusterfs/xieya/image/color/samples_rgb_ab.png')
        

def places365():
    input_tensor = tf.placeholder(
        tf.float32, shape=(1, _INPUT_SIZE, _INPUT_SIZE, 1))
    model = _get_model(input_tensor)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, _CKPT_PATH)
        with open(os.path.join(IMG_DIR, 'val.txt'), 'r') as fin:
            img_name_list = []
            for img_name in fin:
                img_name_list.append(img_name[: -1])
            random.shuffle(img_name_list)

            for img_name in img_name_list[0: 200]:
                print(img_name)
                _colorize_single_img(img_name, model, input_tensor, sess)


def metrics(gt_ab_ss, pred_313, sess, gt_313_tensor, pred_313_tensor, prior_tensor, ce_loss_tensor, rb_loss_tensor):
    # gt_ab_ss = transform.downscale_local_mean(gt_ab, (1, 4, 4, 1))

    # NNEncoder
    # gt_ab_313: [N, H/4, W/4, 313]
    gt_313 = utils._nnencode(gt_ab_ss)

    # Prior_Boost 
    # prior_boost: [N, 1, H/4, W/4]
    prior_boost = utils._prior_boost(gt_313, gamma=0.5, prior_path=_PRIOR_PATH)

    ce_loss, rb_loss = sess.run([ce_loss_tensor, rb_loss_tensor], 
                                feed_dict={gt_313_tensor: gt_313, pred_313_tensor: pred_313, prior_tensor: prior_boost})

    return ce_loss, rb_loss


def cross_entropy_loss(gt_313, conv8_313, prior_boost_nongray):
    flat_conv8_313 = tf.reshape(conv8_313, [-1, 313])
    flat_gt_313 = tf.reshape(gt_313, [-1, 313])
    ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_conv8_313, labels=flat_gt_313)
    ce_loss = tf.reshape(ce_loss, tf.shape(prior_boost_nongray))
    rb_loss = ce_loss * prior_boost_nongray
    ce_loss = tf.reduce_mean(ce_loss, axis=(1, 2, 3))
    rb_loss = tf.reduce_sum(rb_loss, axis=(1, 2, 3)) / tf.reduce_sum(prior_boost_nongray, axis=(1, 2, 3))

    return ce_loss, rb_loss


def colorize_with_language(with_attention=False, concat=False, same_lstm=True, residual=False, paper=False, lstm_version=0, use_vg=False):
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    val_imgs = hf['val_ims']
    val_caps = hf['val_words']
    val_lens = hf['val_length']

    train_vocab_vg = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/glove.6B.100d_voc.p', 'r'))
    vrev_vg = dict((v, k) for (k, v) in train_vocab_vg.iteritems())
    train_vocab = pickle.load(open('/home/xieya/colorfromlanguage/priors/coco_colors_vocab.p', 'r'))
    vrev = dict((v, k) for (k, v) in train_vocab.iteritems())

    orig_dir = os.path.join(_OUTPUT_DIR, 'original')
    new_dir = os.path.join(_OUTPUT_DIR, 'new')
    if not os.path.exists(orig_dir):
        os.makedirs(orig_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    with tf.device('/gpu:0'):
        l_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE, _INPUT_SIZE, 1))
        cap_tensor = tf.placeholder(tf.int32, (1, 20))
        len_tensor = tf.placeholder(tf.int32, (1))
        autocolor = Net(train=False, use_vg=use_vg)
        if concat:
            c313_tensor = autocolor.inference5(l_tensor, cap_tensor, len_tensor, _CAP_LAYERS, same_lstm, residual, paper=paper, lstm_version=lstm_version)
        else:
            biases = [None] * 8
            for l in _CAP_LAYERS:
                biases[l] = 1.
            c313_tensor = autocolor.inference4(l_tensor, cap_tensor, len_tensor, biases, with_attention=with_attention)
        saver = tf.train.Saver()
        print("Saver created.")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver.restore(sess, _CKPT_PATH)

            try:
                idx = [335, 2632, 3735]
                for i in xrange(200, 400):
                    idx.append(i)

                for i in idx:
                    img_bgr = val_imgs[i]
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_lab = color.rgb2lab(img_rgb)
                    img_l = img_lab[None, :, :, 0: 1]
                    img_ab = img_lab[None, :, :, 1:]
                    img_ab_ss = transform.downscale_local_mean(img_ab, (1, 4, 4, 1))
                    if utils.is_grayscale(img_ab_ss):
                        continue
                    img_l = (img_l.astype(dtype=np.float32) - 50.) / 50.
                    img_cap = val_caps[i: i + 1]
                    img_len = val_lens[i: i + 1]
                    if use_vg:
                        for j in xrange(img_len[0]):
                            img_cap[0, j] = train_vocab_vg.get(vrev.get(img_cap[0, j], 'unk'))
                    img_313 = sess.run(c313_tensor, feed_dict={l_tensor: img_l, cap_tensor: img_cap, len_tensor: img_len})
                    img_dec, _ = decode(img_l, img_313, 2.63)

                    word_list = list(img_cap[0, :img_len[0]])
                    if use_vg:
                        img_title = '_'.join(vrev_vg.get(w, 'unk') for w in word_list)
                    else:
                        img_title = '_'.join(vrev.get(w, 'unk') for w in word_list) 
                    io.imsave(os.path.join(orig_dir, '{0}_{1}.jpg').format(i, img_title), img_dec)
                    # io.imsave(os.path.join(orig_dir, '{0}_{1}_att.jpg').format(i, img_title), cv2.resize(img_attention[0, :, :, 0], (224, 224)))
                    print(img_title)

                    if _NEW_CAPTION:
                        new_caption = raw_input('New caption?')
                        new_words = new_caption.strip().split(' ')
                        new_img_cap = np.zeros_like(img_cap)
                        new_img_len = np.zeros_like(img_len)
                        for j in xrange(len(new_words)):
                            if use_vg:
                                new_img_cap[0, j] = train_vocab_vg.get(new_words[j], 0)
                            else:
                                new_img_cap[0, j] = train_vocab.get(new_words[j], 0)
                        new_img_len[0] = len(new_words)
                        new_img_313 = sess.run(c313_tensor, feed_dict={l_tensor: img_l, cap_tensor: new_img_cap, len_tensor: new_img_len})
                        new_img_dec, _ = decode(img_l, new_img_313, 2.63)

                        new_word_list = list(new_img_cap[0, :new_img_len[0]])
                        if use_vg:
                            new_img_title = '_'.join(vrev_vg.get(w, 'unk') for w in new_word_list)
                        else:
                            new_img_title = '_'.join(vrev.get(w, 'unk') for w in new_word_list) 
                        io.imsave(os.path.join(new_dir, '{0}_{1}.jpg').format(i, new_img_title), new_img_dec)
                        # io.imsave(os.path.join(new_dir, '{0}_{1}_att.jpg').format(i, new_img_title), cv2.resize(new_img_attention[0, :, :, 0], (224, 224)))
            finally:
                hf.close()
                print('H5 closed.')


def colorize_video_with_language():
    train_vocab = pickle.load(open('/home/xieya/colorfromlanguage/priors/coco_colors_vocab.p', 'r'))
    vrev = dict((v, k) for (k, v) in train_vocab.iteritems())

    with tf.device('/gpu:0'):
        l_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE, _INPUT_SIZE, 1))
        cap_tensor = tf.placeholder(tf.int32, (1, 20))
        len_tensor = tf.placeholder(tf.int32, (1))
        autocolor = Net(train=False)
        c313_tensor = autocolor.inference4(l_tensor, cap_tensor, len_tensor)
        saver = tf.train.Saver()
        print("Saver created.")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver.restore(sess, _CKPT_PATH)

            caption = raw_input('New caption?')
            words = caption.strip().split(' ')
            img_cap = np.zeros((1, 20), dtype='uint32')
            img_len = np.zeros((1), dtype='uint32')
            for i in xrange(len(words)):
                img_cap[0, i] = train_vocab.get(words[i], 0)
            img_len[0] = len(words)
            word_list = list(img_cap[0, :img_len[0]])
            title = '_'.join(vrev.get(w, 'unk') for w in word_list)
            out_dir = os.path.join(_VIDEO_OUT_DIR, title)
            if not os.path.exists(out_dir): 
                os.makedirs(out_dir)

            img_names = os.listdir(_VIDEO_IN_DIR)
            for img_name in img_names:
                img_bgr = cv2.imread(os.path.join(_VIDEO_IN_DIR, img_name))
                img_bgr = _resize(img_bgr)
                img_l = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                img_l = cv2.resize(img_l, (224, 224), interpolation=cv2.INTER_CUBIC)
                img_l = (img_l.astype(dtype=np.float32)) / 255.0 * 2 - 1
                img_l = img_l[None, :, :, None]
                img_313 = sess.run(c313_tensor, feed_dict={l_tensor: img_l, cap_tensor: img_cap, len_tensor: img_len})
                img_rgb, _ = decode(img_l, img_313, T)
                io.imsave(os.path.join(out_dir, img_name), img_rgb)
                print(img_name)


def colorize_coco_without_language(evaluate=False):
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    val_imgs = hf['val_ims']

    with tf.device('/gpu:0'):
        l_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE, _INPUT_SIZE, 1))
        autocolor = Net(train=False, g_version=_G_VERSION)
        c313_tensor = autocolor.inference(l_tensor)
        if len(c313_tensor) == 2:
            c313_tensor = c313_tensor[0]
        if evaluate:
            gt_313_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 313))
            pred_313_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 313))
            prior_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 1))
            ce_loss_tensor, rb_loss_tensor = cross_entropy_loss(gt_313_tensor, pred_313_tensor, prior_tensor)
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        if evaluate:
            ce_losses = []
            rb_losses = []
            fout = open(os.path.join(_OUTPUT_DIR, 'ce.txt'), 'w')
        with tf.Session(config=config) as sess:
            saver.restore(sess, _CKPT_PATH)
            for i in xrange(192000):
                img_bgr = val_imgs[i]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_lab = color.rgb2lab(img_rgb)
                img_l = img_lab[None, :, :, 0: 1]
                img_ab = img_lab[None, :, :, 1:]
                img_ab_ss = transform.downscale_local_mean(img_ab, (1, 4, 4, 1))
                if utils.is_grayscale(img_ab_ss):
                    continue
                img_l = (img_l.astype(dtype=np.float32) - 50.) / 50.
                img_313 = sess.run(c313_tensor, feed_dict={l_tensor: img_l})
                img_dec, _ = decode(img_l, img_313, T)
                if evaluate:
                    # Evaluate metrics
                    ce_loss, rb_loss = metrics(img_ab_ss, img_313, sess, gt_313_tensor, pred_313_tensor, prior_tensor, ce_loss_tensor, rb_loss_tensor)
                    fout.write('{0}\t{1}\t{2}\n'.format(i, ce_loss, rb_loss))
                    ce_losses.append(ce_loss)
                    rb_losses.append(rb_loss)
                io.imsave(os.path.join(_OUTPUT_DIR, '{0}.jpg'.format(i)), img_dec)
                print(i)
    if evaluate:
        fout.write('ave\t{0}\t{1}\n'.format(np.mean(ce_losses), np.mean(rb_losses)))
        fout.close()
    hf.close()
    print('H5 closed.')


def save_ground_truth():
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    val_imgs = hf['val_ims']
    for i in xrange(200, 400):
        img_bgr = val_imgs[i]
        cv2.imwrite(os.path.join(_OUTPUT_DIR, '{0}.jpg'.format(i)), img_bgr)
        print(i)


def _get_filename_dict(dir_path):
    name_dict = {}
    for fname in os.listdir(dir_path):
        name = os.path.splitext(fname)[0]
        idx, scores = name.split('_', 1)
        idx = int(idx)
        name_dict[idx] = (os.path.join(dir_path, fname), scores)
    return name_dict


def merge(cic_dir, coco_dir, cap_dir, new_cap_dir):
    # gt_dict = _get_filename_dict(gt_dir)
    cic_dict = _get_filename_dict(cic_dir)
    coco_dict = _get_filename_dict(coco_dir)
    cap_dict = _get_filename_dict(cap_dir)
    new_cap_dict = _get_filename_dict(new_cap_dir)
    for idx in coco_dict:
        # gt_img, gt_score = cv2.imread(gt_dict[idx])
        cic_path, cic_score = cic_dict[idx]
        cic_img = cv2.imread(cic_path)
        coco_path, coco_score = coco_dict[idx]
        coco_img = cv2.imread(coco_path)
        cap_path, cap_score = cap_dict[idx]
        cap_img = cv2.imread(cap_path)
        if idx in new_cap_dict:
            new_cap_path, new_cap_score = new_cap_dict[idx]
            new_cap_img = cv2.imread(new_cap_path)
            img = np.hstack((cic_img, coco_img, cap_img, new_cap_img))
            img_name = "{0}_A_{1}_B_{2}_C_{3}_D_{4}.jpg".format(idx, cic_score, coco_score, cap_score, new_cap_score)
            cv2.imwrite(os.path.join(_OUTPUT_DIR, img_name), img)
        else:
            img = np.hstack((cic_img, coco_img, cap_img))
            img_name = "{0}_A_{1}_B_{2}_C_{3}.jpg".format(idx, cic_score, coco_score, cap_score)
            cv2.imwrite(os.path.join(_OUTPUT_DIR, img_name), img)
        print(idx)


def _replace_color(idx, caps, lens, train_vocab):
    path = glob('/srv/glusterfs/xieya/image/color/segcap_21_new/{}_*.jpg'.format(idx))[0]
    fname = os.path.split(path)[1]
    iname = os.path.splitext(fname)[0]
    word_list = iname.split('_')[1:]
    assert len(word_list) == lens[0]
    for i in xrange(len(word_list)):
        caps[0, i] = train_vocab.get(word_list[i], 0)
    return caps


def evaluate(
    with_caption, 
    cross_entropy=False, 
    batch_num=300, 
    is_coco=True, 
    with_attention=False, 
    resize=False, 
    concat=False, 
    use_vg=False, 
    lstm_version=0, 
    with_cocoseg=False,
    random_color=False,
):
    if is_coco:
        dataset_params = {'path': _COCO_PATH, 'thread_num': 1, 'prior_path': _PRIOR_PATH}
        common_params = {'batch_size': _BATCH_SIZE, 'with_caption': '1', 'sampler': '0', }  # with_caption -> False: ignore grayscale images.
        dataset = DataSetCOCO(common_params, dataset_params, False, False, False, with_cocoseg=with_cocoseg)  # No shuffle, same validation set.
    else:
        dataset_params = {'path': '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val.txt', 'thread_num': 8, 
                          'c313': '1', 'cond_l': '0', 'gamma': '0.5'}
        common_params = {'image_size': _INPUT_SIZE, 'batch_size': _BATCH_SIZE, 'is_gan': '0', 'is_rgb': '0'}
        dataset = DataSet(common_params, dataset_params, True, False)  # No shuffle.

    train_vocab_vg = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/glove.6B.100d_voc.p', 'r'))
    # vrev_vg = dict((v, k) for (k, v) in train_vocab_vg.iteritems())
    train_vocab = pickle.load(open('/home/xieya/colorfromlanguage/priors/coco_colors_vocab.p', 'r'))
    vrev = dict((v, k) for (k, v) in train_vocab.iteritems())
    with tf.device('/gpu:0'):
        l_tensor = tf.placeholder(tf.float32, (_BATCH_SIZE, _INPUT_SIZE, _INPUT_SIZE, 1))
        cap_tensor = tf.placeholder(tf.int32, (_BATCH_SIZE, 20))
        len_tensor = tf.placeholder(tf.int32, (_BATCH_SIZE))
        autocolor = Net(train=False, g_version=_G_VERSION, use_vg=use_vg)
        if with_caption:
            if concat:
                c313_tensor = autocolor.inference5(l_tensor, cap_tensor, len_tensor, _CAP_LAYERS, lstm_version=lstm_version)
            else:
                biases = [None] * 8
                for l in _CAP_LAYERS:
                    biases[l] = 1.
                c313_tensor = autocolor.inference4(l_tensor, cap_tensor, len_tensor, biases, with_attention=with_attention)
        else:
            c313_tensor = autocolor.inference(l_tensor)
            if len(c313_tensor) > 1:
                c313_tensor = c313_tensor[0]
        gt_313_tensor = tf.placeholder(tf.float32, (_BATCH_SIZE, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 313))
        prior_tensor = tf.placeholder(tf.float32, (_BATCH_SIZE, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 1))
        if cross_entropy:
            ce_loss_tensor, rb_loss_tensor = cross_entropy_loss(gt_313_tensor, c313_tensor, prior_tensor)
            fout = open(os.path.join(_OUTPUT_DIR, 'ce.txt'), 'w')
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        img_count = 0

        with tf.Session(config=config) as sess:
            saver.restore(sess, _CKPT_PATH)
            print('Ckpt restored.')

            if cross_entropy:
                ce = []
                rb = []
                
            for i in xrange(batch_num):
                print(i)
                if is_coco:
                    img_l, gt_313, prior_boost_nongray, img_cap, img_len = dataset.batch()
                    if random_color:
                        img_cap = _replace_color(img_cap, img_len, train_vocab)
                    if use_vg:
                        for k in xrange(_BATCH_SIZE):
                            for j in xrange(img_len[k]):
                                img_cap[k, j] = train_vocab_vg.get(vrev.get(img_cap[k, j], 'unk'), 0)
                    feed_dict = {l_tensor: img_l, cap_tensor: img_cap, len_tensor: img_len, gt_313_tensor: gt_313, prior_tensor: prior_boost_nongray}
                else:
                    img_l, gt_313, prior_boost_nongray, img_ab = dataset.batch()
                    feed_dict = {l_tensor: img_l, gt_313_tensor: gt_313, prior_tensor: prior_boost_nongray}

                if cross_entropy:
                    img_313, ce_loss, rb_loss = sess.run([c313_tensor, ce_loss_tensor, rb_loss_tensor], feed_dict=feed_dict)
                    ce.extend(ce_loss)
                    rb.extend(rb_loss)
                else:
                    img_313 = sess.run(c313_tensor, feed_dict=feed_dict)
                
                for j in xrange(_BATCH_SIZE):
                    img_count += 1
                    luma = img_l[j: j + 1]
                    if resize:
                        luma = transform.downscale_local_mean(luma, (1, 4, 4, 1))
                    rgb, _ = utils.decode(luma, img_313[j: j + 1], T, return_313=False)
                    if with_caption:
                        word_list = list(img_cap[j, :img_len[j]])
                        img_title = '_'.join(vrev.get(w, 'unk') for w in word_list)
                    else:
                        img_title = ''
                    io.imsave(os.path.join(_OUTPUT_DIR, "{}{}.jpg".format(img_count, img_title, '_rs' if resize else '')), rgb)
                    # if cross_entropy:
                    #     word_list = list(img_cap[j, :img_len[j]])
                    #     if use_vg:
                    #         img_title = '_'.join(vrev_vg.get(w, 'unk') for w in word_list)
                    #     else:
                    #         img_title = '_'.join(vrev.get(w, 'unk') for w in word_list)
                    #     fout.write("{0}_{3}\t{1}\t{2}\n".format(img_count, ce[img_count], rb[img_count], img_title))

            if cross_entropy:
                print("cross entropy {0:.6f}, rebalanced cross entropy {1:.6f}".format(np.mean(ce), np.mean(rb)))
                print("Total {}".format(len(ce)))
                fout.write("ave\t{0}\t{1}\n".format(np.mean(ce), np.mean(rb)))
                fout.close()


if __name__ == "__main__":
    subprocess.check_call(['mkdir', '-p', _OUTPUT_DIR])
    # main(jbu=False)
    # colorize_segcap()
    # reconstruct(jbu=True)
    # places365()
    # demo_wgan_ab()
    # demo_wgan_rgb()
    # _colorize_high_res_img(_IMG_NAME)
    # cifar()
    # colorize_with_language(with_attention=False, concat=True, same_lstm=True, residual=False, paper=True, lstm_version=2, use_vg=False)
    # colorize_video_with_language()
    # colorize_coco_without_language(evaluate=True)
    # save_ground_truth()
    # merge('/srv/glusterfs/xieya/image/color/tf_224_1_476k', 
    #       '/srv/glusterfs/xieya/image/color/tf_coco_5_38k', 
    #       '/srv/glusterfs/xieya/image/color/vgg_5_69k/original', 
    #       '/srv/glusterfs/xieya/image/color/vgg_5_69k/new')
    evaluate(with_caption=True, cross_entropy=True, batch_num=2486, is_coco=True, 
             with_attention=False, resize=False, concat=False, use_vg=False, lstm_version=2, with_cocoseg=True, random_color=False)
    # print("Model {}.".format(_CKPT_PATH))
    # compare_c313_pixelwise()
    # compare_c313()
