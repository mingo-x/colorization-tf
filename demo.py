import random
import subprocess

import h5py
import numpy as np
import pickle
import tensorflow as tf
from utils import *
from net import Net
from skimage import io, color, transform
from sklearn.metrics import auc
import cv2

import utils

_AUC_THRESHOLD = 150
_INPUT_SIZE = 224
_RESIZE_SIZE = 0
_CIFAR_IMG_SIZE = 32
_CIFAR_BATCH_SIZE = 20
_CIFAR_COUNT = 0
_G_VERSION = 1
_CKPT_PATH = '/srv/glusterfs/xieya/tf_coco_5/models/model.ckpt-38000'
IMG_DIR = '/srv/glusterfs/xieya/image/grayscale/colorization_test'
_OUTPUT_DIR = '/srv/glusterfs/xieya/image/color/coco_gt'
_PRIOR_PATH = '/srv/glusterfs/xieya/prior/coco_313_soft.npy'
_IMG_NAME = '/srv/glusterfs/xieya/image/grayscale/cow_gray.jpg'
_VIDEO_IN_DIR = '/srv/glusterfs/xieya/data/DAVIS/JPEGImages/Full-Resolution/bus'
_VIDEO_OUT_DIR = '/srv/glusterfs/xieya/video/bus/vgg_4'
_NEW_CAPTION = True
# T = 2.63
T = 2.63

def _resize(image):
    h = image.shape[0]
    w = image.shape[1]
    resize_size = min(h, w)

    if w > h:
        image = cv2.resize(image, (int(resize_size * w / h), resize_size))

        # crop_start = np.random.randint(0, int(resize_size * w / h) - resize_size + 1)
        crop_start = 0
        image = image[:, crop_start:crop_start + resize_size, :]
    else:
        image = cv2.resize(image, (resize_size, int(resize_size * h / w)))

        # crop_start = np.random.randint(0, int(resize_size * h / w) - resize_size + 1)
        crop_start = 0
        image = image[crop_start:crop_start + resize_size, :, :]
    return image


def _get_model(input_tensor):
    autocolor = Net(train=False, g_version=_G_VERSION)
    conv8_313 = autocolor.inference(input_tensor)
    return conv8_313


def _colorize_single_img(img_name, model, input_tensor, sess):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    # img = _resize(img)
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
    img_rgb, _ = decode(img_l_rs, img_313_rs, T)
    io.imsave(os.path.join(_OUTPUT_DIR, os.path.split(img_name)[1]), img_rgb)


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
    

def main():
    input_tensor = tf.placeholder(
        tf.float32, shape=(1, _INPUT_SIZE, _INPUT_SIZE, 1))
    model = _get_model(input_tensor)
    saver = tf.train.Saver()
    
    sess = tf.Session()
    saver.restore(sess, _CKPT_PATH)

    for img_name in os.listdir(IMG_DIR):
        if img_name.endswith('.jpg') or img_name.endswith('.JPEG'):
            print(img_name)
            _colorize_single_img(img_name, model, input_tensor, sess)
     
    sess.close()


def demo_wgan_ab():
    noise = tf.constant(np.random.normal(size=(64, 128)).astype('float32'))
    model = Net(train=False)
    model.output_dim = 2
    colorized = model.GAN_G(noise)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, _CKPT_PATH)
        ab = sess.run(colorized) # [-1, 1]
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


def is_grayscale(gt_ab):
    thresh = 5
    is_gray = np.sum(np.sum(np.sum(np.abs(gt_ab) > thresh, axis=1), axis=1), axis=1) == 0
    return is_gray


def metrics(gt_ab, pred_313, sess, gt_313_tensor, pred_313_tensor, prior_tensor, ce_loss_tensor, rb_loss_tensor):
    gt_ab_ss = transform.downscale_local_mean(gt_ab, (1, 4, 4, 1))

    #NNEncoder
    #gt_ab_313: [N, H/4, W/4, 313]
    gt_313 = utils._nnencode(gt_ab_ss)

    #Prior_Boost 
    #prior_boost: [N, 1, H/4, W/4]
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
    ce_loss = tf.reduce_mean(ce_loss)
    rb_loss = tf.reduce_sum(rb_loss) / tf.reduce_sum(prior_boost_nongray)

    return ce_loss, rb_loss


def _auc(gt_ab, pred_ab):
    ab_idx = lookup.encode_points(gt_ab[0])
    prior = prior_factor.get_weights(ab_idx)

    l2_dist = np.sqrt(np.sum(np.square(gt_ab[0] - pred_ab), axis=2))
    ones = np.ones_like(l2_dist)
    zeros = np.zeros_like(l2_dist)
    scores = []
    scores_rb = []
    for thr in range(0, _AUC_THRESHOLD + 1):
        score = np.sum(
            np.where(np.less_equal(l2_dist, thr), ones, zeros)) / np.sum(ones)
        score_rb = np.sum(
            np.where(np.less_equal(l2_dist, thr), prior, zeros)) / np.sum(prior)
        scores.append(score)
        scores_rb.append(score_rb)
    x = [i for i in range(0, _AUC_THRESHOLD + 1)]
    auc_score = auc(x, scores)/ _AUC_THRESHOLD
    auc_rb_score = auc(x, scores_rb) / _AUC_THRESHOLD

    return auc_score, auc_rb_score


def colorize_with_language():
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    val_imgs = hf['val_ims']
    val_caps = hf['val_words']
    val_lens = hf['val_length']
    val_num = len(val_imgs)

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
        autocolor = Net(train=False)
        c313_tensor = autocolor.inference4(l_tensor, cap_tensor, len_tensor)
        gt_313_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 313))
        pred_313_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 313))
        prior_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 1))
        ce_loss_tensor, rb_loss_tensor = cross_entropy_loss(gt_313_tensor, pred_313_tensor, prior_tensor)
        saver = tf.train.Saver()
        print("Saver created.")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver.restore(sess, _CKPT_PATH)

            try:
                idx = [335, 3735]
                for i in xrange(200, 400):
                    # i = random.randint(0, val_num - 1)
                    idx.append(i)

                for i in idx:
                    img_bgr = val_imgs[i]
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_lab = color.rgb2lab(img_rgb)
                    img_l = img_lab[None, :, :, 0: 1]
                    img_ab = img_lab[None, :, :, 1:]
                    if is_grayscale(img_ab):
                        continue
                    img_l = (img_l.astype(dtype=np.float32) - 50.) / 50.
                    img_cap = val_caps[i: i + 1]
                    img_len = val_lens[i: i + 1]
                    img_313 = sess.run(c313_tensor, feed_dict={l_tensor: img_l, cap_tensor: img_cap, len_tensor: img_len})
                    img_dec, ab_dec = decode(img_l, img_313, 2.63)
                    ce_loss, rb_loss = metrics(img_ab, img_313, sess, gt_313_tensor, pred_313_tensor, prior_tensor, ce_loss_tensor, rb_loss_tensor)
                    auc_score, auc_rb_score = _auc(img_ab, ab_dec)

                    word_list = list(img_cap[0, :img_len[0]])
                    img_title = '_'.join(vrev.get(w, 'unk') for w in word_list) 
                    io.imsave(os.path.join(orig_dir, '{0}_{1}_{2:.3f}_{3:.3f}_{4:.3f}_{5:.3f}.jpg').format(i, img_title, ce_loss, rb_loss, auc_score, auc_rb_score), img_dec)
                    print(img_title)

                    if _NEW_CAPTION:
                        new_caption = raw_input('New caption?')
                        new_words = new_caption.strip().split(' ')
                        new_img_cap = np.zeros_like(img_cap)
                        new_img_len = np.zeros_like(img_len)
                        for j in xrange(len(new_words)):
                            new_img_cap[0, j] = train_vocab.get(new_words[j], 0)
                        new_img_len[0] = len(new_words)
                        new_img_313 = sess.run(c313_tensor, feed_dict={l_tensor: img_l, cap_tensor: new_img_cap, len_tensor: new_img_len})
                        new_img_dec, new_ab_dec = decode(img_l, new_img_313, 2.63)
                        new_ce_loss, new_rb_loss = metrics(img_ab, new_img_313, sess, gt_313_tensor, pred_313_tensor, prior_tensor, ce_loss_tensor, rb_loss_tensor)
                        new_auc_score, new_auc_rb_score = _auc(img_ab, new_ab_dec)

                        new_word_list = list(new_img_cap[0, :new_img_len[0]])
                        new_img_title = '_'.join(vrev.get(w, 'unk') for w in new_word_list) 
                        io.imsave(os.path.join(new_dir, '{0}_{1}_{2:.3f}_{3:.3f}_{4:.3f}_{5:.3f}.jpg').format(i, new_img_title, new_ce_loss, new_rb_loss, new_auc_score, new_auc_rb_score), new_img_dec)
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


def colorize_coco_without_language():
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    val_imgs = hf['val_ims']
    val_num = len(val_imgs)
    with tf.device('/gpu:0'):
        l_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE, _INPUT_SIZE, 1))
        autocolor = Net(train=False, g_version=_G_VERSION)
        c313_tensor = autocolor.inference(l_tensor)
        gt_313_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 313))
        pred_313_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 313))
        prior_tensor = tf.placeholder(tf.float32, (1, _INPUT_SIZE / 4, _INPUT_SIZE / 4, 1))
        ce_loss_tensor, rb_loss_tensor = cross_entropy_loss(gt_313_tensor, pred_313_tensor, prior_tensor)
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        out_dir = os.path.join(_VIDEO_OUT_DIR, 'nocap')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with tf.Session(config=config) as sess:
            saver.restore(sess, _CKPT_PATH)
            
            # img_names = os.listdir(_VIDEO_IN_DIR)
            for i in xrange(200, 400):
            # for img_name in img_names:
                # img_bgr = cv2.imread(os.path.join(_VIDEO_IN_DIR, img_name))
                # img_bgr = _resize(img_bgr)
                # i = random.randint(0, val_num - 1)
                img_bgr = val_imgs[i]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_lab = color.rgb2lab(img_rgb)
                # img_lab = cv2.resize(img_lab, (224, 224), interpolation=cv2.INTER_CUBIC)
                img_l = img_lab[None, :, :, 0: 1]
                img_ab = img_lab[None, :, :, 1:]
                if is_grayscale(img_ab):
                    continue
                img_l = (img_l.astype(dtype=np.float32) - 50.) / 50.
                img_313 = sess.run(c313_tensor, feed_dict={l_tensor: img_l})
                img_dec, ab_dec = decode(img_l, img_313, T)
                # Evaluate metrics
                ce_loss, rb_loss = metrics(img_ab, img_313, sess, gt_313_tensor, pred_313_tensor, prior_tensor, ce_loss_tensor, rb_loss_tensor)
                auc_score, auc_rb_score = _auc(img_ab, ab_dec)
                io.imsave(os.path.join(_OUTPUT_DIR, '{0}_{1:.3f}_{2:.3f}_{3:.3f}_{4:.3f}.jpg').format(i, ce_loss, rb_loss, auc_score, auc_rb_score), img_dec)
                # io.imsave(os.path.join(out_dir, img_name), img_rgb)
                # cv2.imwrite(os.path.join(_OUTPUT_DIR, '{0}_gt.jpg').format(i), img_bgr)
                print(i)
                # print(img_name)
    hf.close()
    print('H5 closed.')


def save_ground_truth():
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    val_imgs = hf['val_ims']
    for i in xrange(200, 400):
        img_bgr = val_imgs[i]
        cv2.imwrite(os.path.join(_OUTPUT_DIR, '{0}.jpg'.format(i)), img_bgr)
        print(i)


def merge(gt_dir, cic_dir, coco_dir, cap_dir):
    return


if __name__ == "__main__":
    subprocess.check_call(['mkdir', '-p', _OUTPUT_DIR])
    lookup = utils.LookupEncode('resources/pts_in_hull.npy')
    prior_factor = utils.PriorFactor(priorFile=_PRIOR_PATH)
    # main()
    # places365()
    # demo_wgan_ab()
    # demo_wgan_rgb()
    # _colorize_high_res_img(_IMG_NAME)
    # cifar()
    # colorize_with_language()
    # colorize_video_with_language()
    # colorize_coco_without_language()
    save_ground_truth()
