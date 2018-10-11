from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import pickle
from skimage import io, color
import tensorflow as tf

from ops import *
from net import Net
from data_coco import DataSet
import time
from datetime import datetime
import os

import utils


_LOG_FREQ = 10


class Solver_Language(object):

    def __init__(
        self,
        train=True,
        common_params=None,
        solver_params=None,
        net_params=None,
        dataset_params=None
    ):
        if common_params:
            self.device_id = int(common_params['gpus'])
            self.image_size = int(common_params['image_size'])
            self.height = self.image_size
            self.width = self.image_size
            self.batch_size = int(common_params['batch_size'])
            self.num_gpus = 1
            self.d_repeat = int(common_params['d_repeat'])
            self.g_repeat = int(common_params['g_repeat'])
            self.ckpt = common_params['ckpt'] if 'ckpt' in common_params else None
            self.init_ckpt = common_params['init_ckpt'] if 'init_ckpt' in common_params else None
            self.restore_opt = True if common_params['restore_opt'] == '1' else False
            self.gan = True if common_params['gan'] == '1' else False
            self.prior_boost = True if common_params['prior_boost'] == '1' else False
            self.corr = True if common_params['correspondence'] == '1' else False
            self.with_caption = True if common_params['with_caption'] == '1' else False
            if self.corr:
                print('Discriminator has correspondence.')
            else:
                print('Discriminator has no correspondence.')
            if self.gan:
                print('Using GAN.')
            else:
                print('Not using GAN.')
            if self.prior_boost:
                print('Using prior boost.')
            else:
                print('Not using prior boost.')

        if solver_params:
            self.learning_rate = float(solver_params['learning_rate'])
            self.D_learning_rate = float(solver_params['d_learning_rate'])
            print("Learning rate G: {0} D: {1}".format(self.learning_rate, self.D_learning_rate))
            # self.moment = float(solver_params['moment'])
            self.max_steps = int(solver_params['max_iterators'])
            self.train_dir = str(solver_params['train_dir'])
            self.lr_decay = float(solver_params['lr_decay'])
            self.decay_steps = int(solver_params['decay_steps'])
            self.moment = float(solver_params['moment'])
        self.train = train
        self.net = Net(
            train=train, common_params=common_params, net_params=net_params)
        self.dataset = DataSet(
            common_params=common_params, dataset_params=dataset_params)
        print("Solver initialization done.")

    def construct_graph(self, scope, sess):
        with tf.device('/gpu:' + str(self.device_id)):
            self.data_l = tf.placeholder(
                tf.float32, (self.batch_size, self.height, self.width, 1))
            self.captions = tf.placeholder(tf.int32, (self.batch_size, 20))
            self.lens = tf.placeholder(tf.int32, (self.batch_size))
            self.gt_ab_313 = tf.placeholder(
                tf.float32,
                (self.batch_size, int(self.height / 4), int(self.width / 4), 313)
            )
            self.prior_boost_nongray = tf.placeholder(
                tf.float32,
                (self.batch_size, int(self.height / 4), int(self.width / 4), 1)
            )

            if self.with_caption:
                if self.ckpt is None and self.init_ckpt is not None:
                    # Restore gamma and beta of BN.
                    biases = []
                    for i in xrange(8):
                        gamma = tf.get_variable('G/bn_{}/gamma'.format(i + 1), (self.batch_size, ), dtype=tf.float32)
                        beta = tf.get_variable('G/bn_{}/beta'.format(i + 1), (self.batch_size, ), dtype=tf.float32)
                        bn_saver = tf.train.Saver({'G/bn_{}/gamma'.format(i + 1): gamma, 'G/bn_{}/beta'.format(i + 1): beta})
                        bn_saver.restore(sess, self.init_ckpt)
                        gamma = gamma[:, tf.newaxis]
                        beta = beta[:, tf.newaxis]
                        gamma = tf.manip.tile(gamma, (1, self.net.in_dims[i]))
                        beta = tf.manip.tile(beta, (1, self.net.in_dims[i]))
                        bias = tf.concat((gamma, beta), axis=-1)
                        biases.append(sess.run(bias))
                        print(biases[-1].shape)
                self.conv8_313 = self.net.inference4(self.data_l, self.captions, self.lens, biases)
            else:
                self.conv8_313 = self.net.inference(self.data_l)
            # self.colorized_ab = self.net.conv313_to_ab(conv8_313)

            new_loss, g_loss, _ = self.net.loss(
                scope, self.conv8_313, self.prior_boost_nongray,
                self.gt_ab_313, None, self.gan,
                self.prior_boost)
            tf.summary.scalar('new_loss', new_loss)
            tf.summary.scalar('total_loss', g_loss)
            print('Graph constructed.')

            return new_loss, g_loss

    def train_model(self):
        with tf.device('/gpu:' + str(self.device_id)):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                       self.decay_steps, self.lr_decay, staircase=True)

            train_vocab = pickle.load(open('/home/xieya/colorfromlanguage/priors/coco_colors_vocab.p', 'r'))
            vrev = dict((v, k) for (k, v) in train_vocab.iteritems())

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            print("Session configured.")

            with tf.name_scope('gpu') as scope:
                self.new_loss, self.total_loss = self.construct_graph(scope, sess)
                self.summaries = tf.get_collection(
                    tf.GraphKeys.SUMMARIES, scope)

            self.summaries.append(
                tf.summary.scalar('learning_rate', learning_rate))

            opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=self.moment, beta2=0.99)
            grads = opt.compute_gradients(self.new_loss)

            for grad, var in grads:
                if grad is not None:
                    self.summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
            for var in tf.global_variables():
                self.summaries.append(tf.summary.histogram(var.op.name, var))

            apply_gradient_op = opt.apply_gradients(
                grads, global_step=self.global_step)
            variable_averages = tf.train.ExponentialMovingAverage(
                0.999, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            summary_op = tf.summary.merge(self.summaries)
            init = tf.global_variables_initializer()
        
            if self.ckpt is not None:
                saver.restore(sess, self.ckpt)
                print(self.ckpt + " restored.")
                start_step = sess.run(self.global_step)
                start_step -= int(start_step % 10)
                print("Global step: {}".format(start_step))
            else:
                sess.run(init)
                print("Initialized.")
                start_step = 0

                if self.init_ckpt is not None:
                    init_saver = tf.train.Saver(tf.global_variables(scope='G'))
                    init_saver.restore(sess, self.init_ckpt)
                    print('Init generator with {}.'.format(self.init_ckpt))

            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            start_time = time.time()
            start_step = int(start_step)

            for step in xrange(start_step, self.max_steps, self.g_repeat):
                # Generator training.
                data_l, gt_ab_313, prior_boost_nongray, captions, lens = self.dataset.batch()
                sess.run([train_op], feed_dict={
                    self.data_l: data_l, self.gt_ab_313: gt_ab_313, self.prior_boost_nongray: prior_boost_nongray,
                    self.captions: captions, self.lens: lens})

                if step % _LOG_FREQ == 0:
                    duration = time.time() - start_time
                    num_examples_per_step = self.batch_size * self.num_gpus * _LOG_FREQ
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / (self.num_gpus * _LOG_FREQ)

                    loss_value, new_loss_value = sess.run([self.total_loss, self.new_loss], feed_dict={
                        self.data_l: data_l, self.gt_ab_313: gt_ab_313, self.prior_boost_nongray: prior_boost_nongray,
                        self.captions: captions, self.lens: lens})
                    format_str = ('%s: step %d, G loss = %.2f, new loss = %.2f(%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(),
                                         step, loss_value, new_loss_value,
                                         examples_per_sec, sec_per_batch))
                    start_time = time.time()

                if step % 100 == 0:
                    summary_str, img_313s = sess.run([summary_op, self.conv8_313], feed_dict={
                        self.data_l: data_l, self.gt_ab_313: gt_ab_313, self.prior_boost_nongray: prior_boost_nongray, self.captions: captions, self.lens: lens})
                    summary_writer.add_summary(summary_str, step)
                    # Save sample image
                    img_313 = img_313s[0: 1]
                    img_l = data_l[0: 1]
                    img_rgb, _ = utils.decode(img_l, img_313, 2.63)
                    word_list = list(captions[0, :lens[0]])
                    img_caption = '_'.join(vrev.get(w, 'unk') for w in word_list) 
                    io.imsave(os.path.join(self.train_dir, '{0}_{1}.jpg').format(step, img_caption), img_rgb)

                # Save the model checkpoint periodically.
                if step % 1000 == 0:
                    checkpoint_path = os.path.join(
                        self.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            sess.close()
