from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage import io, color
import tensorflow as tf

from ops import *
from net import Net
from data import DataSet
import utils

import time
from datetime import datetime
import os


_LOG_FREQ = 10


class Solver_GAN(object):

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
            self.is_rgb = True if common_params['is_rgb'] == '1' else False
            self.output_dim = 3 if self.is_rgb else 2

        if solver_params:
            self.learning_rate = float(solver_params['learning_rate'])
            self.D_learning_rate = float(solver_params['d_learning_rate'])
            print("Learning rate G: {0} D: {1}".format(self.learning_rate, self.D_learning_rate))
            # self.moment = float(solver_params['moment'])
            self.max_steps = int(solver_params['max_iterators'])
            self.train_dir = str(solver_params['train_dir'])

        self.train = train
        self.net = Net(
            train=train, common_params=common_params, net_params=net_params)
        self.dataset = DataSet(
            common_params=common_params, dataset_params=dataset_params)
        print("Solver initialization done.")

    def construct_graph(self, scope):
        with tf.device('/gpu:' + str(self.device_id)):
            self.data_real = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, self.output_dim))

            data_fake = self.net.GAN_G()
            gen_loss, dis_loss, w_dist, mixed_norm = self.net.GAN_loss(self.data_real, data_fake)

            tf.summary.scalar('W distance', w_dist)
            tf.summary.scalar('G loss', gen_loss)
            tf.summary.scalar('D loss', dis_loss)
            tf.summary.scalar('Mixed norm', mixed_norm)

            return gen_loss, dis_loss, w_dist, mixed_norm

    def train_model(self):
        with tf.device('/gpu:' + str(self.device_id)):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            with tf.name_scope('gpu') as scope:
                self.G_loss, self.D_loss, self.W_dist, self.mixed_norm = self.construct_graph(scope)
                self.summaries = tf.get_collection(
                    tf.GraphKeys.SUMMARIES, scope)

            opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=0., beta2=0.9)
            G_vars = tf.trainable_variables(scope='G')
            grads = opt.compute_gradients(self.G_loss, var_list=G_vars)
            G_apply_gradient_op = opt.apply_gradients(
                grads, global_step=self.global_step)

            D_opt = tf.train.AdamOptimizer(
                learning_rate=self.D_learning_rate, beta1=0., beta2=0.9)
            D_vars = tf.trainable_variables(scope='D')
            D_grads = D_opt.compute_gradients(self.D_loss, var_list=D_vars)
            D_apply_gradient_op = D_opt.apply_gradients(D_grads)

            fixed_noise = tf.constant(np.random.normal(size=(64, 128)).astype('float32'))
            test_samples = self.net.GAN_G(fixed_noise)

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            summary_op = tf.summary.merge(self.summaries)
            init = tf.global_variables_initializer()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            print("Session configured.")

            if self.ckpt is not None:
                if self.restore_opt:
                    saver.restore(sess, self.ckpt)
                else:
                    sess.run(init)
                    init_saver = tf.train.Saver(G_vars + D_vars + [self.global_step])
                    init_saver.restore(sess, self.ckpt)

                print(self.ckpt + " restored.")
                start_step = sess.run(self.global_step)
                start_step -= int(start_step % 10)
                print("Global step: {}".format(start_step))
            else:
                sess.run(init)
                print("Variables initialized.")
                start_step = 0

                if self.init_ckpt is not None:
                    init_saver = tf.train.Saver(G_vars)
                    init_saver.restore(sess, self.init_ckpt)
                    print('Init generator with {}.'.format(self.init_ckpt))

            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            start_time = time.time()
            start_step = int(start_step)

            for step in xrange(start_step, self.max_steps, self.g_repeat):
                for _ in xrange(self.d_repeat):
                    data_real = self.dataset.batch()
                    # Discriminator training.
                    sess.run([D_apply_gradient_op],
                                feed_dict={self.data_real: data_real})

                if step % _LOG_FREQ < self.g_repeat:
                    d_loss_value, w_dist_value = sess.run([self.D_loss, self.W_dist], 
                        feed_dict={self.data_real: data_real})

                for _ in xrange(self.g_repeat):
                    sess.run([G_apply_gradient_op])

                if step % _LOG_FREQ < self.g_repeat:
                    duration = time.time() - start_time
                    num_examples_per_step = self.batch_size * self.num_gpus * _LOG_FREQ
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / (self.num_gpus * _LOG_FREQ)

                    g_loss_value = sess.run(self.G_loss)
                    format_str = ('%s: step %d, G loss = %.5f, D loss= %0.5f, W div = %0.3f(%.1f examples/sec; %.3f '
                                      'sec/batch)')
                    print (format_str % (datetime.now(), step, g_loss_value, d_loss_value, w_dist_value, examples_per_sec, sec_per_batch))
                    start_time = time.time()

                if step % 100 < self.g_repeat:
                    summary_str = sess.run(summary_op, feed_dict={self.data_real: data_real})
                    summary_writer.add_summary(summary_str, step)
                    test_images = sess.run(test_samples)
                    if self.is_rgb:
                        test_images = ((test_images+1.)*(255.99/2)).astype('uint8')
                        test_lab = []
                        for i in xrange(64):
                            lab = color.rgb2lab(test_images[i, :, :, :])
                            lab[:, :, 0] = 50.  # Remove l.
                            test_lab.append(color.lab2rgb(lab))
                        test_lab = np.array(test_lab)
                        utils.save_images(test_lab, os.path.join(self.train_dir, "{}_ab.png".format(step)))
                    else:
                        test_ab = 110. * test_images
                        test_l = np.full((64, 64, 64, 1), 50)
                        test_lab = np.concatenate((test_l, test_ab), axis=-1)
                        test_rgb = []
                        for i in xrange(64):
                            rgb = color.lab2rgb(test_lab[i, :, :, :])
                            test_rgb.append(rgb)
                        test_images = np.array(test_rgb)

                    utils.save_images(test_images, os.path.join(self.train_dir, "{}.png".format(step)))

                # Save the model checkpoint periodically.
                if step % 1000 < self.g_repeat:
                    checkpoint_path = os.path.join(
                        self.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
