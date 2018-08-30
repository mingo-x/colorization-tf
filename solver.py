from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ops import *
from net import Net
from data import DataSet
import time
from datetime import datetime
import os


_LOG_FREQ = 10


class Solver(object):

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
            # self.moment = float(solver_params['moment'])
            self.max_steps = int(solver_params['max_iterators'])
            self.train_dir = str(solver_params['train_dir'])
            self.lr_decay = float(solver_params['lr_decay'])
            self.decay_steps = int(solver_params['decay_steps'])
        self.train = train
        self.net = Net(
            train=train, common_params=common_params, net_params=net_params)
        self.dataset = DataSet(
            common_params=common_params, dataset_params=dataset_params)
        print("Solver initialization done.")

    def construct_graph(self, scope):
        with tf.device('/gpu:' + str(self.device_id)):
            self.data_l = tf.placeholder(
                tf.float32, (self.batch_size, self.height, self.width, 1))
            self.gt_ab_313 = tf.placeholder(
                tf.float32,
                (self.batch_size, int(self.height / 4), int(self.width / 4), 313)
            )
            self.prior_boost_nongray = tf.placeholder(
                tf.float32,
                (self.batch_size, int(self.height / 4), int(self.width / 4), 1)
            )

            conv8_313 = self.net.inference(self.data_l)
            # conv8_313_prob = tf.nn.softmax(conv8_313)
            ab_fake_ss = self.net.conv313_to_ab(conv8_313) / 110.
            # ab_fake = tf.image.resize_images(ab_fake_ss, (self.height, self.width))
            data_l_ss = self.data_l[:, ::4, ::4, :] / 50.
            data_fake = tf.concat([data_l_ss, ab_fake_ss], axis=-1)
            # data_fake = tf.image.resize_images(data_fake, (self.height, self.width))
            D_fake_pred = self.net.discriminator(data_fake)
            self.data_real = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 3))
            # self.data_l_ss_real = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 1))
            # self.gt_ab_313_real = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 313))
            # self.data_lab_real = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 3))
            # data_real = tf.concat([self.data_l_ss_real, self.gt_ab_313_real], axis=-1)
            D_real_pred = self.net.discriminator(self.data_real, True)  # Reuse the variables.

            new_loss, g_loss, adv_loss = self.net.loss(
                scope, conv8_313, self.prior_boost_nongray,
                self.gt_ab_313, D_fake_pred, self.gan,
                self.prior_boost)
            tf.summary.scalar('new_loss', new_loss)
            tf.summary.scalar('total_loss', g_loss)

            if self.gan:
                D_loss, real_score, fake_score = self.net.discriminator_loss(D_real_pred, D_fake_pred)
                tf.summary.scalar('D loss', D_loss)
                tf.summary.scalar('real_score', real_score)
                tf.summary.scalar('fake_score', fake_score)
                tf.summary.scalar('adv_loss', adv_loss)
                return (new_loss, g_loss, adv_loss, D_loss, real_score, fake_score)
            else:
                return (new_loss, g_loss, adv_loss, None, None, None)


    def train_model(self):
        with tf.device('/gpu:' + str(self.device_id)):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
            #                                      self.decay_steps, self.lr_decay, staircase=True)
            learning_rate = self.learning_rate
            D_learning_rate = 2e-4

            with tf.name_scope('gpu') as scope:
                self.new_loss, self.total_loss, self.adv_loss, self.D_loss, self.real_score, self.fake_score = self.construct_graph(scope)
                self.summaries = tf.get_collection(
                    tf.GraphKeys.SUMMARIES, scope)

            self.summaries.append(
                tf.summary.scalar('learning_rate', learning_rate))

            opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=0.5, beta2=0.99)
            G_vars = tf.trainable_variables(scope='G')
            T_var = tf.trainable_variables(scope='T')
            grads = opt.compute_gradients(self.new_loss * self.net.alpha, var_list=G_vars)
            grads_adv = opt.compute_gradients(self.adv_loss, var_list=G_vars + T_var)

            for grad, var in grads:
                if grad is not None:
                    self.summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            for grad, var in grads_adv:
                if grad is not None:
                    self.summaries.append(tf.summary.histogram(var.op.name + '/gradients_adv', grad))

            with tf.variable_scope('T', reuse=True):
                T = tf.get_variable('T')
            self.summaries.append(tf.summary.scalar(T.op.name, T[0]))

            # for var in G_vars:
            #     self.summaries.append(tf.summary.histogram(var.op.name, var))

            apply_gradient_op = opt.apply_gradients(
                grads, global_step=self.global_step)
            apply_gradient_adv_op = opt.apply_gradients(
                grads_adv)
            variable_averages = tf.train.ExponentialMovingAverage(
                0.999, self.global_step)
            variables_averages_op = variable_averages.apply(G_vars + T_var)
            train_op = tf.group(apply_gradient_op, apply_gradient_adv_op, variables_averages_op)

            if self.gan:
                D_opt = tf.train.AdamOptimizer(
                    learning_rate=D_learning_rate, beta1=0.5, beta2=0.99)
                D_vars = tf.trainable_variables(scope='D')
                D_grads = D_opt.compute_gradients(self.D_loss, var_list=D_vars)
                D_apply_gradient_op = D_opt.apply_gradients(D_grads)

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
                    init_saver = tf.train.Saver(G_vars + T_var + D_vars + [self.global_step])
                    init_saver.restore(sess, self.ckpt)

                print(self.ckpt + " restored.")
                start_step = sess.run(self.global_step)
                # start_step = 230000
                # sess.run(self.global_step.assign(start_step))
                print("Global step: {}".format(start_step))
            else:
                sess.run(init)
                print("Initialized.")
                start_step = 0

                if self.init_ckpt is not None:
                    init_saver = tf.train.Saver(G_vars)
                    init_saver.restore(sess, self.init_ckpt)
                    print('Init generator with {}.'.format(self.init_ckpt))

            start_temp = sess.run(T)
            print("Temp {}.".format(start_temp))

            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            start_time = time.time()

            for step in xrange(start_step, self.max_steps, self.g_repeat):
                if self.gan:
                    for _ in xrange(self.d_repeat):
                        data_l, _, _, data_real = self.dataset.batch()
                        if not self.corr:
                            _, _, _, data_real = self.dataset.batch()
                        # Discriminator training.
                        sess.run([D_apply_gradient_op],
                                  feed_dict={self.data_l: data_l, self.data_real: data_real})

                if step % _LOG_FREQ == 0:
                    d_loss_value, adv_loss_prev_value = sess.run([self.D_loss, self.adv_loss], 
                        feed_dict={self.data_l:data_l, self.data_real: data_real})

                # Generator training.
                # sess.run([train_op], 
                          # feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})
                for _ in xrange(self.g_repeat):
                    data_l, gt_ab_313, prior_boost_nongray, _ = self.dataset.batch()
                    sess.run([train_op], 
                              feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})

                if step % _LOG_FREQ == 0:
                    duration = time.time() - start_time
                    num_examples_per_step = self.batch_size * self.num_gpus * _LOG_FREQ
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / (self.num_gpus * _LOG_FREQ)

                    if self.gan:
                        loss_value, new_loss_value, adv_loss_value = sess.run(
                          [self.total_loss, self.new_loss, self.adv_loss], 
                          feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray, self.data_real: data_real})
                        format_str = ('%s: step %d, G loss = %.2f, new loss = %.2f, adv prev = %0.5f, adv = %0.5f, D = %0.5f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                        # assert not np.isnan(adv_loss_value), 'Adversarial diverged with loss = NaN'
                        # assert not np.isnan(D_loss_value), 'Discriminator diverged with loss = NaN'
                        print (format_str % (datetime.now(), step, loss_value, new_loss_value, adv_loss_prev_value, adv_loss_value, d_loss_value,
                                             examples_per_sec, sec_per_batch))
                    else:
                        loss_value, new_loss_value = sess.run(
                            [self.total_loss, self.new_loss],
                            feed_dict={self.data_l: data_l, self.gt_ab_313: gt_ab_313, self.prior_boost_nongray: prior_boost_nongray})
                        format_str = ('%s: step %d, G loss = %.2f, new loss = %.2f(%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                        # assert not np.isnan(adv_loss_value), 'Adversarial diverged with loss = NaN'
                        # assert not np.isnan(D_loss_value), 'Discriminator diverged with loss = NaN'
                        print (format_str % (datetime.now(),
                                             step, loss_value, new_loss_value,
                                             examples_per_sec, sec_per_batch))
                    start_time = time.time()

                if step % 100 == 0:
                    if self.gan:
                        summary_str = sess.run(summary_op, feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray, self.data_real: data_real})
                    else:
                        summary_str = sess.run(summary_op, feed_dict={
                            self.data_l: data_l, self.gt_ab_313: gt_ab_313, self.prior_boost_nongray: prior_boost_nongray})
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0:
                    checkpoint_path = os.path.join(
                        self.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
