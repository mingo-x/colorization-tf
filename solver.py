from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from ops import *
from net import Net
from data import DataSet
import time
from datetime import datetime
import os
import sys


_LOG_FREQ = 10


class Solver(object):

  def __init__(self, train=True, common_params=None, solver_params=None, net_params=None, dataset_params=None):
    if common_params:
      self.device_id = int(common_params['gpus'])
      self.image_size = int(common_params['image_size'])
      self.height = self.image_size
      self.width = self.image_size
      self.batch_size = int(common_params['batch_size'])
      self.num_gpus = 1
    if solver_params:
      self.learning_rate = float(solver_params['learning_rate'])
      # self.moment = float(solver_params['moment'])
      self.max_steps = int(solver_params['max_iterators'])
      self.train_dir = str(solver_params['train_dir'])
      self.lr_decay = float(solver_params['lr_decay'])
      self.decay_steps = int(solver_params['decay_steps'])
    self.train = train
    self.net = Net(train=train, common_params=common_params, net_params=net_params)
    self.dataset = DataSet(common_params=common_params, dataset_params=dataset_params)
    print("Solver initialization done.")

  def construct_graph(self, scope):
    with tf.device('/gpu:' + str(self.device_id)):
      self.data_l = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 1))
      self.gt_ab_313 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 313))
      self.prior_boost_nongray = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 1))

      self.conv8_313 = self.net.inference(self.data_l)
      ab_fake = self.net.conv313_to_ab(self.conv8_313)
      data_l_ss = self.data_l[:, ::4, ::4, :]
      data_lab_fake = tf.concat([data_l_ss, ab_fake], axis=-1)
      D_fake_pred = self.net.discriminator(data_lab_fake)
      self.data_lab_real = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 2))
      self.D_real_pred = self.net.discriminator(self.data_lab_real, True)  # Reuse the variables.

      new_loss, g_loss, adv_loss = self.net.loss(scope, self.conv8_313, self.prior_boost_nongray, self.gt_ab_313, D_fake_pred)
      tf.summary.scalar('new_loss', new_loss)
      tf.summary.scalar('total_loss', g_loss)

      D_loss, real_score, fake_score = self.net.discriminator_loss(self.D_real_pred, D_fake_pred)
      tf.summary.scalar('D loss', D_loss)
      tf.summary.scalar('real_score', real_score)
      tf.summary.scalar('fake_score', fake_score)
    return new_loss, g_loss, adv_loss, D_loss, real_score, fake_score

  def train_model(self):
    with tf.device('/gpu:' + str(self.device_id)):
      self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           self.decay_steps, self.lr_decay, staircase=True)
      opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99)
      D_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99)
      with tf.name_scope('gpu') as scope:
        self.new_loss, self.total_loss, self.adv_loss, self.D_loss, self.real_score, self.fake_score = self.construct_graph(scope)
        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

      D_vars = tf.trainable_variables(scope='D')
      G_vars = tf.trainable_variables(scope='G')

      grads = opt.compute_gradients(self.new_loss, var_list=G_vars)
      D_grads = D_opt.compute_gradients(self.D_loss, var_list=D_vars)

      self.summaries.append(tf.summary.scalar('learning_rate', learning_rate))

      # for grad, var in grads:
      #   if grad is not None:
      #     self.summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

      apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
      D_apply_gradient_op = D_opt.apply_gradients(D_grads, global_step=self.global_step)

      # for var in tf.trainable_variables():
      #   self.summaries.append(tf.summary.histogram(var.op.name, var))

      variable_averages = tf.train.ExponentialMovingAverage(
          0.999, self.global_step)
      variables_averages_op = variable_averages.apply(tf.trainable_variables(scope='G'))

      train_op = tf.group(apply_gradient_op, variables_averages_op)

      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
      # saver1 = tf.train.Saver()
      summary_op = tf.summary.merge(self.summaries)
      init =  tf.global_variables_initializer()
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
      print("Session configured.")
      sess.run(init)
      print("Initialized.")
      #saver1.restore(sess, './models/model.ckpt')
      #nilboy
      summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
      start_time = time.time()
      for step in xrange(self.max_steps):

        t1 = time.time()
        data_l, gt_ab_313, prior_boost_nongray, data_lab_real = self.dataset.batch()
        # _, _, _, data_lab_real = self.dataset.batch()
        t2 = time.time()
        if t2 - t1 > 0.05:
          print ('step: {0} io: {1}'.format(step, t2 - t1))
        # Discriminator training.
        sess.run([D_apply_gradient_op], feed_dict={self.data_l: data_l, self.data_lab_real: data_lab_real})
        # t3 = time.time()
        # Generator training.
        sess.run([train_op], feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})
        # t4 = time.time()

        if step % _LOG_FREQ == 0:
          duration = time.time() - start_time
          num_examples_per_step = self.batch_size * self.num_gpus * _LOG_FREQ
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = duration / (self.num_gpus * _LOG_FREQ)

          loss_value, new_loss_value, real_score_value, fake_score_value = sess.run(
            [self.total_loss, self.new_loss, self.real_score, self.fake_score], 
            feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray, self.data_lab_real: data_lab_real})
          format_str = ('%s: step %d, G loss = %.2f, new loss = %.2f, real score = %0.2f, fake score = %0.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          # assert not np.isnan(adv_loss_value), 'Adversarial diverged with loss = NaN'
          # assert not np.isnan(D_loss_value), 'Discriminator diverged with loss = NaN'
          print (format_str % (datetime.now(), step, loss_value, new_loss_value, real_score_value, fake_score_value,
                               examples_per_sec, sec_per_batch))
          start_time = time.time()
        
        if step % 100 == 0:
          summary_str = sess.run(summary_op, feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray, self.data_lab_real: data_lab_real})
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0:
          checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
