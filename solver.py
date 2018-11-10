from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage import io, color
import tensorflow as tf

from ops import *
from net import Net
from data import DataSet
import time
from datetime import datetime
import os


_LOG_FREQ = 10


def scalar_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


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
        self.val_dataset = DataSet(
            common_params=common_params, dataset_params=dataset_params, training=False)
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

            new_loss, g_loss, wd_loss, rb_loss = self.net.loss(
                scope, conv8_313, self.prior_boost_nongray,
                self.gt_ab_313)
            tf.summary.scalar('new_loss', new_loss)
            tf.summary.scalar('rb_loss', rb_loss)
            tf.summary.scalar('wd_loss', wd_loss)
            tf.summary.scalar('total_loss', g_loss)

            return (new_loss, g_loss, rb_loss)

    def lr_decay_on_plateau(self, sess, curr_loss, threshold=3):
        if curr_loss >= self.prev_loss:
            self.increasing_count += 1
            if self.increasing_count == threshold:
                # Decay.
                old_lr = self.learning_rate_tensor.value()
                sess.run(self.learning_rate_tensor.assign(old_lr * 0.1))
                print('Learning rate decayed to {0}.'.format(old_lr.eval(session=sess)))
                self.increasing_count = 0
        else:
            self.increasing_count = 0

        self.prev_loss = curr_loss

    def train_model(self):
        with tf.device('/gpu:' + str(self.device_id)):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            
            self.learning_rate_tensor = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                                   self.decay_steps, self.lr_decay, staircase=True)
            with tf.name_scope('gpu') as scope:
                self.new_loss, self.total_loss, self.rb_loss = self.construct_graph(scope)
                self.summaries = tf.get_collection(
                    tf.GraphKeys.SUMMARIES, scope)

            self.summaries.append(
                tf.summary.scalar('learning_rate', self.learning_rate_tensor))

            opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_tensor, beta1=self.moment, beta2=0.99)
            G_vars = tf.trainable_variables(scope='G')
            grads = opt.compute_gradients(self.new_loss, var_list=G_vars)

            total_param = 0
            for var in tf.global_variables(scope='G'):
                print(var)
                total_param += np.prod(var.get_shape())
            print('Total params: {}.'.format(total_param))

            apply_gradient_op = opt.apply_gradients(
                grads, global_step=self.global_step)
            variable_averages = tf.train.ExponentialMovingAverage(
                0.999, self.global_step)
            variables_averages_op = variable_averages.apply(G_vars)
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            savable_vars = tf.global_variables()
            saver = tf.train.Saver(savable_vars, write_version=tf.train.SaverDef.V2, max_to_keep=5, keep_checkpoint_every_n_hours=1)
            summary_op = tf.summary.merge(self.summaries)
            init = tf.global_variables_initializer()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            print("Session configured.")

            if self.ckpt is not None:
                # sess.run(self.learning_rate_tensor.initializer)
                if self.restore_opt:
                    saver.restore(sess, self.ckpt)
                else:
                    sess.run(init)
                    init_saver = tf.train.Saver(G_vars + T_vars + D_vars + [self.global_step])
                    init_saver.restore(sess, self.ckpt)

                print(self.ckpt + " restored.")
                start_step = sess.run(self.global_step)
                start_step -= int(start_step % 10)
                # start_step = 230000
                # sess.run(self.global_step.assign(start_step))
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
                for _ in xrange(self.g_repeat):
                    data_l, gt_ab_313, prior_boost_nongray, _ = self.dataset.batch()
                    sess.run([train_op], 
                             feed_dict={self.data_l: data_l, self.gt_ab_313: gt_ab_313, self.prior_boost_nongray: prior_boost_nongray})

                if step % _LOG_FREQ < self.g_repeat:
                    duration = time.time() - start_time
                    num_examples_per_step = self.batch_size * self.num_gpus * _LOG_FREQ
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / (self.num_gpus * _LOG_FREQ)

                    loss_value, new_loss_value, rb_loss_value = sess.run(
                        [self.total_loss, self.new_loss, self.rb_loss],
                        feed_dict={self.data_l: data_l, self.gt_ab_313: gt_ab_313, self.prior_boost_nongray: prior_boost_nongray})
                    format_str = ('%s: step %d, G loss = %.2f, new loss = %.2f rb loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    # assert not np.isnan(adv_loss_value), 'Adversarial diverged with loss = NaN'
                    # assert not np.isnan(D_loss_value), 'Discriminator diverged with loss = NaN'
                    print (format_str % (datetime.now(),
                                         step, loss_value, new_loss_value, rb_loss_value,
                                         examples_per_sec, sec_per_batch))
                    start_time = time.time()

                if step % 100 < self.g_repeat:
                    summary_str = sess.run(summary_op, feed_dict={
                        self.data_l: data_l, self.gt_ab_313: gt_ab_313, self.prior_boost_nongray: prior_boost_nongray})
                    eval_loss = 0.0
                    eval_loss_rb = 0.0
                    eval_iters = 30
                    for _ in xrange(eval_iters):
                        val_data_l, val_gt_ab_313, val_prior_boost_nongray, _ = self.val_dataset.batch()
                        loss_value, rb_loss_value = sess.run([self.total_loss, self.rb_loss], feed_dict={
                            self.data_l: val_data_l, self.gt_ab_313: val_gt_ab_313, self.prior_boost_nongray: val_prior_boost_nongray})
                        eval_loss += loss_value
                        eval_loss_rb += rb_loss_value
                    eval_loss /= eval_iters
                    eval_loss_rb /= eval_iters
                    eval_loss_sum = scalar_summary('eval/loss', eval_loss)
                    eval_loss_rb_sum = scalar_summary('eval/loss_rb', eval_loss_rb)
                    summary_writer.add_summary(eval_loss_sum, step)
                    summary_writer.add_summary(eval_loss_rb_sum, step)
                    print('Evaluation at step {0}: loss {1}, rebalanced loss {2}.'.format(step, eval_loss, eval_loss_rb))
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 < self.g_repeat:
                    checkpoint_path = os.path.join(
                        self.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
