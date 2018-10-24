from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import pickle
from skimage import io, color
import tensorflow as tf

from net import Net
from data_coco import DataSet
import time
from datetime import datetime
import os


_LOG_FREQ = 10


def scalar_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


class Solver_Sampler(object):

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
            self.kernel_zero = True if common_params['kernel_zero'] == '1' else False
            if self.prior_boost:
                print('Using prior boost.')
            else:
                print('Not using prior boost.')
            if self.with_caption:
                print('Training with captions.')
            else:
                print('Training without captions.')

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
            common_params=common_params, dataset_params=dataset_params, with_ab=True)
        self.val_dataset = DataSet(common_params=common_params, dataset_params=dataset_params, train=False, with_ab=True)

        print("Solver initialization done.")

    def construct_graph(self, scope, sess):
        with tf.device('/gpu:' + str(self.device_id)):
            self.data_l = tf.placeholder(
                tf.float32, (self.batch_size, self.height, self.width, 1))
            self.data_l_ss = tf.placeholder(
                tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 1))
            self.gt_ab = tf.placeholder(
                tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 2))
            self.captions = tf.placeholder(tf.int32, (self.batch_size, 20))
            self.lens = tf.placeholder(tf.int32, (self.batch_size))
            self.prior = tf.placeholder(
                tf.float32,
                (self.batch_size, int(self.height / 4), int(self.width / 4), 1)
            )

            # cc = np.load('resources/pts_in_hull.npy')
            # self.grid = tf.constant(cc, dtype=tf.float32)  # [313, 2]
            _, color_emb = self.net.inference1(self.data_l)
            self.pred_ab = self.net.sample_by_caption(self.captions, self.lens, self.data_l_ss, color_emb)

            huber_loss, total_loss, wd_loss = self.net.sample_loss(scope, self.gt_ab, self.pred_ab, self.prior)

            tf.summary.scalar('huber_loss', huber_loss)
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('weight_loss', wd_loss)
            print('Graph constructed.')

            return total_loss, huber_loss, wd_loss

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
                self.total_loss, self.huber_loss, self.wd_loss = self.construct_graph(scope, sess)
                self.summaries = tf.get_collection(
                    tf.GraphKeys.SUMMARIES, scope)

            self.summaries.append(
                tf.summary.scalar('learning_rate', learning_rate))

            opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=self.moment, beta2=0.99)
            grads = opt.compute_gradients(self.total_loss, var_list=tf.trainable_variables(scope='Sampler'))

            apply_gradient_op = opt.apply_gradients(
                grads, global_step=self.global_step)
            variable_averages = tf.train.ExponentialMovingAverage(
                0.999, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables(scope='Sampler'))
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10, keep_checkpoint_every_n_hours=1)
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
                data_l, data_l_ss, prior, captions, lens, gt_ab_ss = self.dataset.batch()
                if step % _LOG_FREQ == 0:
                    duration = time.time() - start_time
                    num_examples_per_step = self.batch_size * self.num_gpus * _LOG_FREQ
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / (self.num_gpus * _LOG_FREQ)

                    total_loss_value, huber_loss_value = sess.run([self.total_loss, self.huber_loss], feed_dict={
                        self.data_l: data_l, self.data_l_ss: data_l_ss, self.gt_ab: gt_ab_ss, self.captions: captions, self.lens: lens, self.prior: prior})
                    format_str = ('%s: step %d, total loss = %.2f, huber loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(),
                                         step, total_loss_value, huber_loss_value, examples_per_sec, sec_per_batch))
                    start_time = time.time()

                # Generator training.
                sess.run([train_op], feed_dict={self.data_l: data_l, self.data_l_ss: data_l_ss, self.gt_ab: gt_ab_ss, self.captions: captions, self.lens: lens, self.prior: prior})

                if step % 100 == 0:
                    summary_str = sess.run(summary_op, feed_dict={self.data_l: data_l, self.data_l_ss: data_l_ss, self.gt_ab: gt_ab_ss, self.captions: captions, self.lens: lens, self.prior: prior})
                    summary_writer.add_summary(summary_str, step)

                    # Evaluate 1000 images.
                    eval_total_loss = 0.0
                    eval_huber_loss = 0.0
                    eval_iters = 30
                    for _ in xrange(eval_iters):
                        val_data_l, val_data_l_ss, val_prior, val_captions, val_lens, val_gt_ab_ss = self.val_dataset.batch()
                        total_loss_value, huber_loss_value, pred_abs = sess.run([self.total_loss, self.huber_loss, self.pred_ab], feed_dict={
                            self.data_l: val_data_l, self.gt_ab: val_gt_ab_ss, self.data_l_ss: val_data_l_ss, self.captions: val_captions, self.lens: val_lens, self.prior: val_prior})
                        eval_total_loss += total_loss_value
                        eval_huber_loss += huber_loss_value
                    eval_total_loss /= eval_iters
                    eval_huber_loss /= eval_iters
                    eval_total_loss_sum = scalar_summary('eval_total_loss', eval_total_loss)
                    eval_huber_loss_sum = scalar_summary('eval_huber_loss', eval_huber_loss)
                    summary_writer.add_summary(eval_total_loss_sum, step)
                    summary_writer.add_summary(eval_huber_loss_sum, step)
                    print('Evaluation at step {0}: total loss {1}, huber loss {2}.'.format(step, eval_total_loss, eval_huber_loss))

                    # Save sample image
                    img_ab = pred_abs[0: 1]
                    print('1')
                    # img_ab = cv2.resize(img_ab_ss, (224, 224), interpolation=cv2.INTER_CUBIC)
                    print('11')
                    img_l = val_data_l_ss[0: 1]
                    print('111')
                    img_l += 1
                    img_l *= 50
                    print('1111')
                    img_lab = np.concatenate((img_l, img_ab), axis=-1).astype(np.float64)
                    print('11111')
                    img_rgb = color.lab2rgb(img_lab)
                    print('111111')
                    word_list = list(val_captions[0, :val_lens[0]])
                    print('1111111')
                    img_caption = '_'.join(vrev.get(w, 'unk') for w in word_list) 
                    io.imsave(os.path.join(self.train_dir, '{0}_{1}.jpg').format(step, img_caption), img_rgb)
                    print('11111111')

                # Save the model checkpoint periodically.
                if step % 1000 == 0:
                    checkpoint_path = os.path.join(
                        self.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            sess.close()
