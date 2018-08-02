from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from ops import *
from data import DataSet
import time
from datetime import datetime
import os
import sys

class Net(object):

    def __init__(self, train=True, common_params=None, net_params=None):
        self.train = train
        self.weight_decay = 0.0
        if common_params:
          gpu_nums = len(str(common_params['gpus']).split(','))
          self.batch_size = int(int(common_params['batch_size'])/gpu_nums)
        if net_params:
          self.weight_decay = float(net_params['weight_decay'])
          self.alpha = float(net_params['alpha'])

    def inference(self, data_l):
        with tf.variable_scope('G'):
            #conv1
            conv_num = 1

            temp_conv = conv2d('conv_{}'.format(conv_num), data_l, [3, 3, 1, 64], stride=1, wd=self.weight_decay)
            conv_num += 1

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 64], stride=2, wd=self.weight_decay)
            conv_num += 1

            #self.nilboy = temp_conv

            temp_conv = batch_norm('conv_{}'.format(conv_num), temp_conv,train=self.train)
            #conv2
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 128], stride=1, wd=self.weight_decay)
            conv_num += 1
            
            temp_conv = conv2d('conv_{}'.format(conv_num) + str(conv_num), temp_conv, [3, 3, 128, 128], stride=2, wd=self.weight_decay)
            conv_num += 1

            temp_conv = batch_norm('conv_{}'.format(conv_num), temp_conv,train=self.train)
            #conv3
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1    

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=2, wd=self.weight_decay)
            conv_num += 1

            temp_conv = batch_norm('bn_3', temp_conv, train=self.train)
            #conv4
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1

            
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1

            temp_conv = batch_norm('bn_4', temp_conv,train=self.train)

            #conv5
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1    

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1

            temp_conv = batch_norm('bn_5', temp_conv,train=self.train)
            #conv6
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1    

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1    

            temp_conv = batch_norm('bn_6', temp_conv,train=self.train)    
            #conv7
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1

            temp_conv = batch_norm('bn_7', temp_conv,train=self.train)
            #conv8
            temp_conv = deconv2d('conv_{}'.format(conv_num), temp_conv, [4, 4, 512, 256], stride=2, wd=self.weight_decay)
            conv_num += 1    

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1

            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1

            #Unary prediction
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1

        conv8_313 = temp_conv
        return conv8_313

    def loss(self, scope, conv8_313, prior_boost_nongray, gt_ab_313, D_pred):
        flat_conv8_313 = tf.reshape(conv8_313, [-1, 313])
        flat_gt_ab_313 = tf.reshape(gt_ab_313, [-1,313])
        g_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=flat_conv8_313, labels=flat_gt_ab_313)) / (self.batch_size)
        
        tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses', scope=scope)))
        #
        dl2c = tf.gradients(g_loss, conv8_313)
        dl2c = tf.stop_gradient(dl2c)
        #
        new_loss = tf.reduce_sum(dl2c * conv8_313 * prior_boost_nongray) + tf.add_n(tf.get_collection('losses', scope=scope))

        # Adversarial loss.
        adv_loss = -tf.reduce_sum(D_pred) / self.batch_size
        new_loss += self.alpha * adv_loss

        return new_loss, g_loss

    def discriminator(self, data_ab):
        '''
        Args:
            data_ab
        '''
        with tf.variable_scope('D'):
            data_ab = tf.stop_gradient(data_ab)
            original_shape = tf.shape(data_ab)

            conv_num = 1
            conv_1 = conv2d('d_conv_{}'.format(conv_num), data_ab, [4, 4, 2, 64], stride=1, wd=None)

            conv_num += 1
            conv_2 = conv2d('d_conv_{}'.format(conv_num), conv_1, [4, 4, 64, 128], stride=2, wd=None)

            conv_num += 1
            conv_3 = conv2d('d_conv_{}'.format(conv_num), conv_2, [4, 4, 128, 256], stride=2, wd=None)
            
            conv_num += 1
            conv_4 = conv2d('d_conv_{}'.format(conv_num), conv_3, [4, 4, 256, 1], stride=1, relu=False, wd=None)
            
            upsampled_output = tf.image.resize_images(conv_4, original_shape[1:3], method=ResizeMethod.NEAREST_NEIGHBOR)

        return upsampled_output


    def discriminator_loss(self, original, colorized):
        original_loss = -tf.log(original)
        colorized_loss = -tf.log(1 - colorized)
        total_loss = tf.reduce_sum(original_loss + colorized_loss) / self.batch_size
        # tf.summary.scalar('D_weight_loss', tf.add_n(tf.get_collection('losses', scope=scope)))
        # total_loss += tf.add_n(tf.get_collection('losses', scope=scope))

        return total_loss

    def conv313_to_ab(self, conv8_313, rebalance=2.63):
        '''
        conv 313 to ab.
        Return: []
        '''
        enc_dir = './resources'
        cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
        cc = tf.constant(cc, dtype=tf.float32)  # [313, 2]
        # cc = tf.expand_dims(cc, 0) 
        # conv8_313 = conv8_313[0, :, :, :]
        conv8_313_rh = conv8_313 * rebalance
        class8_313_rh = tf.nn.softmax(conv8_313_rh, axis=-1)  # [N, H/4, W/4, 313]
        shape = tf.shape(class8_313_rh)
        class8_313_rh = tf.reshape(class8_313_rh, (-1, 313))  # [N*H*W/16, 313]
        data_ab = tf.matmul(class8_313_rh, cc)  # [N*H*W/16, 2]
        data_ab = tf.reshape(data_ab, shape[0:3] + [2])  # [N, H/4, W/4, 2]

        return data_ab


