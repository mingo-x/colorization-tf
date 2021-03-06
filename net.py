from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from ops import *
import os
import pickle


class Net(object):
    def __init__(self, train=True, g_version=0, batch_size=1, use_vg=False, common_params=None, net_params=None):
        self.train = train
        self.weight_decay = 0.0
        self.eps = 1e-8
        self.lstm_hid_dim = 256
        self.in_dims = [64, 128, 256, 512, 512, 512, 512, 256]
        self.batch_size = batch_size
        self.use_vg = use_vg

        if common_params:
            gpu_nums = len(str(common_params['gpus']).split(','))
            self.batch_size = int(int(common_params['batch_size']) / gpu_nums)
            self.is_rgb = True if common_params['is_rgb'] == '1' else False
            self.output_dim = 3 if self.is_rgb else 2
            self.use_vg = common_params['use_vg'] == '1' if 'use_vg' in common_params else False

        if self.use_vg:
            self.word_embedding = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/glove.6B.100d_emb.p', 'r'))
        else:
            self.word_embedding = pickle.load(open('/srv/glusterfs/xieya/data/w2v_embeddings_colors.p', 'r'))

        if net_params:
            self.weight_decay = float(net_params['weight_decay'])
            self.alpha = float(net_params['alpha'])
            print('Adversarial weight {}'.format(self.alpha))
            self.g_version = int(net_params['g_version'])
            print('Generator version {}'.format(self.g_version))
            self.version = int(net_params['version'])
            print('Discriminator version {}'.format(self.version))
            self.temp_trainable = True if net_params['temp_trainable'] == '1' else False
            self.gp_lambda = float(net_params['gp_lambda'])
            print('Gradient penalty {}.'.format(self.gp_lambda))
            self.k = float(net_params['k'])
            print('Gradient norm {}.'.format(self.k))
            self.unet = net_params['unet'] == '1'
        else:
            self.g_version = g_version

    def inference(self, data_l):
        if self.g_version == 0:
            return self.inference0(data_l)
        elif self.g_version == 1:
            return self.inference1(data_l)
        elif self.g_version == 2:
            return self.inference2(data_l)
        elif self.g_version == 3:
            return self.inference3(data_l)   

    def inference0(self, data_l):
        output = data_l * 50
        with tf.variable_scope('G'):
            # conv1
            conv_num = 1
            temp_conv = conv2d('conv_{}'.format(conv_num), output, [3, 3, 1, 64], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 64], stride=2, wd=self.weight_decay)
            temp_conv = batch_norm('bn_1', temp_conv, train=self.train)

            # conv2
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 128], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 128], stride=2, wd=self.weight_decay)
            temp_conv = batch_norm('bn_2'.format(conv_num), temp_conv, train=self.train)

            # conv3
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=2, wd=self.weight_decay, relu=False)
            conv3 = temp_conv
            temp_conv = tf.nn.relu(temp_conv)
            temp_conv = batch_norm('bn_3', temp_conv, train=self.train)

            # conv4
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            temp_conv = batch_norm('bn_4', temp_conv, train=self.train)

            # conv5
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            temp_conv = batch_norm('bn_5', temp_conv, train=self.train)

            # conv6
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)    
            temp_conv = batch_norm('bn_6', temp_conv, train=self.train)

            # conv7
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            temp_conv = batch_norm('bn_7', temp_conv, train=self.train)

            # conv8
            conv_num += 1
            temp_conv = deconv2d('conv_{}'.format(conv_num), temp_conv, [4, 4, 512, 256], stride=2, wd=self.weight_decay, relu=False)
            if self.unet:
                conv3 = deconv2d('deconv_3', conv3, [4, 4, 256, 256], stride=2, wd=self.weight_decay, relu=False)
                temp_conv = tf.concat((conv3, temp_conv), axis=-1)
            temp_conv = tf.nn.relu(temp_conv)
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512 if self.unet else 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1

            # Unary prediction
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1

        conv8_313 = temp_conv
        return conv8_313

    def inference1(self, data_l):
        with tf.variable_scope('G'):
            # conv1
            conv_num = 1
            batch_num = 1
            # 176x176
            temp_conv = conv2d('conv_{}'.format(conv_num), data_l, [3, 3, 1, 64], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 64], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv2
            # 88x88
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 128], stride=1, wd=self.weight_decay)
            conv_num += 1
            # 88x88
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 128], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv3
            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1 
            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv4
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 512], stride=1, wd=self.weight_decay)
            conv_num += 1            
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv5
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv6
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1  
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
            conv_num += 1
            batch_num += 1    
 
            # conv7
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
            conv_num += 1
            batch_num += 1   

            # conv8
            # 22x22
            temp_conv = deconv2d('conv_{}'.format(conv_num), temp_conv, [4, 4, 512, 256], stride=2, wd=self.weight_decay)
            conv_num += 1      
            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
            conv_num += 1
            batch_num += 1  

            # Unary prediction
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1

        conv8_313 = temp_conv
        return conv8_313

    def inference2(self, data_l):
        ''' U-net'''
        with tf.variable_scope('G'):
            # conv1
            conv_num = 1
            batch_num = 1

            # 176x176
            temp_conv = conv2d('conv_{}'.format(conv_num), data_l, [4, 4, 1, 64], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 176x176
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 64], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv2
            # 88x88
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 128], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            
            # 88x88
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 128], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            conv2 = tf.nn.leaky_relu(temp_conv)
            temp_conv = conv2
            conv_num += 1
            batch_num += 1

            # conv3
            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 256], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            
            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1 

            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            conv3 = tf.nn.leaky_relu(temp_conv)
            temp_conv = temp_conv
            conv_num += 1
            batch_num += 1

            # conv4
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            conv4 = tf.nn.leaky_relu(temp_conv)
            temp_conv = conv4
            conv_num += 1
            batch_num += 1

            # conv5
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1 

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv6
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1  

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            conv6 = tf.nn.leaky_relu(temp_conv)
            temp_conv = conv6
            conv_num += 1
            batch_num += 1    

            # 22x22x1024
            # temp_conv = tf.concat((conv6, conv4), axis=-1)
 
            # conv7
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1   

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 256], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            conv7 = tf.nn.leaky_relu(temp_conv)
            temp_conv = conv7
            conv_num += 1
            batch_num += 1  

            # 22x22x512
            temp_conv = tf.concat((conv7, conv3), axis=-1) 

            # conv8
            # 22x22
            temp_conv = deconv2d('conv_{}'.format(conv_num), temp_conv, [4, 4, 512, 256], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1     

            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 128], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            conv8 = tf.nn.leaky_relu(temp_conv)
            temp_conv = conv8
            conv_num += 1
            batch_num += 1   

            # 44x44x256
            temp_conv = tf.concat((conv8, conv2), axis=-1)

            # Unary prediction
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1

        conv8_313 = temp_conv
        return conv8_313

    def inference3(self, data_l):
        with tf.variable_scope('G'):
            # conv1
            conv_num = 1
            batch_num = 1

            # 176x176
            temp_conv = conv2d('conv_{}'.format(conv_num), data_l, [3, 3, 1, 64], stride=1, relu=False, wd=self.weight_decay)
            # temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            # batch_num += 1

            # 176x176
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 64], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv2
            # 88x88
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 128], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            
            # 88x88
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 128], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv3
            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 256], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            
            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1 

            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv4
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv5
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1

            # conv6
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1    
 
            # conv7
            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1 

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 22x22
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1   

            # conv8
            # 22x22
            temp_conv = deconv2d('conv_{}'.format(conv_num), temp_conv, [4, 4, 512, 256], stride=2, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1      

            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1

            # 44x44
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, relu=False, wd=self.weight_decay)
            temp_conv = batch_norm('bn_{}'.format(batch_num), temp_conv, train=self.train)
            temp_conv = tf.nn.leaky_relu(temp_conv)
            conv_num += 1
            batch_num += 1   

            # Unary prediction
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1

        conv8_313 = temp_conv
        return conv8_313

    def inference4(self, data_l, captions, lens, biases=None, kernel_initializer=None, with_attention=False, lstm_version=0):
        caption_feature = self.caption_encoding(lstm_version, captions, lens)
        # caption_feature = tf.zeros_like(caption_feature)
        with tf.variable_scope('Film'):
            gammas = []
            betas = []
            for i in range(8):
                if biases is not None and biases[i] is None:
                    gammas.append(None)
                    betas.append(None)
                else:
                    if biases is None:
                        initializer = tf.zeros_initializer(dtype=tf.float32)
                    else:
                        initializer = tf.constant_initializer(biases[i], dtype=tf.float32)
                    dense = Linear('dense_{}'.format(i), caption_feature, self.in_dims[i] * 2, initializer, kernel_initializer)
                    gamma, beta = tf.split(dense, 2, axis=-1)
                    gammas.append(gamma)
                    betas.append(beta)

        with tf.variable_scope('G'):
            # conv1
            block_idx = 0
            conv_num = 1
            temp_conv = conv2d('conv_{}'.format(conv_num), data_l, [3, 3, 1, 64], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 64], stride=2, relu=False, wd=self.weight_decay)
            conv_num += 1
            if gammas[block_idx] is None:
                temp_conv = batch_norm('bn_1', temp_conv, train=self.train)
            else:
                temp_conv = bn('bn_1', temp_conv, train=self.train)
                temp_conv = gammas[block_idx][:, tf.newaxis, tf.newaxis, :] * temp_conv + betas[block_idx][:, tf.newaxis, tf.newaxis, :]
            temp_conv = tf.nn.relu(temp_conv)
            
            # conv2
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 128], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 128], stride=2, relu=False, wd=self.weight_decay)
            conv_num += 1
            if gammas[block_idx] is None:
                temp_conv = batch_norm('bn_2', temp_conv, train=self.train)
            else:
                temp_conv = bn('bn_2', temp_conv, train=self.train)
                temp_conv = gammas[block_idx][:, tf.newaxis, tf.newaxis, :] * temp_conv + betas[block_idx][:, tf.newaxis, tf.newaxis, :]
            temp_conv = tf.nn.relu(temp_conv)

            # conv3
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=2, relu=False, wd=self.weight_decay)
            conv_num += 1
            if gammas[block_idx] is None:
                temp_conv = batch_norm('bn_3', temp_conv, train=self.train)
            else:
                temp_conv = bn('bn_3', temp_conv, train=self.train)
                temp_conv = gammas[block_idx][:, tf.newaxis, tf.newaxis, :] * temp_conv + betas[block_idx][:, tf.newaxis, tf.newaxis, :]
            temp_conv = tf.nn.relu(temp_conv)

            # conv4
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1
            if gammas[block_idx] is None:
                temp_conv = batch_norm('bn_4', temp_conv, train=self.train)
            else:
                temp_conv = bn('bn_4', temp_conv, train=self.train)
                temp_conv = gammas[block_idx][:, tf.newaxis, tf.newaxis, :] * temp_conv + betas[block_idx][:, tf.newaxis, tf.newaxis, :]
            temp_conv = tf.nn.relu(temp_conv)

            # conv5
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, dilation=2, wd=self.weight_decay)
            conv_num += 1
            if gammas[block_idx] is None:
                temp_conv = batch_norm('bn_5', temp_conv, train=self.train)
            else:
                temp_conv = bn('bn_5', temp_conv, train=self.train)
                temp_conv = gammas[block_idx][:, tf.newaxis, tf.newaxis, :] * temp_conv + betas[block_idx][:, tf.newaxis, tf.newaxis, :]
            temp_conv = tf.nn.relu(temp_conv)

            # conv6
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, dilation=2, wd=self.weight_decay)
            conv_num += 1  
            if gammas[block_idx] is None:
                temp_conv = batch_norm('bn_6', temp_conv, train=self.train)
            else:  
                temp_conv = bn('bn_6', temp_conv, train=self.train)    
                temp_conv = gammas[block_idx][:, tf.newaxis, tf.newaxis, :] * temp_conv + betas[block_idx][:, tf.newaxis, tf.newaxis, :]
            temp_conv = tf.nn.relu(temp_conv)
            conv_6 = temp_conv

        if with_attention:
            print('Using attention map.')
            attention_map = self.attention_block(conv_6, caption_feature)

        with tf.variable_scope('G'):

            # conv7
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1
            if gammas[block_idx] is None:
                temp_conv = batch_norm('bn_7', temp_conv, train=self.train)
            else:
                temp_conv = bn('bn_7', temp_conv, train=self.train)
                if with_attention:
                      temp_conv = attention_map * (
                        gammas[block_idx][:, tf.newaxis, tf.newaxis, :] * temp_conv + betas[block_idx][:, tf.newaxis, tf.newaxis, :]) + (
                        1 - attention_map) * temp_conv
                else:
                      temp_conv = gammas[block_idx][:, tf.newaxis, tf.newaxis, :] * temp_conv + betas[block_idx][:, tf.newaxis, tf.newaxis, :]

            temp_conv = tf.nn.relu(temp_conv)

            # conv8
            block_idx += 1
            temp_conv = deconv2d('conv_{}'.format(conv_num), temp_conv, [4, 4, 512, 256], stride=2, wd=self.weight_decay)
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1
            if gammas[block_idx] is None:
                temp_conv = batch_norm('bn_8', temp_conv, train=self.train)
            else:
                temp_conv = bn('bn_8', temp_conv, train=self.train)
                temp_conv = gammas[block_idx][:, tf.newaxis, tf.newaxis, :] * temp_conv + betas[block_idx][:, tf.newaxis, tf.newaxis, :]
            temp_conv = tf.nn.relu(temp_conv)

            # Unary prediction
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1

        conv8_313 = temp_conv
        return conv8_313

    def inference5(self, data_l, captions, lens, cap_layers=[0, 1, 2, 3, 4, 5, 6, 7], same_lstm=True, res=False, paper=False, lstm_version=0):
        '''Concat.'''
        if same_lstm:
            caption_features = {-1: self.caption_encoding(lstm_version, captions, lens)}
        else:
            caption_features = {}
            for i in cap_layers:
                caption_features[i] = self.caption_encoding(lstm_version, captions, lens, i) 

        with tf.variable_scope('G'):
            # conv0
            block_idx = 0
            conv_num = 1
            temp_conv = conv2d('conv_{}'.format(conv_num), data_l, [3, 3, 1, 64], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 64], stride=2, relu=False, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_1', temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
            
            # conv1
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 64, 128], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 128], stride=2, relu=False, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_2', temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)

            # conv2
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 128, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=2, relu=False, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_3', temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)

            # conv3
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            conv3 = temp_conv
            conv_num += 1
            temp_conv = batch_norm('bn_4', temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)

        with tf.variable_scope('Concat'):
            add_on = None
            if block_idx in cap_layers:
                print("Concat at block {}.".format(block_idx))
                shape = tf.shape(temp_conv)
                caption_feature = caption_features[-1] if same_lstm else caption_features[block_idx]
                cap_emb_expand = caption_feature[:, tf.newaxis, tf.newaxis, :]
                cap_emb_expand = tf.tile(cap_emb_expand, (1, shape[1], shape[2], 1))  # NxHxWx512
                concat_block = tf.concat((temp_conv, cap_emb_expand), axis=-1)  # NxHxWx1024
                concat_block = conv2d('conv{}'.format(block_idx), concat_block, [3, 3, concat_block.get_shape()[3], 512], stride=1, wd=self.weight_decay)
                if res:
                    add_on = concat_block
                else:
                    temp_conv = concat_block

        with tf.variable_scope('G'):
            # conv4
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            if add_on is not None:
                temp_conv = temp_conv + add_on
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_5', temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)

        with tf.variable_scope('Concat'):
            add_on = None
            if block_idx in cap_layers:
                print("Concat at block {}.".format(block_idx))
                shape = tf.shape(temp_conv)
                caption_feature = caption_features[-1] if same_lstm else caption_features[block_idx]
                cap_emb_expand = caption_feature[:, tf.newaxis, tf.newaxis, :]
                cap_emb_expand = tf.tile(cap_emb_expand, (1, shape[1], shape[2], 1))  # NxHxWx512
                concat_block = tf.concat((temp_conv, cap_emb_expand), axis=-1)  # NxHxWx1024
                concat_block = conv2d('conv{}'.format(block_idx), concat_block, [3, 3, concat_block.get_shape()[3], 512], stride=1, wd=self.weight_decay)
                if res:
                    add_on = concat_block
                else:
                    temp_conv = concat_block

        with tf.variable_scope('G'):
            # conv5
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            if add_on is not None:
                temp_conv = temp_conv + add_on
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, dilation=2, wd=self.weight_decay)
            conv_num += 1  
            temp_conv = batch_norm('bn_6', temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
       
        with tf.variable_scope('Concat'):
            add_on = None
            if block_idx in cap_layers:
                print("Concat at block {}.".format(block_idx))
                shape = tf.shape(temp_conv)
                caption_feature = caption_features[-1] if same_lstm else caption_features[block_idx]
                cap_emb_expand = caption_feature[:, tf.newaxis, tf.newaxis, :]
                cap_emb_expand = tf.tile(cap_emb_expand, (1, shape[1], shape[2], 1))  # NxHxWx512
                concat_block = tf.concat((temp_conv, cap_emb_expand), axis=-1)  # NxHxWx1024
                concat_block = conv2d('conv{}'.format(block_idx), concat_block, [3, 3, concat_block.get_shape()[3], 512], stride=1, wd=self.weight_decay)
                if res:
                    add_on = concat_block
                else:
                    temp_conv = concat_block

        with tf.variable_scope('G'):
            # conv6
            block_idx += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            if add_on is not None:
                temp_conv = temp_conv + add_on
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 512, 512], stride=1, relu=False, wd=self.weight_decay)
            conv_6 = temp_conv
            conv_num += 1
            temp_conv = batch_norm('bn_7', temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)
      
        with tf.variable_scope('Concat'):
            add_on = None
            if block_idx in cap_layers:
                print("Concat at block {}.".format(block_idx))
                shape = tf.shape(temp_conv)
                caption_feature = caption_features[-1] if same_lstm else caption_features[block_idx]
                cap_emb_expand = caption_feature[:, tf.newaxis, tf.newaxis, :]
                cap_emb_expand = tf.tile(cap_emb_expand, (1, shape[1], shape[2], 1))  # NxHxWx512
                if paper:
                    print("Imitate structure from paper.")
                    im_features = tf.nn.l2_normalize(tf.concat((conv3, conv_6), axis=-1), 3)
                    lan_features = tf.nn.l2_normalize(cap_emb_expand, 3)
                    concat_block = tf.concat((im_features, lan_features), axis=-1)  # NxHxWx1024
                    concat_block = conv2d('conv{}_1'.format(block_idx), concat_block, [1, 1, concat_block.get_shape()[3], 512], stride=1, wd=self.weight_decay)
                    concat_block = conv2d('conv{}_2'.format(block_idx), concat_block, [1, 1, 512, 512], stride=1, wd=self.weight_decay)
                else:
                    concat_block = tf.concat((temp_conv, cap_emb_expand), axis=-1)  # NxHxWx1024
                    concat_block = conv2d('conv{}'.format(block_idx), concat_block, [3, 3, concat_block.get_shape()[3], 512], stride=1, wd=self.weight_decay)
                if res:
                    add_on = concat_block
                else:
                    temp_conv = concat_block

        with tf.variable_scope('G'):
            # conv7
            block_idx += 1
            temp_conv = deconv2d('conv_{}'.format(conv_num), temp_conv, [4, 4, 512, 256], stride=2, wd=self.weight_decay)
            if add_on is not None:
                temp_conv = temp_conv + add_on
            conv_num += 1    
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
            conv_num += 1
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [3, 3, 256, 256], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1
            temp_conv = batch_norm('bn_8', temp_conv, train=self.train)
            temp_conv = tf.nn.relu(temp_conv)

            # Unary prediction
            temp_conv = conv2d('conv_{}'.format(conv_num), temp_conv, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)
            conv_num += 1

        conv8_313 = temp_conv
        return conv8_313

    def GAN_G(self, noise=None):
        dim = 64
        with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
            if noise is None:
                noise = tf.random_normal([self.batch_size, 128])

            output = Linear('dense1', noise, 4 * 4 * 8 * dim)
            output = tf.reshape(output, [-1, 4, 4, 8 * dim])

            output = ResidualBlock('Generator.Res1', 8 * dim, 8 * dim, 3, output, resample='up', train=self.train)
            output = ResidualBlock('Generator.Res2', 8 * dim, 4 * dim, 3, output, resample='up', train=self.train)
            output = ResidualBlock('Generator.Res3', 4 * dim, 2 * dim, 3, output, resample='up', train=self.train)
            output = ResidualBlock('Generator.Res4', 2 * dim, 1 * dim, 3, output, resample='up', train=self.train)
            output = Normalize('Generator.OutputN', output, train=self.train)
            output = tf.nn.relu(output)
            output = conv2d('Generator.Output', output, [3, 3, 1 * dim, self.output_dim], relu=False, wd=self.weight_decay)
            output = tf.tanh(output)

        return output

    def GAN_D(self, inputs):
        dim = 64
        with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
            output = inputs
            output = conv2d('Discriminator.Input', output, [3, 3, self.output_dim, dim], relu=False, wd=self.weight_decay)

            output = ResidualBlock('Discriminator.Res1', dim, 2 * dim, 3, output, resample='down', train=self.train)
            output = ResidualBlock('Discriminator.Res2', 2 * dim, 4 * dim, 3, output, resample='down', train=self.train)
            output = ResidualBlock('Discriminator.Res3', 4 * dim, 8 * dim, 3, output, resample='down', train=self.train)
            output = ResidualBlock('Discriminator.Res4', 8 * dim, 8 * dim, 3, output, resample='down', train=self.train)

            output = Linear('Discriminator.Output', output, 1)

        return tf.reshape(output, [-1])

    def GAN_loss(self, real_data, fake_data):
        disc_real = self.GAN_D(real_data)
        disc_fake = self.GAN_D(fake_data)

        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        w_dist = -disc_cost

        alpha = tf.random_uniform(
            shape=[self.batch_size, 1, 1, 1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)
        gradients = tf.gradients(self.GAN_D(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
        disc_cost += 10. * gradient_penalty

        return gen_cost, disc_cost, w_dist, tf.reduce_mean(slopes)

    def loss(self, scope, conv8_313, prior_boost_nongray,
             gt_ab_313, cap_prior=None, bbox=None):
        flat_conv8_313 = tf.reshape(conv8_313, [-1, 313])
        flat_gt_ab_313 = tf.reshape(gt_ab_313, [-1, 313])
        flat_gt_ab_313 = tf.stop_gradient(flat_gt_ab_313)
        g_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_conv8_313, labels=flat_gt_ab_313)

        dl2c = tf.gradients(g_loss, conv8_313)
        dl2c = tf.stop_gradient(dl2c)

        g_loss = tf.reshape(g_loss, tf.shape(prior_boost_nongray))
        rb_loss = g_loss * prior_boost_nongray
        g_loss = tf.reduce_mean(g_loss)
        rb_loss = tf.reduce_sum(rb_loss) / tf.reduce_sum(prior_boost_nongray)

        wd_loss = tf.add_n(tf.get_collection('losses', scope=scope))
        l2_loss = tf.get_variable('regularization', (), initializer=tf.zeros_initializer, dtype=tf.float32)
        for var in tf.trainable_variables():
            if 'kernel' in var.name:
                l2_loss = l2_loss + tf.nn.l2_loss(var)
        l2_loss = l2_loss * self.weight_decay
        wd_loss += l2_loss 

        new_loss = dl2c * conv8_313 * prior_boost_nongray
        if cap_prior is not None:
            new_loss = new_loss * cap_prior[:, tf.newaxis, tf.newaxis, tf.newaxis]
        if bbox is not None:
            print('Rebalancing loss with bbox.')
            new_loss = new_loss * bbox
            # new_loss = new_loss / tf.reduce_sum(bbox, axis=(1, 2, 3), keepdims=True)
        new_loss = tf.reduce_sum(new_loss)
        new_loss = new_loss + wd_loss

        return new_loss, g_loss, wd_loss, rb_loss

    def discriminator(self, data_313, reuse=False):
        '''
        Args:
            data_lab
        '''
        with tf.variable_scope('D', reuse=reuse):
            if self.version == 0:
                self.downscale = 16
                # 44x44
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [4, 4, 314, 128], stride=2, wd=None)

                # 22x22
                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), conv_1, [4, 4, 128, 64], stride=2, wd=None)

                # 11x11
                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), conv_2, [4, 4, 64, 1], stride=1, relu=False, wd=None, sigmoid=True)

                discriminator = conv_3
            elif self.version == 2:
                self.downscale = 16
                # 44x44x64
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [4, 4, 3, 64], stride=1, relu=False, wd=None, leaky=True)
                # 22x22x128
                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), conv_1, [4, 4, 64, 128], stride=2, relu=False, wd=None)
                bn_1 = batch_norm('bn_1', conv_2, train=self.train)
                conv_2 = tf.nn.leaky_relu(bn_1)
                # 11x11x256
                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), conv_2, [4, 4, 128, 256], stride=2, relu=False, wd=None)
                bn_2 = batch_norm('bn_2', conv_3, train=self.train)
                conv_3 = tf.nn.leaky_relu(bn_2)
                # 11x11x1
                conv_num += 1
                conv_4 = conv2d('d_conv_{}'.format(conv_num), conv_3, [4, 4, 256, 1], stride=1, relu=False, wd=None, sigmoid=True)
                
                discriminator = conv_4
            elif self.version == 3:
                self.downscale = (176. * 176.) / (11. * 11.)
                # 176x176x64
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [4, 4, 3, 64], stride=1, relu=False, wd=None, leaky=True)
                # 88x88x64
                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), conv_1, [4, 4, 64, 64], stride=2, relu=False, wd=None)
                bn_1 = batch_norm('bn_1', conv_2, train=self.train)
                conv_2 = tf.nn.leaky_relu(bn_1)
                # 44x44x128
                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), conv_2, [4, 4, 64, 128], stride=2, relu=False, wd=None)
                bn_2 = batch_norm('bn_2', conv_3, train=self.train)
                conv_3 = tf.nn.leaky_relu(bn_2)
                # 22x22x128
                conv_num += 1
                conv_4 = conv2d('d_conv_{}'.format(conv_num), conv_3, [3, 3, 128, 128], stride=2, relu=False, wd=None)
                bn_3 = batch_norm('bn_3', conv_4, train=self.train)
                conv_4 = tf.nn.leaky_relu(bn_3)
                # 11x11x256
                conv_num += 1
                conv_5 = conv2d('d_conv_{}'.format(conv_num), conv_4, [3, 3, 128, 256], stride=2, relu=False, wd=None)
                bn_4 = batch_norm('bn_4', conv_5, train=self.train)
                conv_5 = tf.nn.leaky_relu(bn_4)
                # 11x11x1
                conv_num += 1
                conv_6 = conv2d('d_conv_{}'.format(conv_num), conv_5, [3, 3, 256, 1], stride=1, relu=False, wd=None, sigmoid=True)
                
                discriminator = conv_6
            elif self.version == 5:
                self.downscale = (176. * 176.) / (11. * 11.)
                # 176x176x64
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [4, 4, 3, 64], stride=1, relu=False, wd=None, leaky=True)
                # 88x88x64
                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), conv_1, [4, 4, 64, 64], stride=2, relu=False, wd=None)
                bn_1 = batch_norm('bn_1', conv_2, train=self.train)
                conv_2 = tf.nn.leaky_relu(bn_1)
                # 44x44x64
                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), conv_2, [4, 4, 64, 64], stride=2, relu=False, wd=None)
                bn_2 = batch_norm('bn_2', conv_3, train=self.train)
                conv_3 = tf.nn.leaky_relu(bn_2)
                # 22x22x128
                conv_num += 1
                conv_4 = conv2d('d_conv_{}'.format(conv_num), conv_3, [3, 3, 64, 128], stride=2, relu=False, wd=None)
                bn_3 = batch_norm('bn_3', conv_4, train=self.train)
                conv_4 = tf.nn.leaky_relu(bn_3)
                # 11x11x128
                conv_num += 1
                conv_5 = conv2d('d_conv_{}'.format(conv_num), conv_4, [3, 3, 128, 128], stride=2, relu=False, wd=None)
                bn_4 = batch_norm('bn_4', conv_5, train=self.train)
                conv_5 = tf.nn.leaky_relu(bn_4)
                # 11x11x1
                conv_num += 1
                conv_6 = conv2d('d_conv_{}'.format(conv_num), conv_5, [3, 3, 128, 1], stride=1, relu=False, wd=None, sigmoid=True)
                
                discriminator = conv_6
            elif self.version == 4:
                # 44x44x128
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [4, 4, 314, 128], stride=1, relu=False, wd=None, leaky=True)
                # 22x22x64
                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), conv_1, [4, 4, 128, 64], stride=2, relu=False, wd=None)
                bn_1 = batch_norm('bn_1', conv_2, train=self.train)
                conv_2 = tf.nn.leaky_relu(bn_1)
                # 11x11x32
                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), conv_2, [4, 4, 64, 32], stride=2, relu=False, wd=None)
                bn_2 = batch_norm('bn_2', conv_3, train=self.train)
                conv_3 = tf.nn.leaky_relu(bn_2)
                # 11x11x1
                conv_num += 1
                conv_4 = conv2d('d_conv_{}'.format(conv_num), conv_3, [4, 4, 32, 1], stride=1, relu=False, wd=None, sigmoid=True)
                
                discriminator = conv_4
            elif self.version == 6:
                # 44x44x314
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [3, 3, 314, 128], stride=1, relu=False, wd=None)
                conv_1 = tf.nn.leaky_relu(conv_1)
                # 22x22x256
                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), conv_1, [3, 3, 128, 256], stride=2, relu=False, wd=None)
                conv_2 = tf.nn.leaky_relu(conv_2)
                # 11x11x512
                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), conv_2, [3, 3, 256, 512], stride=2, relu=False, wd=None)
                conv_3 = tf.nn.leaky_relu(conv_3)
                # 5x5x512
                conv_num += 1
                conv_4 = conv2d('d_conv_{}'.format(conv_num), conv_3, [3, 3, 512, 512], stride=2, relu=False, wd=None, same=False)
                conv_4 = tf.nn.leaky_relu(conv_4)
                # 2x2x512
                conv_num += 1
                conv_5 = conv2d('d_conv_{}'.format(conv_num), conv_4, [3, 3, 512, 512], stride=2, relu=False, wd=None, same=False)
                conv_5 = tf.nn.leaky_relu(conv_5)

                flatten = tf.layers.flatten(conv_5)
                discriminator = Linear('dense', flatten, 1)
            elif self.version == 7:
                # 88x88x64
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [3, 3, 3, 64], stride=2, wd=None)
                # 44x44x128
                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), conv_1, [3, 3, 64, 128], stride=2, wd=None)
                # 22x22x256
                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), conv_2, [3, 3, 128, 256], stride=2, wd=None)
                # 11x11x512
                conv_num += 1
                conv_4 = conv2d('d_conv_{}'.format(conv_num), conv_3, [3, 3, 256, 512], stride=2, wd=None)
                # 5x5x512
                conv_num += 1
                conv_5 = conv2d('d_conv_{}'.format(conv_num), conv_4, [3, 3, 512, 512], stride=2, wd=None, same=False)
                # 2x2x512
                conv_num += 1
                conv_6 = conv2d('d_conv_{}'.format(conv_num), conv_5, [3, 3, 512, 512], stride=2, wd=None, same=False)

                flatten = tf.layers.flatten(conv_6)
                discriminator = Linear('dense', flatten, 1)
            elif self.version == 8:
                # 176x176
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [1, 1, 3, 32], stride=1, wd=None)
                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), conv_1, [3, 3, 32, 32], stride=1, wd=None)
                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), conv_2, [3, 3, 32, 64], stride=2, wd=None)
                # 88x88
                conv_num += 1
                conv_4 = conv2d('d_conv_{}'.format(conv_num), conv_3, [3, 3, 64, 64], stride=1, wd=None)
                conv_num += 1
                conv_5 = conv2d('d_conv_{}'.format(conv_num), conv_4, [3, 3, 64, 128], stride=2, wd=None)
                # 44x44
                conv_num += 1
                conv_6 = conv2d('d_conv_{}'.format(conv_num), conv_5, [3, 3, 128, 128], stride=1, wd=None)
                conv_num += 1
                conv_7 = conv2d('d_conv_{}'.format(conv_num), conv_6, [3, 3, 128, 256], stride=2, wd=None)
                # 22x22
                conv_num += 1
                conv_8 = conv2d('d_conv_{}'.format(conv_num), conv_7, [3, 3, 256, 256], stride=1, wd=None)
                conv_num += 1
                conv_9 = conv2d('d_conv_{}'.format(conv_num), conv_8, [3, 3, 256, 512], stride=2, wd=None)
                # 11x11
                conv_num += 1
                conv_10 = conv2d('d_conv_{}'.format(conv_num), conv_9, [3, 3, 512, 512], stride=1, wd=None)
                conv_num += 1
                conv_11 = conv2d('d_conv_{}'.format(conv_num), conv_10, [3, 3, 512, 512], stride=2, wd=None, same=False)
                # 5x5
                conv_num += 1
                conv_12 = conv2d('d_conv_{}'.format(conv_num), conv_11, [3, 3, 512, 512], stride=1, wd=None)
                conv_num += 1
                conv_13 = conv2d('d_conv_{}'.format(conv_num), conv_12, [3, 3, 512, 512], stride=2, wd=None, same=False)
                # 2x2
                flatten = tf.layers.flatten(conv_13)
                discriminator = tf.layers.dense(flatten, 1, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True, dtype=tf.float32))
            elif self.version == 9:
                # 44x44x314
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [3, 3, 314, 128], stride=1, relu=False, wd=None)
                conv_1 = tf.nn.leaky_relu(conv_1)
                # 22x22x256
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), conv_1, [3, 3, 128, 256], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)
                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 256, 256], stride=2, relu=False, wd=None)
                conv_2 = tf.nn.leaky_relu(conv_2)
                # 11x11x512
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), conv_2, [3, 3, 256, 512], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)
                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 512, 512], stride=2, relu=False, wd=None)
                conv_3 = tf.nn.leaky_relu(conv_3)
                # 5x5x512
                conv_num += 1
                conv_4 = conv2d('d_conv_{}'.format(conv_num), conv_3, [3, 3, 512, 512], stride=2, relu=False, wd=None, same=False)
                conv_4 = tf.nn.leaky_relu(conv_4)
                # 2x2x512
                conv_5 = conv2d('d_conv_{}'.format(conv_num), conv_4, [3, 3, 512, 512], stride=2, relu=False, wd=None, same=False)
                conv_5 = tf.nn.leaky_relu(conv_5)

                flatten = tf.layers.flatten(conv_5)
                discriminator = Linear('dense', flatten, 1)

            elif self.version == 11:
                # 44x44x314
                conv_num = 1
                output = conv2d('d_conv_{}'.format(conv_num), data_313, [3, 3, 314, 128], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 128, 128], stride=2, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                # 22x22x128
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 128, 256], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                conv_num += 1
                conv_2 = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 256, 256], stride=2, relu=False, wd=None)
                conv_2 = tf.nn.leaky_relu(conv_2)

                # 11x11x256
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), conv_2, [3, 3, 256, 512], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                conv_num += 1
                conv_3 = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 512, 512], stride=2, relu=False, wd=None)
                conv_3 = tf.nn.leaky_relu(conv_3)
                
                # 5x5x512
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), conv_3, [1, 1, 512, 1], stride=1, relu=False, wd=None)

                discriminator = tf.reduce_mean(output, [1, 2, 3])

            elif self.version == 10:
                # 44x44x3
                conv_num = 1
                conv_1 = conv2d('d_conv_{}'.format(conv_num), data_313, [3, 3, 3, 64], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(conv_1)

                # 44x44x64
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 64, 128], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                # 44x44x128
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 128, 128], stride=2, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                # 22x22x128
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 128, 256], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                # 22x22x256
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 256, 256], stride=2, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                # 11x11x256
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 256, 512], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                # 11x11x512
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 512, 512], stride=2, relu=False, wd=None, same=False)
                output = tf.nn.leaky_relu(output)

                # 5x5x512
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 512, 512], stride=2, relu=False, wd=None, same=False)
                output = tf.nn.leaky_relu(output)

                flatten = tf.layers.flatten(output)
                discriminator = Linear('dense', flatten, 1)
            elif self.version == 12:
                # 44x44x314
                conv_num = 1
                output = conv2d('d_conv_{}'.format(conv_num), data_313, [3, 3, 314, 128], stride=1, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 128, 128], stride=2, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                # 22x22x128
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 128, 256], stride=2, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)

                # 11x11x256
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [3, 3, 256, 512], stride=2, relu=False, wd=None)
                output = tf.nn.leaky_relu(output)
                
                # 5x5x512
                conv_num += 1
                output = conv2d('d_conv_{}'.format(conv_num), output, [1, 1, 512, 1], stride=1, relu=False, wd=None)

                discriminator = tf.reduce_mean(output, [1, 2, 3])

        return discriminator

    def discriminator_loss(self, real_data, fake_data):
        # original_loss = -0.9 * tf.log(original + self.eps) - 0.1 * tf.log(1. - original + self.eps)  # Label smoothing.
        # colorized_loss = -tf.log(1 - colorized + self.eps)
        # fake_loss = tf.reduce_mean(colorized_loss)
        # real_loss = tf.reduce_mean(original_loss)
        # total_loss = (fake_loss + real_loss) / 2.
        # tf.summary.scalar('D_weight_loss', tf.add_n(tf.get_collection('losses', scope=scope)))
        # total_loss += tf.add_n(tf.get_collection('losses', scope=scope))

        # WGAN-GP
        real_score = self.discriminator(real_data, reuse=tf.AUTO_REUSE)
        fake_score = self.discriminator(fake_data, reuse=tf.AUTO_REUSE)
        # drift_loss = tf.reduce_mean(tf.square(real_score))
        fake_score = tf.reduce_mean(fake_score)
        real_score = tf.reduce_mean(real_score)
        total_loss = fake_score - real_score

        alpha = tf.random_uniform(
            shape=[self.batch_size, 1, 1, 1], 
            minval=0.,
            maxval=1.,
            dtype=fake_data.dtype
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences) 
        gradients = tf.gradients(self.discriminator(interpolates, reuse=tf.AUTO_REUSE), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - self.k)**2)
        total_loss += self.gp_lambda * gradient_penalty

        return total_loss, real_score, fake_score, tf.reduce_mean(slopes)

    def conv313_to_ab(self, conv8_313, rebalance=2.63):
        '''
        conv 313 to ab.
        Return: []
        '''
        enc_dir = './resources'
        cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
        cc = tf.constant(cc, dtype=tf.float32)  # [313, 2]
        shape = tf.shape(conv8_313)
        conv8_313_rh = conv8_313 * rebalance
        conv8_313_rh = tf.reshape(conv8_313_rh, (-1, 313))  # [N*H*W/16, 313]
        class8_313_rh = tf.nn.softmax(conv8_313_rh, axis=-1)  # [N*H*W/16, 313]
        
        data_ab = tf.matmul(class8_313_rh, cc)  # [N*H*W/16, 2]
        data_ab = tf.reshape(data_ab, (shape[0], shape[1], shape[2], 2))  # [N, H/4, W/4, 2]

        return data_ab

    def caption_encoding(self, version, captions, lens, idx=-1):
        print('Caption encoding version {}.'.format(version))
        if version == 0:
            return self.caption_encoding0(captions, lens, idx)
        elif version == 1:
            return self.caption_encoding1(captions, lens, idx)
        elif version == 2:
            return self.caption_encoding2(captions, lens, idx)

    def caption_encoding0(self, captions, lens, idx=-1):
        with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
            embedding = tf.constant(self.word_embedding, name='word_embedding', dtype='float32')
            encoded_captions = tf.nn.embedding_lookup(embedding, captions, name='lookup')
            encoded_captions = tf.nn.dropout(encoded_captions, 0.8 if self.train else 1.)
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, dtype=tf.float32)
            fw_name = '' if idx == -1 else 'fw_{}'.format(idx)
            bw_name = '' if idx == -1 else 'bw_{}'.format(idx)
            lstm_fw = tf.nn.rnn_cell.LSTMCell(self.lstm_hid_dim, reuse=tf.AUTO_REUSE, initializer=initializer, name=fw_name)
            lstm_bw = tf.nn.rnn_cell.LSTMCell(self.lstm_hid_dim, reuse=tf.AUTO_REUSE, initializer=initializer, name=bw_name)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, encoded_captions, sequence_length=lens, dtype='float32')
            hidden = tf.concat((state_fw.h, state_bw.h), 1)
            return hidden

    def caption_encoding1(self, captions, lens, idx=-1):
        with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
            embedding = tf.constant(self.word_embedding, name='word_embedding', dtype='float32')
            encoded_captions = tf.nn.embedding_lookup(embedding, captions, name='lookup')
            encoded_captions = tf.nn.dropout(encoded_captions, 0.8 if self.train else 1.)
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, dtype=tf.float32)
            fw_name = '' if idx == -1 else 'fw_{}'.format(idx)
            lstm_fw = tf.nn.rnn_cell.LSTMCell(512, reuse=tf.AUTO_REUSE, initializer=initializer, name=fw_name)
            _, state_fw = tf.nn.dynamic_rnn(lstm_fw, encoded_captions, sequence_length=lens, dtype='float32')
            hidden = state_fw.h
            return hidden
    
    def caption_encoding2(self, captions, lens, idx=-1):
        '''GRU 1024'''
        with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
            embedding = tf.constant(self.word_embedding, name='word_embedding', dtype='float32')
            encoded_captions = tf.nn.embedding_lookup(embedding, captions, name='lookup')
            encoded_captions = tf.nn.dropout(encoded_captions, 0.8 if self.train else 1.)
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, dtype=tf.float32)
            fw_name = '' if idx == -1 else 'fw_{}'.format(idx)
            lstm_fw = tf.nn.rnn_cell.GRUCell(1024, reuse=tf.AUTO_REUSE, kernel_initializer=initializer, name=fw_name)
            _, state = tf.nn.dynamic_rnn(lstm_fw, encoded_captions, sequence_length=lens, dtype='float32')
            print(state.get_shape())
            return state

    def sample_by_caption(self, captions, lens, l_ss, color_emb):
        with tf.variable_scope('Sampler', reuse=tf.AUTO_REUSE):
            cap_emb = self.caption_encoding(captions, lens)
            shape = tf.shape(l_ss)
            out_dim = color_emb.get_shape()[-1]
            cap_emb_expand = cap_emb[:, tf.newaxis, tf.newaxis, :]
            cap_emb_expand = tf.tile(cap_emb_expand, (1, shape[1], shape[2], 1))  # NxHxWx512
            emb = tf.concat((cap_emb_expand, l_ss), axis=-1)  # NxHxWx513
            conv_1 = conv2d('conv_1', emb, [3, 3, 513, 512], stride=1, wd=self.weight_decay)
            conv_2 = conv2d('conv_2', conv_1, [3, 3, 512, 256], stride=1, wd=self.weight_decay)
            gamma = conv2d('conv_3', conv_2, [3, 3, 256, out_dim], stride=1, wd=self.weight_decay, relu=False)
            beta = conv2d('conv_4', conv_2, [3, 3, 256, out_dim], stride=1, wd=self.weight_decay, relu=False)
            color_emb = tf.stop_gradient(color_emb)
            output = color_emb * gamma + beta
            output = tf.concat((output, l_ss), axis=-1)
            output = conv2d('conv_5', output, [3, 3, out_dim + 1, 128], stride=1, wd=self.weight_decay)
            output = conv2d('conv_6', output, [3, 3, 128, 64], stride=1, wd=self.weight_decay)
            output = conv2d('conv_7', output, [3, 3, 64, 2], stride=1, wd=self.weight_decay, relu=False)
            return output

    def sample_by_caption_1(self, captions, lens, l_ss, color_emb):
        with tf.variable_scope('Sampler', reuse=tf.AUTO_REUSE):
            cap_emb = self.caption_encoding(captions, lens)
            shape = tf.shape(l_ss)
            out_dim = color_emb.get_shape()[-1]
            cap_emb_expand = cap_emb[:, tf.newaxis, tf.newaxis, :]
            cap_emb_expand = tf.tile(cap_emb_expand, (1, shape[1], shape[2], 1))  # NxHxWx512
            l_emb = tf.stop_gradient(color_emb)
            emb = tf.concat((cap_emb_expand, l_emb), axis=-1)  # NxHxWx1024
            conv_1 = conv2d('conv_1', emb, [3, 3, 1024, 512], stride=1, wd=self.weight_decay)
            conv_2 = conv2d('conv_2', conv_1, [3, 3, 512, 256], stride=1, wd=self.weight_decay)
            conv_3 = conv2d('conv_3', conv_2, [3, 3, 256, 64], stride=1, wd=self.weight_decay)
            attention = conv2d('conv_4', conv_3, [3, 3, 64, 1], stride=1, wd=self.weight_decay, relu=False)
            
            
            output = color_emb * gamma + beta
            output = tf.concat((output, l_ss), axis=-1)
            output = conv2d('conv_5', output, [3, 3, out_dim + 1, 128], stride=1, wd=self.weight_decay)
            output = conv2d('conv_6', output, [3, 3, 128, 64], stride=1, wd=self.weight_decay)
            output = conv2d('conv_7', output, [3, 3, 64, 2], stride=1, wd=self.weight_decay, relu=False)
            return output

    def sample_loss(self, scope, gt_ab, pred_ab, prior):
        huber_loss = tf.losses.huber_loss(gt_ab, pred_ab, weights=prior)
        wd_loss = tf.get_variable('wd_loss', (), initializer=tf.zeros_initializer, dtype=tf.float32)
        for var in tf.trainable_variables(scope='Sampler'):
            if 'weights' in var.name or 'kernel' in var.name:
                wd_loss = wd_loss + tf.nn.l2_loss(var)
        wd_loss = wd_loss * self.weight_decay
        return huber_loss, huber_loss + wd_loss, wd_loss

    def attention_block(self, conv, cap_emb):
        with tf.variable_scope('Attention', reuse=tf.AUTO_REUSE):
            conv = tf.stop_gradient(conv)

            shape = tf.shape(conv)
            cap_emb_expand = cap_emb[:, tf.newaxis, tf.newaxis, :]
            cap_emb_expand = tf.tile(cap_emb_expand, (1, shape[1], shape[2], 1))  # NxHxWx512
            
            temp = tf.concat((conv, cap_emb_expand), axis=-1)  # NxHxWx1024
            temp = conv2d('conv_1', temp, [3, 3, 1024, 512], stride=1, wd=self.weight_decay)
            temp = conv2d('conv_2', temp, [3, 3, 512, 128], stride=1, wd=self.weight_decay)
            temp = conv2d('conv_3', temp, [3, 3, 128, 16], stride=1, wd=self.weight_decay)
            attention_map = conv2d('conv_4', temp, [3, 3, 16, 1], stride=1, wd=self.weight_decay, relu=False, sigmoid=True)

        return attention_map

    def attention_block_1(self, l, cap_emb):
        with tf.variable_scope('Attention', reuse=tf.AUTO_REUSE):
            # conv = tf.stop_gradient(conv)
            conv = conv2d('l1', l, [3, 3, 1, 64], stride=2, wd=self.weight_decay)  # 112
            conv = conv2d('l2', conv, [3, 3, 64, 128], stride=2, wd=self.weight_decay)  # 56
            conv = conv2d('l3', conv, [3, 3, 128, 256], stride=2, wd=self.weight_decay)  # 28

            shape = tf.shape(conv)
            cap_emb_expand = cap_emb[:, tf.newaxis, tf.newaxis, :]
            cap_emb_expand = tf.tile(cap_emb_expand, (1, shape[1], shape[2], 1))  # NxHxWx512
            
            temp = tf.concat((conv, cap_emb_expand), axis=-1)  # NxHxWx1024
            temp = conv2d('conv_1', temp, [3, 3, 768, 384], stride=1, wd=self.weight_decay)
            temp = conv2d('conv_2', temp, [3, 3, 384, 128], stride=1, wd=self.weight_decay)
            temp = conv2d('conv_3', temp, [3, 3, 128, 16], stride=1, wd=self.weight_decay)
            attention_map = conv2d('conv_4', temp, [3, 3, 16, 1], stride=1, wd=self.weight_decay, relu=False, sigmoid=True)

        return attention_map
