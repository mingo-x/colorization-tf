from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
# import numpy as np
# import re
nilboy_weight_decay = 0.001
def variable(name, shape, initializer, trainable=True):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the Variable
    shape: list of ints
    initializer: initializer of Variable

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
  return var

def variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with truncated normal distribution
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable 
    shape: list of ints
    stddev: standard devision of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight 
    decay is not added for this Variable.

 Returns:
    Variable Tensor 
  """
  # var = _variable(name, shape,
  #   tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  var = variable(name, shape,
    tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, dtype=tf.float32))
  
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def ConvMeanPool(scope, input, filter_size):
    output = conv2d(scope, input, filter_size, relu=False, wd=0.001)
    output = tf.add_n([output[:,::2,::2, :], output[:,1::2,::2,:], output[:,::2,1::2, :], output[:,1::2,1::2, :]]) / 4.
    return output


def Linear(scope, input, dim, bias_initializer=tf.constant_initializer(0.0)):
  kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, dtype=tf.float32)
  return tf.layers.dense(input, dim, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)


def MeanPoolConv(scope, input, filter_size):
    output = input
    output = tf.add_n([output[:,::2,::2, :], output[:,1::2,::2,:], output[:,::2,1::2, :], output[:,1::2,1::2, :]]) / 4.
    output = conv2d(scope, output, filter_size, relu=False, wd=0.001)
    return output


def UpsampleConv(scope, input, filter_size):
    output = input
    output = tf.concat([output, output, output, output], axis=3)
    # output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    # output = tf.transpose(output, [0,3,1,2])
    output = conv2d(scope, output, filter_size, relu=False, wd=0.001)
    return output


def Normalize(name, inputs, train):
    if ('D' in name):
        return tf.contrib.layers.layer_norm(inputs)
    else:
        return batch_norm(name, inputs, train)


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, train=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1 = functools.partial(conv2d, kernel_size=[filter_size, filter_size, input_dim, input_dim], relu=False, wd=0.001)
        conv_2 = functools.partial(ConvMeanPool, filter_size=[filter_size, filter_size, input_dim, output_dim])
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1 = functools.partial(UpsampleConv, filter_size=[filter_size, filter_size, input_dim, output_dim])
        conv_2 = functools.partial(conv2d, kernel_size=[filter_size, filter_size, output_dim, output_dim], relu=False, wd=0.001)

    shortcut = conv_shortcut(name+'.Shortcut', inputs, [1, 1, input_dim, output_dim])

    output = inputs
    output = Normalize(name+'.BN1', output, train)
    output = tf.nn.relu(output)
    output = conv_1(scope=name+'.Conv1', input=output)
    output = Normalize(name+'.BN2', output, train)
    output = tf.nn.relu(output)
    output = conv_2(scope=name+'.Conv2', input=output)

    return shortcut + output


def conv2d(scope, input, kernel_size, stride=1, dilation=1, relu=True, wd=nilboy_weight_decay, sigmoid=False, same=True):
  # name = scope
  with tf.variable_scope(scope) as scope:
    kernel = variable_with_weight_decay('weights', 
                                    shape=kernel_size,
                                    stddev=5e-2,
                                    wd=wd)
    if dilation == 1:
      conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME' if same else 'VALID')
    else:
      conv = tf.nn.atrous_conv2d(input, kernel, dilation, padding='SAME')
    biases = variable('biases', kernel_size[3:], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    if relu:
      conv1 = tf.nn.relu(bias)
    elif sigmoid:
      conv1 = tf.nn.sigmoid(bias)
    else:
      conv1 = bias
  return conv1

def deconv2d(scope, input, kernel_size, stride=1, relu=True, wd=nilboy_weight_decay):
  """convolutional layer

  Args:
    input: 4-D tensor [batch_size, height, width, depth]
    scope: variable_scope name 
    kernel_size: [k_height, k_width, in_channel, out_channel]
    stride: int32
  Return:
    output: 4-D tensor [batch_size, height * stride, width * stride, out_channel]
  """
  # pad_size = int((kernel_size[0] - 1)/2)
  #input = tf.pad(input, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")
  batch_size, height, width, in_channel = [int(i) for i in input.get_shape()]
  out_channel = kernel_size[3] 
  kernel_size = [kernel_size[0], kernel_size[1], kernel_size[3], kernel_size[2]]
  output_shape = [batch_size, height * stride, width * stride, out_channel]
  with tf.variable_scope(scope) as scope:
    kernel = variable_with_weight_decay('weights', 
                                    shape=kernel_size,
                                    stddev=5e-2,
                                    wd=wd)
    deconv = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding='SAME')

    biases = variable('biases', (out_channel), tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(deconv, biases)
    if relu:
      deconv1 = tf.nn.relu(bias)
    else:
      deconv1 = bias

  return deconv1

def batch_norm(scope, x, train=True):
  return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train, trainable=True, scope=scope)


def bn(scope, x, train=True):
  return tf.contrib.layers.batch_norm(x, epsilon=1e-5, center=False, scale=False, updates_collections=None, is_training=train, trainable=False, scope=scope)