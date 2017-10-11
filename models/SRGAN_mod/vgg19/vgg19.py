import tensorflow as tf

import sys
sys.path.append('../utils')
from ..utils.layer import *

class VGG19:
    def __init__(self, x, t, is_training):
        if x is None: return
        self.out, self.phi = self.build_model(x, is_training)
        self.loss = self.inference_loss(self.out, t)

    def build_model(self, x, is_training, reuse=False):
        """

        :param x:
        :param is_training:
        :param reuse:
        :return:
        """
        VGG_MEAN = [103.939, 116.779, 123.68]  # 0 : blue_mean, 1 : green_mean, 2 : red_mean
        grid = 96
        rgb_scaled = x * 255.0
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [grid, grid, 1]
        assert green.get_shape().as_list()[1:] == [grid, grid, 1]
        assert blue.get_shape().as_list()[1:] == [grid, grid, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [grid, grid, 3]

        with tf.variable_scope('vgg19', reuse=reuse):
            phi = []
            with tf.variable_scope('conv1a'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv1b'):
                x = conv_layer(x, [3, 3, 64, 64], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv2a'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv2b'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv3a'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv3b'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv3c'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv3d'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv4a'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv4b'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv4c'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv4d'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv5a'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv5b'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv5c'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv5d'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            x = flatten_layer(x)
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 4096)
                x = lrelu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 4096)
                x = lrelu(x)
            with tf.variable_scope('softmax'):
                x = full_connection_layer(x, 1000)

            return x, phi


    def inference_loss(self, out, t):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(t, 1000),
            logits=out)
        return tf.reduce_mean(cross_entropy)

