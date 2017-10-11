import tensorflow as tf
import numpy as np
import sys

sys.path.append('../utils')
sys.path.append('../vgg19')
from layer import *
from vgg19 import VGG19
import cv2

class SRGAN:
    def __init__(self, x, is_training, batch_size=32):
        self.batch_size = batch_size
        self.vgg = VGG19(None, None, None)
        self.downscaled = self.downscale(x)
        self.imitation = self.generator(self.downscaled, is_training, False)
        # using generator, unuse
        if not is_training:
            self.real_output = self.discriminator(x, is_training, False)
            self.fake_output = self.discriminator(self.imitation, is_training, True)
            self.g_loss, self.d_loss = self.inference_losses(x, self.imitation, self.real_output, self.fake_output)

    def downscale(self, x):
        downscaled = tf.image.resize_images(x, (24, 24))
        return downscaled

    def generator(self, x, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                x = deconv_layer(x, [3, 3, 64, 3], [self.batch_size, 24, 24, 64], 1)

            x = tf.nn.relu(x)
            shortcut = x
            for i in range(5):
                mid = x
                with tf.variable_scope('block{}a'.format(i + 1)):
                    x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('block{}b'.format(i + 1)):
                    x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                    x = batch_normalize(x, is_training)
                x = tf.add(x, mid)

            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.add(x, shortcut)
            # end residual block

            with tf.variable_scope('deconv3'):  # pixel_shuffle_layer x2
                x = deconv_layer(x, [3, 3, 256, 64], [self.batch_size, 24, 24, 256], 1)
                x = pixel_shuffle_layer(x, 2, 64)  # n_split = 256 / 2 ** 2
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv4'):  # pixel_shuffle_layer x2x2
                x = deconv_layer(x, [3, 3, 64, 64], [self.batch_size, 48, 48, 64], 1)
                x = pixel_shuffle_layer(x, 2, 16)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv5'):
                x = deconv_layer(
                    x, [3, 3, 3, 16], [self.batch_size, 96, 96, 3], 1)

        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return x

    def discriminator(self, x, is_training, reuse):

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = lrelu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 64], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 128], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 512, 512], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            x = flatten_layer(x)
            print('1', x)
            with tf.variable_scope('fc'):
                x = full_connection_layer(x, 1024)
                x = lrelu(x)
            print('2', x)
            with tf.variable_scope('softmax'):
                x = full_connection_layer(x, 1)
            print('3', x)

        self.d_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return x

    def inference_losses(self, x, imitation, true_output, fake_output):
        def inference_content_loss(x, imitation):
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), False)  # First
            _, imitation_phi = self.vgg.build_model(
                imitation, tf.constant(False), True)  # Second

            content_loss = None
            for i in range(len(x_phi)):
                l2_loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                if content_loss is None:
                    content_loss = l2_loss
                else:
                    content_loss = content_loss + l2_loss
            return tf.reduce_mean(content_loss)

        def inference_adversarial_loss_with_sigmoid(real_output, fake_output):
            alpha = 1e-3
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output), logits=fake_output)
            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output)
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output)
            d_loss = (d_loss_real + d_loss_fake)  # modified
            return (g_loss * alpha, d_loss * alpha)

        content_loss = inference_content_loss(x, imitation)
        generator_loss, discriminator_loss = (inference_adversarial_loss_with_sigmoid(true_output, fake_output))
        g_loss = content_loss + generator_loss
        d_loss = discriminator_loss
        return (g_loss, d_loss)

