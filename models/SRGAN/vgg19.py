import tensorflow as tf
from modelHelper import layers

class VGG19:
    def __init__(self, input, t, is_training):
        if input is None: return
        self.out, self.phi = self.build_model(input, is_training)
        self.loss = self.inference_loss(self.out, t)

    def build_model(self, t_input, is_training, reuse=False):

        with tf.variable_scope('vgg19', reuse=reuse):
            phi = []
            with tf.variable_scope('conv1a'):
                conv1a = layers.conv2d(
                                    layerName='vgg19_conv1a',
                                    t_input=t_input,
                                    ksize= [3, 3, 3, 64],
                                    strides=[1,1,1,1],
                                    padding='SAME')
                conv1a = layers.batch_norm(conv1a, is_training)

                BN = layers.batch_norm()
                conv1a = BN(conv1a)
                conv1a = layers.lrelu(conv1a)


            with tf.variable_scope('conv1b'):
                conv1b = layers.conv2d(layerName='vgg19_conv1b',
                                  t_input=conv1a,
                                  ksize=[3,3,64,64],
                                  strides=[1,1,1,1],
                                  padding='SAME')
                BN = layers.batch_norm()
                conv1b = BN(conv1b)
                conv1b = layers.lrelu(conv1b)
            phi.append(conv1b)

            pool_first = layers.maxPool(
                                        layerName='pool_first',
                                        t_input=conv1b,
                                        ksize=2,
                                        strides=2)
            # pool 1

            with tf.variable_scope('conv2a'):
                conv2a = layers.conv2d(layerName='vgg19_conv2a',
                                       t_input=pool_first,
                                       ksize=[3, 3, 64, 128],
                                       strides=[1,1,1,1],
                                       padding='SAME')

                BN = layers.batch_norm()
                conv2a = BN(conv2a)
                conv2a = layers.lrelu(conv2a)


            with tf.variable_scope('conv2b'):
                conv2b = layers.conv2d(layerName='vgg19_conv2b',
                                       t_input=conv2a,
                                       ksize=[3, 3, 128, 128],
                                       strides=[1,1,1,1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv2b = BN(conv2b)
                conv2b = layers.lrelu(conv2b)
            phi.append(conv2b)

            pool_second= layers.maxPool(conv2b, 2, 2)

            # pool 2

            with tf.variable_scope('conv3a'):
                conv3a = layers.conv2d(layerName='vgg19_conv3a',
                                       t_input=pool_second,
                                       ksize=[3, 3, 128, 256],
                                       strides=[1,1,1,1],
                                       padding='SAME')

                BN = layers.batch_norm()
                conv3a = BN(conv3a)
                conv3a = layers.lrelu(conv3a)


            with tf.variable_scope('conv3b'):
                conv3b = layers.conv2d(layerName='vgg19_conv3b',
                                       t_input=conv3a,
                                       ksize=[3, 3, 256, 256],
                                       strides=[1,1,1,1],
                                       padding='SAME')

                BN = layers.batch_norm()
                conv3b = BN(conv3b)
                conv3b = layers.lrelu(conv3b)

            with tf.variable_scope('conv3c'):
                conv3c = layers.conv2d(layerName='vgg19_conv3c',
                                       t_input=conv3b,
                                       ksize=[3, 3, 256, 256],
                                       strides=[1,1,1,1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv3c = BN(conv3c)
                conv3c = layers.lrelu(conv3c)

            with tf.variable_scope('conv3d'):
                conv3d = layers.conv2d(layerName='vgg19_conv3d',
                                       t_input=conv3c,
                                       ksize=[3, 3, 256, 256],
                                       strides=[1,1,1,1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv3d = BN(conv3d)
                conv3d = layers.lrelu(conv3d)
            phi.append(conv3d)

            pool_third = layers.maxPool(conv3d, 2, 2)


            with tf.variable_scope('conv4a'):
                conv4a = layers.conv2d(layerName='vgg19_conv4a',
                                       t_input=pool_third,
                                       ksize=[3, 3, 256, 512],
                                       strides=[1,1,1,1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv4a = BN(conv4a)
                conv4a = layers.lrelu(conv4a)

            with tf.variable_scope('conv4b'):
                conv4b = layers.conv2d(layerName='vgg19_conv4b',
                                       t_input=conv4a,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv4b = BN(conv4b)
                conv4b = layers.lrelu(conv4b)


            with tf.variable_scope('conv4c'):
                conv4c = layers.conv2d(layerName='vgg19_conv4c',
                                       t_input=conv4b,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv4c = BN(conv4c)
                conv4c = layers.lrelu(conv4c)


            with tf.variable_scope('conv4d'):
                conv4d = layers.conv2d(layerName='vgg19_conv4d',
                                       t_input=conv4c,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv4d = BN(conv4d)
                conv4d = layers.lrelu(conv4d)

            phi.append(conv4d)

            pool_fourth = layers.maxPool(conv4d, 2, 2)



            with tf.variable_scope('conv5a'):
                conv5a = layers.conv2d(layerName='vgg19_conv5a',
                                       t_input=pool_fourth,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv5a = BN(conv5a)
                conv5a = layers.lrelu(conv5a)

            with tf.variable_scope('conv5b'):
                conv5b = layers.conv2d(layerName='vgg19_conv5b',
                                       t_input=conv5a,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv5b = BN(conv5b)
                conv5b = layers.lrelu(conv5b)

            with tf.variable_scope('conv5c'):
                conv5c = layers.conv2d(layerName='vgg19_conv5c',
                                       t_input=conv5b,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv5c = BN(conv5c)
                conv5c = layers.lrelu(conv5c)

            with tf.variable_scope('conv5d'):
                conv5d = layers.conv2d(layerName='vgg19_conv5d',
                                       t_input=conv5c,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv5d = BN(conv5d)
                conv5d = layers.lrelu(conv5d)

            phi.append(conv5d)

            pool_fifth = layers.maxPool(conv5d, 2, 2)

            flatten = layers.flatten(t_input=pool_fifth,flatDim=1)

            with tf.variable_scope('fc1'):
                fc1 = layers.fullyConnected(flatten, 4096)
                fc1 = layers.lrelu(fc1)
            with tf.variable_scope('fc2'):
                fc2 = layers.fullyConnected(fc1, 4096)
                fc2 = layers.lrelu(fc2)
            with tf.variable_scope('softmax'):
                fc3 = layers.fullyConnected(fc2, 100)

            return fc3, phi


    def inference_loss(self, out, t):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(t, 100),
            logits=out)
        return tf.reduce_mean(cross_entropy)
