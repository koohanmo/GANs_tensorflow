"""
 SRGAN_tensorflow
 Photo-Realistic Single Image Super-Resolution 
 Using a Generative Adversarial Network
 https://arxiv.org/abs/1609.04802
"""

import tensorflow as tf
import modelHelper as mh


FLAGS = tf.app.flags.FLAGS

class SRGAN(object):
    def __init__(self,data,
                 input_img_shape,
                 output_img_shape):
        self.input_img_shape = input_img_shape
        self.output_img_shape = output_img_shape
        pass

    def build(self):
        pass

    def inference(self):
        pass

    def train(self):
        pass

    def generator(self):
        pass

    def discriminator(self,inputs,lables, reuse=False):
        with tf.name.scope('discriminator'):
            # [batchSize, self.output_img_shape[0], self.output_img_shape[1],3]
            # [batchSize, 640, 480, 3]
            layer1 = mh.layers.conv2d(layerName='dis_layer1',
                                      t_input=inputs,
                                      ksize = [3,3,3,64],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            act1 = mh.layers.lrelu(layer1)

            # [batchSize, 640, 480, 64]
            layer2 = mh.layers.conv2d(layerName='dis_layer2',
                                      t_input=act1,
                                      ksize=[3, 3, 64, 64],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')
            bn2 = mh.layers.batch_norm(name='dis_layer2_BN')
            act2 = mh.layers.lrelu(bn2(layer2))

            # [batchSize, 320, 240, 64]
            layer3 = mh.layers.conv2d(layerName='dis_layer3',
                                      t_input=act2,
                                      ksize=[3, 3, 64, 128],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
            bn3 = mh.layers.batch_norm(name='dis_layer3_BN')
            act3 = mh.layers.lrelu(bn3(layer3))

            # [batchSize, 320, 240, 128]
            layer4 = mh.layers.conv2d(layerName='dis_layer4',
                                      t_input=act3,
                                      ksize=[3, 3, 128, 128],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')
            bn4 = mh.layers.batch_norm(name='dis_layer4_BN')
            act4 = mh.layers.lrelu(bn4(layer4))

            # [batchSize, 160, 120, 128]
            layer5 = mh.layers.conv2d(layerName='dis_layer5',
                                      t_input=act4,
                                      ksize=[3, 3, 128, 256],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
            bn5 = mh.layers.batch_norm(name='dis_layer5_BN')
            act5 = mh.layers.lrelu(bn5(layer5))

            # [batchSize, 160, 120, 256]
            layer6 = mh.layers.conv2d(layerName='dis_layer6',
                                      t_input=act5,
                                      ksize=[3, 3, 256, 256],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')
            bn6 = mh.layers.batch_norm(name='dis_layer6_BN')
            act6 = mh.layers.lrelu(bn6(layer6))

            # [batchSize, 80, 60, 256]
            layer7 = mh.layers.conv2d(layerName='dis_layer7',
                                      t_input=act6,
                                      ksize=[3, 3, 256, 512],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
            bn7 = mh.layers.batch_norm(name='dis_layer7_BN')
            act7 = mh.layers.lrelu(bn7(layer7))

            # [batchSize, 80, 60, 512]
            layer8 = mh.layers.conv2d(layerName='dis_layer8',
                                      t_input=act7,
                                      ksize=[3, 3, 512, 512],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')
            bn8 = mh.layers.batch_norm(name='dis_layer2_BN')
            act8 = mh.layers.lrelu(bn8(layer8))

            # [batchSize, 40, 30, 512]
            dense1 = mh.layers.flatten(act8, 40*30*512,layerName='dense1')
            act9 =  mh.layers.lrelu(dense1) # 0.2
            dense2 = mh.layers.fullyConnected(act9,[40*30*512,1])
            ret = tf.nn.sigmoid(dense2)

        return ret

    def residual_block(self, inputs, layerName='residual'):
        with tf.name_scope(layerName):
            conv1 = mh.layers.conv2d(layerName='conv1',
                                      t_input=inputs,
                                      ksize = [3,3,64,64],
                                      strides=[1,1,1,1],
                                      padding='SAME')
            bn1 = mh.layers.batch_norm(name='BN1')
            act1 = mh.layers.lrelu(bn1(conv1),leak=0.25)

            conv2 = mh.layers.conv2d(layerName='conv2',
                                     t_input=act1,
                                     ksize=[3, 3, 64, 64],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME')
            bn2 = mh.layers.batch_norm(name='_BN2')
            pre_act = bn2(conv2)
            ret = tf.add(inputs, pre_act)
        return ret


if __name__=="__main__":
    print ('------SRGAN------')