import tensorflow as tf
from . import variables

class batch_norm(object):
    """
    Batch normalization Layer 
    origin source : https://github.com/carpedm20/DCGAN-tensorflow
    참고 : https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    Ex) bn = batch_norm('sample layer')
        afterBN = bn(preBN_variable)
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        """
        :param epsilon: 
        Small float added to variance to avoid dividing by zero.
        
        :param momentum:
        Decay for the moving average. 
        Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc. 
        Lower decay value (recommend trying decay=0.9) if model experiences reasonably good training performance but poor validation and/or test performance.
        Try zero_debias_moving_mean=True for improved stability.
        
        :param name: 
        Variable name
        """
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        """
        call tf.contrib.layers.batch_norm
        :param x: 
        :param train:
        Whether or not the layer is in training mode. 
        In training mode it would accumulate the statistics of the moments into moving_mean and moving_variance using an exponential moving average with the given decay. 
        When it is not in training mode then it would use the values of the moving_mean and the moving_variance.
        
        :return:  
        """
        return tf.contrib.layers.batch_norm(x,
                                                decay=self.momentum,
                                                updates_collections=None,
                                                epsilon=self.epsilon,
                                                scale=True,
                                                is_training=train,
                                                scope=self.name)


def conv2d(input,ksize,strides,padding,layerName,initializer=variables.variable_xavier):
    """
    Convolution Layer
    :param input:
     Input tensor([batch_size, height, width, channel])
    :param ksize:
     Kernel(filter) size
     Ex) [1,2,2,1]
    :param strides: 
     Stride dim
     Ex) [1,2,2,1]
    :param padding: 
    Padding Type
     Ex) 'SAME' or 'VALID'
    :param layerName:
     Tensorboard name
    :param initializer:
     Default : xavier
    :return: 
     Output tensor 
    """

def maxPool(input,ksize,strides,padding,layerName):
    """
    MaxPool Layer
    :param input: 
     Input tensor
    :param ksize:
     kernel(filter) size
    :param strides: 
     Stride dim
    :param padding:
     Padding Type
     Ex) 'SAME' or 'VALID'
    :param layerName:
     Tensorboard name
    :return: 
     Output tensor
    """

def avgPool(input,ksize,strides,padding,layerName):
    """
    AvgPool Layer
    :param input: 
     Input tensor
    :param ksize:
     kernel(filter) size
    :param strides: 
     Stride dim
    :param padding:
     Padding Type
     Ex) 'SAME' or 'VALID'
    :param layerName:
     Tensorboard name
    :return: 
     Output tensor
    """


def nnLayer(input,outputSize,layerName,initializer=variables.variable_xavier):
    """
    이름이 안떠올라요 도와주세요....
    WX+B
    :param input:
     Input tensor
    :param outputSize:
      Output tensor dim
    :param layerName: 
     Tensorboard name
    :param initializer:
     Default : xavier
    :return:
     Output tensor
    """


if __name__=="__main__":
    """
    Test code...
    """