import tensorflow as tf

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




