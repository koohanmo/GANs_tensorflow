import tensorflow as tf

def variable_xavier(name,shape):
    """
    변수를 xavier초기화로 초기화 해서 반환
    :param shape:
        사용 할 variable의 shape, d
    :return:
        tf.variable
    """
    var = tf.get_variable(name = name, shape = shape , initializer=tf.contrib.layers.xavier_initializer())
    variable_summaries(var)
    return var

def variable_truncated(name, shape, stddev=0.1):
    """
    변수를 truncated_norm으로 초기화 해서 반환
    :param name:
        사용 할 variable의 name
    :param shape:
        사용 할 variable의 shape, d
    :param stddev:
        default = 0.1
        tf. truncated_norm에서 사용 될 표준편차
    :return:
        tf.variable
    """
    var = tf.truncated_normal(name = name, shape = shape, stddev = stddev)
    variable_summaries(var)
    return var

def variable_summaries(var):
    """
    tf.variable의 tensorboard에서 볼 수 있도록 정리
    Ex) average, stddev, max, min, histogram.....
    :param var:
        tensorboard에 표시 할 변수
    """
    with tf.name_scope('summary'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram(var)


if __name__=="__main__":
    """
    Test code...
    """