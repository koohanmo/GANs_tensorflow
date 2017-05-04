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
    variable_summaries(name = name, var=var)
    return var

def variable_random(name, shape, stddev = 0.01, mean = 0.0):
    """
    변수를 random_norm으로 초기화해서 반환
    :param name:
        사용 할 variable의 이름
    :param shape:
        사용 할 variable의 shape, dim
    :param stddev:
        default = 0.1
         random_norm에서 사용될 표준편차
    :param mean:
        default = 0.0
        random_norm 에서 사용될 평균
    :return:
        tf.variable
    """
    var = tf.random_normal(shape = shape, mean = mean , stddev = stddev , name = name)
    variable_summaries(name=name,var=var)
    return tf.Variable(var)

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
    variable_summaries(name = name, var = var)
    return tf.Variable(var)

def variable_summaries(name,var):
    """
    tf.variable의 tensorboard에서 볼 수 있도록 정리
    Ex) average, stddev, max, min, histogram.....
    :param name:
        namescope로 변수를 묶어둠.
        수정: 이전 'summary'로 묶을 시 summary summary_1 summary_2 로 의미없는 명이 계속사용됨.
    :param var:
        tensorboard에 표시 할 변수
    """
    # before_namescope/namescope('summary')/
    with tf.name_scope(name=name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('/mean', mean)
        tf.summary.scalar('/stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
        tf.summary.scalar('/max', tf.reduce_max(var))
        tf.summary.scalar('/min', tf.reduce_min(var))
        tf.summary.histogram('/histogram',var)

if __name__=="__main__":
    """
    Test code...
    """