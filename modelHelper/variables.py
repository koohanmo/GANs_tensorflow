import tensorflow as tf

def variable_xavier(name,shape):
    """
    변수를 xavier초기화로 초기화 해서 반환
    :param shape:
        사용 할 variable의 shape, d
    :return:
        tf.variable
    """
    pass

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
    pass

def variable_summaries(var):
    """
    tf.variable의 tensorboard에서 볼 수 있도록 정리
    Ex) average, stddev, max, min, histogram.....
    :param var:
        tensorboard에 표시 할 변수
    """

if __name__=="__main__":
    """
    Test code...
    """