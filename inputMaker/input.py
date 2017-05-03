import numpy as np
import scipy as sp
import collections

class Datasets(object):
    """
    train, test, validation 3가지로 나누어서
    데이터셋을 반환 할 때 사용하는 클래스
    """
    def __init__(self,train=6, test=3, validation=1):
        """
        :param train:
         훈련 데이터 비율
        :param test:
         테스트 데이터 비율
        :param validation: 
         검증 데이터 비율
        """
        self._train= train
        self._test = test
        self._validation=validation

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def validation(self):
        return self._validation

def load_image_data_without_label(name):

    pass

def load_image_data_with_label(name):

    pass

def load_video_data(name):

    pass

def load_video_data_with_label(name):

    pass


if __name__=="__main__":
    """
    Test code...
    """