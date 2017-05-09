import numpy as np
from utils import image

class ImageSetWithoutLable(object):
    """
    Lable이 없는 이미지데이터셋
    """
    def __init__(self,
                 images,
                 normalize=False,
                 serialize=False,
                 dtype=np.float32,
                 shuffle=True):
        """
        :param images:
         데이터셋의 이미지들
        :param normalize: 
         0~1로 정규화 할 것인지
        :param serialize: 
         [W,H,Ch] -> [W*H*Ch], 1열로 만들 것인지
        :param dtype: 
         이미지의 Data type
        :param shuffle: 
         데이터들의 순서를 섞을 것인지
        """
        images = images.astype(dtype)
        if serialize:
            # [num_examples, rows, columns, depth] -> [num_examples, rows*columns*depth]
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])

        if normalize:
            # [0,255] -> [0,1]
            images = np.multiply(images, 1.0 / 255.0)

        self._shuffle = shuffle
        self._num_examples = images.shape[0]
        self._images = images
        self._epochs_completed = 0
        self._index_in_epoch = 0


    def next_batch(self, batch_size):
        """
        학습 시, Batch_size 만큼 옵션에 맞추어서 데이터를 반환
        :param batch_size:
         이번에 사용향 batch크기
        :return: 
         batch 크기의 이미지 데이터(np.array)
        """

        assert batch_size <= self._num_examples, (
            "batch_size : %s num_examples : %s" % (batch_size,
                                                   self._num_examples))

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            if self._shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]

            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._images[start:end]


class ImageSetWithLabel(object):
    """
    이미지 데이터 셋
    """
    def __init__(self,
                 images,
                 labels,
                 normalize=False,
                 serialize=False,
                 dtype=np.float32,
                 shuffle=True):
        """
        :param images:
         데이터셋의 이미지들
        :param labels:
         데이터셋의 Label들
        :param normalize: 
         0~1로 정규화 할 것인지
        :param serialize: 
         [W,H,Ch] -> [W*H*Ch], 1열로 만들 것인지
        :param dtype: 
         이미지의 Data type
        :param shuffle: 
         데이터들의 순서를 섞을 것인지 
        """
        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape,
                                                   labels.shape))
        images = images.astype(dtype)
        if serialize:
            # [num_examples, rows, columns, depth] -> [num_examples, rows*columns*depth]
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])

        if normalize:
            # [0,255] -> [0,1]
            images = np.multiply(images, 1.0 / 255.0)

        self._shuffle = shuffle
        self._num_examples = images.shape[0]
        self._images = images
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """
        학습 시, Batch_size 만큼 옵션에 맞추어서 데이터를 반환
        :param batch_size:
         이번에 사용향 batch크기
        :return: 
         batch 크기의 이미지 데이터(np.array), 이미지 Label(np.array)
        """
        assert batch_size <= self._num_examples, (
            "batch_size : %s num_examples : %s" % (batch_size,
                                                   self._num_examples))

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            if self._shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class ImagePathSetWithLable(object):
    """
    이미지의 Path만 저장하고 있다가 
    next_batch 실행시에 이미지를 불러와서 사용
    (ImageSet 클래스들은 모든 이미지를 np.array에 들고 있음)
    """
    def __init__(self,
                 paths,
                 labels,
                 normalize=False,
                 serialize=False,
                 dtype=np.float32,
                 shuffle=True):
        """
        :param paths:
         데이터셋의 이미지 path들
        :param labels:
         데이터셋의 Label들
        :param normalize: 
         0~1로 정규화 할 것인지
        :param serialize: 
         [W,H,Ch] -> [W*H*Ch], 1열로 만들 것인지
        :param dtype: 
         이미지의 Data type
        :param shuffle: 
         데이터들의 순서를 섞을 것인지 
        """
        self._normalize = normalize
        self._serialize = serialize
        self._dtype = dtype
        self._shuffle = shuffle
        self._num_examples = paths.shape[0]
        self._paths = paths
        self._labels= labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """
        학습 시, Batch_size 만큼 옵션에 맞추어서 데이터를 반환
        :param batch_size:
         이번에 사용향 batch크기
        :return: 
         batch 크기의 이미지 데이터(np.array), 이미지 Label(np.array)
        """
        assert batch_size <= self._num_examples, (
            "batch_size : %s num_examples : %s" % (batch_size,
                                                   self._num_examples))

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            if self._shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._paths = self._paths[perm]
                self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        # 이미지 로딩, 처리
        temp_images=[image.image_path_to_np(x) for x in self._images[start:end]]
        temp_images=np.array(temp_images)

        temp_images = temp_images.astype(self._dtype)
        if self._serialize:
            # [num_examples, rows, columns, depth] -> [num_examples, rows*columns*depth]
            temp_images = temp_images.reshape(temp_images.shape[0], temp_images.shape[1] * temp_images.shape[2] * temp_images.shape[3])

        if self._normalize:
            # [0,255] -> [0,1]
            temp_images = np.multiply(temp_images, 1.0 / 255.0)

        return temp_images, self._labels[start:end]


class ImagePathSetWithoutLable(object):
    """
    이미지의 Path만 저장하고 있다가 
    next_batch 실행시에 이미지를 불러와서 사용
    (ImageSet 클래스들은 모든 이미지를 np.array에 들고 있음)
    """
    def __init__(self,
                 paths,
                 normalize=False,
                 serialize=False,
                 dtype=np.float32,
                 shuffle=True):

        self._normalize = normalize
        self._serialize = serialize
        self._dtype = dtype
        self._shuffle = shuffle
        self._paths = np.array(paths)
        print(paths)
        self._num_examples = self._paths.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """
        학습 시, Batch_size 만큼 옵션에 맞추어서 데이터를 반환
        :param batch_size:
         이번에 사용향 batch크기
        :return: 
         batch 크기의 이미지 데이터(np.array)
        """
        assert batch_size <= self._num_examples, (
            "batch_size : %s num_examples : %s" % (batch_size,
                                                   self._num_examples))

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            if self._shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._paths = self._paths[perm]
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        temp_images=[image.image_path_to_np(x) for x in self._paths[start:end]]
        temp_images=np.array(temp_images)

        temp_images = temp_images.astype(self._dtype)
        if self._serialize:
            # [num_examples, rows, columns, depth] -> [num_examples, rows*columns*depth]
            temp_images = temp_images.reshape(temp_images.shape[0], temp_images.shape[1] * temp_images.shape[2] * temp_images.shape[3])

        if self._normalize:
            # [0,255] -> [0,1]
            temp_images = np.multiply(temp_images, 1.0 / 255.0)

        return temp_images


if __name__=="__main__":
    """
    Test code...
    """
    # TODO : 테스트 해봐야 함!