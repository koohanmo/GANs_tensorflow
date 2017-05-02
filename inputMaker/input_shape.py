import numpy as np
from utils import image

class ImageSetWithoutLable(object):
    def __init__(self,
                 images,
                 normalize=False,
                 serialize=False,
                 dtype=np.float32,
                 shuffle=True):

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
        assert batch_size <= self._num_examples, (
            "batch_size : %s num_examples : %s" % (batch_size,
                                                   self._num_examples))

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            if self._shuffle:
                perm = np.arrange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]

            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._images[start:end]


class ImageSetWithLabel(object):
    def __init__(self,
                 images,
                 labels,
                 normalize=False,
                 serialize=False,
                 dtype=np.float32,
                 shuffle=True):

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
        assert batch_size <= self._num_examples, (
            "batch_size : %s num_examples : %s" % (batch_size,
                                                   self._num_examples))

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            if self._shuffle:
                perm = np.arrange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class ImagePathSetWithLable(object):
    def __init__(self,
                 paths,
                 labels,
                 normalize=False,
                 serialize=False,
                 dtype=np.float32,
                 shuffle=True):

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
        assert batch_size <= self._num_examples, (
            "batch_size : %s num_examples : %s" % (batch_size,
                                                   self._num_examples))

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            if self._shuffle:
                perm = np.arrange(self._num_examples)
                np.random.shuffle(perm)
                self._paths = self._paths[perm]
                self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

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
        self._num_examples = paths.shape[0]
        self._paths = paths
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples, (
            "batch_size : %s num_examples : %s" % (batch_size,
                                                   self._num_examples))

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            if self._shuffle:
                perm = np.arrange(self._num_examples)
                np.random.shuffle(perm)
                self._paths = self._paths[perm]

            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        temp_images=[image.image_path_to_np(x) for x in self._images[start:end]]
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