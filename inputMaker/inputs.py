import numpy as np
import scipy as sp
import collections
import shape
import utils

class Datasets(object):
    """
    train, test, validation 3가지로 나누어서
    데이터셋을 반환 할 때 사용하는 클래스
    """

def load_image_data_without_label(
                                    name,
                                    option ='original',
                                    resize=True,
                                    pathload=False,
                                    normalize=False,
                                    serialize=False,
                                    dtype=np.float32,
                                    shuffle=True,
                                    ratio=[6,3,1]
                                    ):
    """
    lable이 없는 이미지를 읽어서 데이터셋을 반환
    :param name: 
     동영상의 이름
    :param option: 
     어떤 처리를 가한 이미지를 가져올 것인지
    :param resize: 
     사이즈를 줄일 것인지
    :param pathload: 
     이미지의 경로를 가지고 있다가 사용 할 때 로드 할 것인지
    :param normalize: 
     [0, 255] -> [0, 1] 로 정규화 할 것인지
    :param serialize:
     이미지를 1차원으로 펼칠 지
    :param dtype: 
     데이터의 타입
     Ex ) np.float32
    :param shuffle: 
     데이터셋을 섞을지
    :param ratio: 
     train, test, validaion의 비율
    :return: 
     Datasets라는 클래스  반환(train, test, validation)
    """
    datasets = Datasets()

    images_dir_path=None
    if resize : images_dir_path = utils.path.getImageDowngradeDirPath(name, option)
    else : images_dir_path = utils.path.getImageOriginDirPath(name, option)

    images_files_path = utils.path.getFileList(images_dir_path)

    # 데이터 분할 train : test :validation
    tot_data = len(images_files_path)
    tot_ratio = sum(ratio)
    train_start = 0
    train_end = int((ratio[0]/tot_ratio)* tot_data)
    test_start =train_end
    test_end =int((ratio[0]+ratio[1])/tot_ratio* tot_data)
    validation_start = test_end
    validation_end =tot_data

    images_train = [utils.image.image_path_to_np(x) for x in images_files_path[train_start:train_end] ]
    images_train = np.array(images_train)
    images_test = [utils.image.image_path_to_np(x) for x in images_files_path[test_start:test_end]]
    images_test = np.array(images_test)
    images_validatiaon = [utils.image.image_path_to_np(x) for x in images_files_path[validation_start:validation_end]]
    images_validatiaon = np.array(images_validatiaon)

    if pathload:
        datasets.train = shape.ImagePathSetWithoutLable(images_files_path[train_start:train_end],
                                                              normalize=normalize,
                                                              serialize=serialize,
                                                              dtype=dtype,
                                                              shuffle=shuffle)
        datasets.validation = shape.ImagePathSetWithoutLable(images_files_path[test_start:test_end],
                                                                   normalize=normalize,
                                                                   serialize=serialize,
                                                                   dtype=dtype,
                                                                   shuffle=shuffle)
        datasets.test = shape.ImagePathSetWithoutLable(images_files_path[validation_start:validation_end],
                                                             normalize=normalize,
                                                             serialize=serialize,
                                                             dtype=dtype,
                                                             shuffle=shuffle)
    else :
        datasets.train = shape.ImageSetWithoutLable(images_train,
                                                         normalize=normalize,
                                                         serialize=serialize,
                                                         dtype=dtype,
                                                         shuffle=shuffle)
        datasets.test = shape.ImageSetWithoutLable(images_test,
                                                               normalize=normalize,
                                                               serialize=serialize,
                                                               dtype=dtype,
                                                               shuffle=shuffle)
        datasets.validation = shape.ImageSetWithoutLable(images_validatiaon,
                                                         normalize=normalize,
                                                         serialize=serialize,
                                                         dtype=dtype,
                                                         shuffle=shuffle)

    return datasets

def load_image_data_with_label(
                                    name,
                                    option ='original',
                                    label = True,
                                    resize=True,
                                    pathload=False,
                                    normalize=False,
                                    serialize=False,
                                    dtype=np.float32,
                                    shuffle=True,
                                    ratio=[6,3,1]
                                    ):
    """
    이미지를 읽어서 데이터셋을 반환
    :param name: 
     동영상의 이름
    :param option: 
     어떤 처리를 가한 이미지를 가져올 것인지
    :param label: 
     레이블이 1인지 0인지
    :param resize: 
     사이즈를 줄일 것인지
    :param pathload: 
     이미지의 경로를 가지고 있다가 사용 할 때 로드 할 것인지
    :param normalize: 
     [0, 255] -> [0, 1] 로 정규화 할 것인지
    :param serialize:
     이미지를 1차원으로 펼칠 지
    :param dtype: 
     데이터의 타입
     Ex ) np.float32
    :param shuffle: 
     데이터셋을 섞을지
    :param ratio: 
     train, test, validaion의 비율
    :return: 
     Datasets라는 클래스  반환(train, test, validation)
    """
    datasets = Datasets()

    images_files_path=None
    if resize : images_dir_path = utils.path.getImageDowngradeDirPath(name, option)
    else : images_dir_path = utils.path.getImageOriginDirPath(name, option)

    print(images_files_path)
    images_files_path = utils.path.getFileList(images_dir_path)

    # 데이터 분할 train : test :validation
    tot_data = len(images_files_path)
    tot_ratio = sum(ratio)
    train_start = 0
    train_end = int((ratio[0]/tot_ratio)* tot_data)
    test_start =train_end
    test_end =int((ratio[0]+ratio[1])/tot_ratio* tot_data)
    validation_start = test_end
    validation_end =tot_data

    images_train = [utils.image.image_path_to_np(x) for x in images_files_path[train_start:train_end] ]
    images_train = np.array(images_train)
    images_test = [utils.image.image_path_to_np(x) for x in images_files_path[test_start:test_end]]
    images_test = np.array(images_test)
    images_validatiaon = [utils.image.image_path_to_np(x) for x in images_files_path[validation_start:validation_end]]
    images_validatiaon = np.array(images_validatiaon)

    if pathload:
        datasets.train = shape.ImageSetWithLabel(images_files_path[train_start:train_end],
                                                              normalize=normalize,
                                                              serialize=serialize,
                                                              dtype=dtype,
                                                              shuffle=shuffle)
        datasets.test = shape.ImageSetWithLabel(images_files_path[test_start:test_end],
                                                                   normalize=normalize,
                                                                   serialize=serialize,
                                                                   dtype=dtype,
                                                                   shuffle=shuffle)
        datasets.validation = shape.ImageSetWithLabel(images_files_path[validation_start:validation_end],
                                                             normalize=normalize,
                                                             serialize=serialize,
                                                             dtype=dtype,
                                                             shuffle=shuffle)
    else :
        datasets.train = shape.ImagePathSetWithLable(images_train,
                                                         normalize=normalize,
                                                         serialize=serialize,
                                                         dtype=dtype,
                                                         shuffle=shuffle)
        datasets.test = shape.ImagePathSetWithLable(images_test,
                                                               normalize=normalize,
                                                               serialize=serialize,
                                                               dtype=dtype,
                                                               shuffle=shuffle)
        datasets.validation = shape.ImagePathSetWithLable(images_validatiaon,
                                                         normalize=normalize,
                                                         serialize=serialize,
                                                         dtype=dtype,
                                                         shuffle=shuffle)

    return datasets

def load_video_data(name):

    pass

def load_video_data_with_label(name):

    pass


if __name__=="__main__":
    """
    Test code...
    """
    a= load_image_data_without_label('conan',resize=False,pathload=True)
    for x in range(10):
        print(type(a.train.next_batch(3)))
