import os
import glob
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import requests
import tarfile

def preprocess():

    print('... loading data')
    if not os.path.exists('data/raw'):
        os.mkdir('data/raw')
    if not os.path.exists('data/raw/train'):
        os.mkdir('data/raw/train')
    if not os.path.exists('data/raw/test'):
        os.mkdir('data/raw/test')
    if not os.path.exists('data/raw/npy'):
        os.mkdir('data/raw/npy')

    conans = glob.glob('data/conan/*')
    paths = np.array([x for x in conans])
    np.random.shuffle(paths)

    r = int(len(paths) * 0.95)
    train_paths = paths[:r]
    test_paths = paths[r:]

    # make train data-set
    x_train = []
    pbar = tqdm(total=(len(train_paths)))
    for i, d in enumerate(train_paths):
        pbar.update(1)
        image = cv2.imread(d)
        img = cv2.resize(image, (96, 96))

        if img is None:
            continue
        x_train.append(img)
        name = "{}.jpg".format("{0:08d}".format(i))
        imgpath = os.path.join('data/raw/train', name)
        cv2.imwrite(imgpath, img)
    pbar.close()
    print('finish..... make train npy')

    # make test data-set
    x_test = []
    pbar = tqdm(total=(len(test_paths)))
    for i, d in enumerate(test_paths):
        pbar.update(1)
        image = cv2.imread(d)
        img = cv2.resize(image, (96,96))
        if img is None:
            continue
        x_test.append(img)
        name = "{}.jpg".format("{0:08d}".format(i))
        imgpath = os.path.join('data/raw/test', name)
        cv2.imwrite(imgpath, img)
    pbar.close()

    x_train = np.array(x_train, dtype=np.uint8)
    x_test = np.array(x_test, dtype=np.uint8)
    np.save('data/npy/x_train.npy', x_train)
    np.save('data/npy/x_test.npy', x_test)



def main():
    preprocess()


if __name__ == '__main__':
    main()

