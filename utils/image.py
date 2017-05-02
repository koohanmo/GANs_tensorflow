from scipy import misc
from utils import path
import numpy as np

def image_path_to_np(path):
    img = misc.imread(path)
    return img

def image_dir_to_np(dir):
    files = path.getFileList(dir)
    if files is None :
        raise FileNotFoundError
    np_images = [image_path_to_np(file) for file in files]
    np_images = np.array(np_images)
    return np_images

if __name__ == "__main__":
    """
    Test code...
    """
    imageDir = "D:/Project/GANs_tensorflow/Image/origin"
    i = image_dir_to_np(imageDir)
    print(type(i))
    print(i.shape)