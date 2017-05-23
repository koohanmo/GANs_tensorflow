import os

# path variables about video
videos = ['conan', 'naruto', 'onepiece']
videoDir = "D:\Dataset\Video"
origin_video_dir = os.path.join(videoDir,'origin')
hr_video_dir = os.path.join(videoDir, 'HR')
lr_video_dir = os.path.join(videoDir, 'LR')

imageDir = "D:\Dataset\Image"
hr_image_dir = os.path.join(imageDir, 'HR')
lr_image_dir = os.path.join(imageDir, 'LR')

def is_exists(path):
    if(not os.path.exists(path)):
        print('not exist directory or file')
        return None
    else:
        return path

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_origin_video_path(name,option='original'):
    """
    'name'이라는 영상의 원본 Video 폴더
    :param name:
     동영상 이름
     Ex) conan
    :param option:
     original : 원본
     edit : 오프닝, 엔딩 잘라낸 영상
    :return:
     'name'의 원본 비디오 폴더 경로
     Ex) 'Video/origin/conan'
    """
    ret = os.path.join(origin_video_dir,name,option)
    return is_exists(ret)

def get_hr_video_path(name,option='original'):
    """
    'name'이라는 영상의 고화질 Video 폴더
    :param name:
     동영상 이름
     Ex) conan
    :param option:
     동영상 의 옵션
    :return:
     'name'의 원본 비디오 폴더 경로
     Ex) 'Video/origin/conan'
    """
    ret = os.path.join(hr_video_dir,name,option)
    return is_exists(ret)

def get_lr_video_path(name, option='original'):
    """
    'name'이라는 영상의 저화질 Video 폴더
    :param name:
     동영상 이름
     Ex) conan
    :param option:
     동영상 의 옵션
    :return:
     'name'의 원본 비디오 폴더 경로
     Ex) 'Video/origin/conan'
    """
    ret = os.path.join(lr_video_dir, name, option)
    return is_exists(ret)


def get_hr_image_path(name, option='original'):
    """
    'name'이라는 영상의 Image(frame으로 자른) 폴더
    :param name:
     동영상 이름
     Ex) conan
    :return:
     'name' 이미지 폴더
     Ex) 'ImageFiles/origin/conan'
    """
    ret = os.path.join(hr_image_dir, name, option)
    return is_exists(ret)

def get_lr_image_path(name, option='original'):
    """
    'name'이라는 영상의 Image(frame으로 자른) 폴더
    :param name:
     동영상 이름
     Ex) conan
    :return:
     'name' 이미지 폴더
     Ex) 'ImageFiles/origin/conan'
    """
    ret = os.path.join(lr_image_dir, name, option)
    return is_exists(ret)


def getDirList(dir):
    """
    dir의 하위 dir 리스트 반환
    :param dir:
     특정 폴더
     Ex) Video
    :return:
     하위 폴더 리스트
     Ex) [Video/modifed, Video/origin]
    """
    allList = [os.path.join(dir,x) for x in os.listdir(dir)]
    dirList = [x for x in allList if os.path.isdir(x)]

    if not dirList:
        print('no below directories')
        return None
    else:
        return dirList


def getFileList(dir):
    """
    dir의 하위 file 리스트 반환
    :param dir:
     특정 폴더
     Ex) Video/origin/conan
    :return:
     하위 폴더 리스트
     Ex) [Video/origin/conan/01.mp4, Video/origin/conan/02.mp4]
    """
    allList = [os.path.join(dir, x) for x in os.listdir(dir)]
    fileList = [x for x in allList if not os.path.isdir(x)]
    if not fileList:
        print('no below files')
        return None
    else:
        return fileList


def getNthPath(dir, nth):
    """
    dir에서 n번째 파일 경로를 반환
    :param originDir:
     다른 함수로 얻은 dir 경로,(이미지나 비디오가 들어있는 폴더)
     Ex) ImageFiles/origin/conan
    :param nth:
     dir에 있는 파일 중 n번째 경로
    :return:
     이미지 경로 반환
     Ex) ImageFiles/origin/conan/frame001.png
    """
    dirList = os.listdir(dir)
    if nth>=len(dirList):
        print('no nth file')
        return None
    return os.path.join(dir,dirList[nth])

    # if nth<len(dirList):
    #     if not dirList[nth]:
    #         print('no nth file')
    #         return None
    #     else:
    #         return os.path.join(dir,dirList[nth])


if __name__ == '__main__':
    """
    Test code...
    """
    print("------path.py------")
