import os

# path variables about video
videos = ['conan', 'naruto', 'onepiece']
videoDir = "D:/Project/VSRGAN/Video"
originVideoDir = os.path.join(videoDir, 'origin')
modifiedVideoDir = os.path.join(videoDir, 'modified')


def getVideoOriginDirPath(name):
    """
    'name'이라는 영상의 Video 폴더
    :param name:
     동영상 이름
     Ex) conan
    :return: 
     'name'의 원본 비디오 폴더 경로
     Ex) 'Video/origin/conan'
    """

def getVideoModifedDirPath(name, option='crop'):
    """
    'name'이라는 영상의 Image에 변형을 가한 폴더
    :param name:
     동영상 이름
     Ex) conan
    :param option:
     이미지에 가한 변형
     Ex) crop, rotation, ......etc
    :return: 
     'name'의 원본 비디오를 option을 가한 비디오 폴더 경로
     Ex) 'Video/modified/conan/crop'
    """

def getImageOriginDirPath(name):
    """
    'name'이라는 영상의 Image(frame으로 자른) 폴더
    :param name:
     동영상 이름
     Ex) conan
    :return: 
     'name' 이미지 폴더
     Ex) 'ImageFiles/origin/conan'
    """

def getImageModifedDirPath(name, option='crop'):
    """
    'name'이라는 영상의 Image에 변형을 가한 폴더
    :param name:
     동영상 이름
     Ex) conan
    :param option:
     이미지에 가한 변형
     Ex) crop, rotation, ......etc
    :return: 
     'name' 이미지에 option을 가한 이미지 폴더
     Ex) 'Image/modified/conan/crop'
    """


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


if __name__ == '__main__':
    """
    Test code...
    """
    print("------path.py------")
    print (videoDir)
    print(os.listdir(originVideoDir))
    print(os.listdir(modifiedVideoDir))
