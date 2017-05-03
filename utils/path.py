import os

# path variables about video
videos = ['conan', 'naruto', 'onepiece']
videoDir = "E:\Project\GANs_tensorflow\Video"
originVideoDir = os.path.join(videoDir, 'origin')
downgradeVideoDir = os.path.join(videoDir, 'downgrade')
imageDir = "E:\Project\GANs_tensorflow\Image"
originImageDir = os.path.join(imageDir, 'origin')
downgradeImageDir = os.path.join(imageDir, 'downgrade')

def setVideoOriginDirPath(name, option='original'):
    """
    'name'이라는 영상의 Video 폴더 생성
    :param name:
     동영상 이름
     Ex) conan
    :return:
     'name'의 만들어진 option 폴더 경로
     Ex) 'Video/origin/conan/original'
    """
    dirname = os.path.join(originVideoDir, name, option)
    print(dirname)
    if not os.path.isdir(os.path.join(originVideoDir, name)):
        os.mkdir(os.path.join(originVideoDir, name))
    os.mkdir(dirname)
    return dirname


def getVideoOriginDirPath(name, option='original'):
    """
    'name'이라는 영상의 Video 폴더
    :param name:
     동영상 이름
     Ex) conan
    :return:
     'name'의 원본 비디오 폴더 경로
     Ex) 'Video/origin/conan'
    """
    ret = os.path.join(originVideoDir, name, option)
    if (not os.path.exists(ret)):
        print('not exist option directory')
        return "None"
    else:
        return ret


def setVideoDowngradeDirPath(name, option='original'):
    """
    'name'이라는 영상의 Downgrade Video 폴더 생성
    :param name:
     동영상 이름
     Ex) conan
    :return:
     'name'의 만들어진 option 폴더 경로
     Ex) 'Video/downgrade/conan/original'
    """
    dirname = os.path.join(downgradeVideoDir, name, option)
    print(dirname)
    if not os.path.isdir(os.path.join(downgradeVideoDir, name)):
        os.mkdir(os.path.join(downgradeVideoDir, name))
    os.mkdir(dirname)
    return dirname


def getVideoDowngradeDirPath(name, option='original'):
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
     Ex) 'Video/downgrade/conan/crop'
    """
    ret = os.path.join(downgradeVideoDir, name,option)
    if(not os.path.exists(ret)):
        print('not exist option directory')
        return "None"
    else:
        return ret


def getImageOriginDirPath(name, option='original'):
    """
    'name'이라는 영상의 Image(frame으로 자른) 폴더
    :param name:
     동영상 이름
     Ex) conan
    :return:
     'name' 이미지 폴더
     Ex) 'ImageFiles/origin/conan'
    """
    ret = os.path.join(originImageDir, name, option)
    if (not os.path.exists(ret)):
        print('not exist option directory')
        return 'None'
    else:
        return ret


def getImageDowngradeDirPath(name, option='original'):
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
     Ex) 'Image/downgrade/conan/crop'
    """
    ret = os.path.join(downgradeImageDir , name , option )
    if (not os.path.exists(ret)):
        print('not exist option directory')
        return 'None'
    else:
        return ret


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
    """
    print(dir)
    print(os.listdir(dir))
    dirList = [os.path.join(dir,x) for x in os.listdir(dir) if os.path.isdir(x)]
    """
    dirList = []
    for i in (os.listdir(dir)):
        belowDir = os.path.join(dir, i)
        if (os.path.isdir(belowDir)):
            dirList.append(belowDir)

    if (not dirList):
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
    fileList = [os.path.join(dir,x) for x in os.listdir(dir) if not os.path.isdir(x)]
    if (not fileList):
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
    if(nth<len(dirList)):
        if (not dirList[nth]):
            print('no nth file')
            return 'None'
        else:
            return os.path.join(dir,dirList[nth])


if __name__ == '__main__':
    """
    Test code...
    """
    print("------path.py------")
    print (videoDir)
    print(originVideoDir)
    print(downgradeVideoDir)
    print()
    print ('getVideoOrigin : ' + getVideoOriginDirPath(videos[0]))                 # videos[0]에 해당하는 origin/conan의 경로
    print('getVideoDowngrade : ' + getVideoDowngradeDirPath(videos[0]))              # videos[0]에 해당하는 downgrade/conan/original의 경로
    print('getVideoDowngrade : ' + getVideoDowngradeDirPath(videos[0],'darken'))    # videos[0]에 해당하는 downgrade/conan/darken의 경로
    print()
    print(originImageDir)
    print(downgradeImageDir)
    print()
    print('getImageOrigin : ' + getImageOriginDirPath(videos[0]))  # videos[0]에 해당하는 origin/conan/original의 경로(이미지)
    print('getImageOrigin : ' + getImageOriginDirPath(videos[0],'blacknwhite'))  # videos[0]에 해당하는 origin/conan/blacknwhite의 경로(이미지)
    print('getImageDowngrade : ' + getImageDowngradeDirPath(videos[0]))  # videos[0]에 해당하는 downgrade/conan/original의 경로(이미지)
    print('getImageDowngrade : ' + getImageDowngradeDirPath(videos[0],'invert'))  # videos[0]에 해당하는 downgrade/conan/invert의 경로(이미지)
    print()
    print('getList : ', getDirList(videoDir))  # videoDir 하위폴더 origin / downgrade 반환
    print('getList : ', getDirList(downgradeVideoDir))  # downgradeVideoDir 하위폴더 conan / onepiece / naruto 반환
    print('getList : ', getDirList(getVideoDowngradeDirPath(videos[0])))  # downgrade/conan/original의 하위폴더 없음
    print()
    print('getFileList : ', getFileList(getImageOriginDirPath(videos[0],'blacknwhite')))  # origin/conan/blacknwhite의 하위파일 반환
    print()
    print(getNthPath(getImageOriginDirPath(videos[0],'blacknwhite'), 3))  # original/conan의 n번째 파일 반환
    print(getNthPath(getImageDowngradeDirPath(videos[0],'blacknwhite'), 1))  # downgrade/conan/crop의 n번째 파일 반환
