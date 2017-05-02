import os

# path variables about video
videos = ['conan', 'naruto', 'onepiece']
videoDir = "D:\Project\VSRGAN\Video"
originVideoDir = os.path.join(videoDir, 'origin')
modifiedVideoDir = os.path.join(videoDir, 'modified')
imageDir = "F:\Project\VSRGAN\Image"
originImageDir = os.path.join(imageDir, 'origin')
modifiedImageDir = os.path.join(imageDir, 'modified')


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
    return os.path.abspath(originVideoDir+"\\"+name)


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
    ret = modifiedVideoDir+"\\"+name+"\\"+option

    if(not os.path.exists(ret)):
        print('not exist option directory')
        return
    else:
        return ret


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
    return os.path.abspath(originImageDir+'\\'+name)


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
    ret = modifiedImageDir + '\\' + name + '\\' + option
    if (not os.path.exists(ret)):
        print('not exist option directory')
        return
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
    dirList = []
    for i in (os.listdir(dir)):
        belowDir = dir + "\\" + i
        if(os.path.isdir(belowDir)):
            dirList.append(belowDir)

    if (not dirList):
        print('no below directories')
        return
    else:
        return dirList


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
    # print(nth, dirList[nth])
    if (not dirList[nth]):
        print('no nth file')
        return
    else:
        return dirList[nth]


if __name__ == '__main__':
    """
    Test code...
    """
    print("------path.py------")
    print (videoDir)
    print(os.listdir(originVideoDir))
    print(os.listdir(modifiedVideoDir))
    print()
    print ('getVideoOrigin : ' + getVideoOriginDirPath(videos[0]))
    print ('getVideoModified : ' + getVideoModifiedDirPath(videos[0]))
    print("getVideoModified File list : " , os.listdir(getVideoModifiedDirPath(videos[0])))
    print()
    print(originImageDir)
    print(modifiedImageDir)
    print()
    print ('getImageOrigin : ' + getImageOriginDirPath(videos[0]))
    print('getImageO List : ' , os.listdir(getImageOriginDirPath(videos[0])))
    print('getImageModified : ' + getImageModifiedDirPath(videos[0]))
    print()
    print('getList : ', getDirList(videoDir))
    print('getList : ', getDirList(modifiedVideoDir))
    print('getList : ', getDirList(getVideoModifiedDirPath(videos[0])))
    print()
    print (getNthPath(getImageOriginDirPath(videos[0]), 2))
    print(getNthPath(getImageModifiedDirPath(videos[0]), 2))
