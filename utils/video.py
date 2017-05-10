#-*- coding: utf-8 -*-
import numpy as np
from moviepy.editor import *
import moviepy.video.fx.all as vfx
import os
import imageio
from . import path

editedDir = "D:\Project\GANs_tensorflow\Edited"
header_list = ['BW', 'DK', 'IV', 'LR', 'UD', 'FL', 'LC', 'GR', 'RD', 'BL']
option_list = ['blacknwhite', 'darken', 'invert', 'lrflip', 'udflip', 'lrudflip','lumcon','greenish','reddish', 'blueish']

"""
If you see an error when you import moviepy.editor:
NeedDownloadError: Need ffmpeg exe.

import imageio
imageio.plugins.ffmpeg.download()
"""


# 안씀 보류 #
def editOpenEnd(filename,opening_startT, opening_endT, ending_startT, ending_endT):
    original = VideoFileClip(filename)
    clip1 = original.subclip(0,opening_startT)
    clip2 = original.subclip(opening_endT,ending_startT)
    clip3 = original.subclip(ending_endT)

    final_clip = concatenate_videoclips([clip1,clip2,clip3])

    filename = "Edited "+filename.split('.')[0]+".mp4"
    final_clip.write_videofile(filename)


def invert_green(image):
    """
    영상 초록색화
    """
    return image[:,:,[0,2,1]]


def invert_red(image):
    """
    영상 붉은색화
    """
    return image[:,:,[2,0,1]]


def invert_blue(image):
    """
    영상 파란색화
    """
    return image[:,:,[1,0,2]]


def distort(filename, option=0):
    """
    영상 변형
     - 흑백
     - 어둡게
     - 반전
     - 좌우반전
     - 상하반전
     - 상하좌우반전
     - 밝게
     - 초록색화
     - 붉은색화
     - 파란색화
    :param filename:
     영상 파일명
    :param option:
     origin(0) downgrade(1) 구분
    """
    clip_list = []
    original = VideoFileClip(filename)
    clip_list.append(original.fx(vfx.blackwhite))
    clip_list.append(original.fx(vfx.colorx, 0.3))
    clip_list.append(original.fx(vfx.invert_colors))
    clip_list.append(original.fx(vfx.mirror_x))
    clip_list.append(original.fx(vfx.mirror_y))
    clip_list.append(clip_list[4].fx(vfx.mirror_y))
    clip_list.append(original.fx(vfx.lum_contrast, 100))
    clip_list.append(original.fl_image(invert_green))
    clip_list.append(original.fl_image(invert_red))
    clip_list.append(original.fl_image(invert_blue))

    for clipIdx in range(len(clip_list)):
        newFilename = header_list[clipIdx] + os.path.basename(filename).split('.')[0] + ".mp4"
        print(newFilename)
        if option==0:
            clip_name = saveOriginVideo(newFilename, clip_list[clipIdx], option_list[clipIdx])
        else:
            clip_name = saveDowngradeVideo(newFilename, clip_list[clipIdx], option_list[clipIdx])


def editVideoCut(filename, opening, ending):
    """
    영상에서 오프닝과 엔딩제거
    :param filename:
     영상 파일명
     Ex) FULLMETAL ALCHEMIST-01.avi
    :param opening:
     오프닝 종료 시간
    :param ending:
     엔딩 시작 시간
    :return:

     Ex) 
    """
    original = VideoFileClip(filename)
    print(original.fps)
    clip = original.subclip(opening, ending)

    clip_name = os.path.join(editedDir, os.path.basename(filename).split('.')[0]) + ".mp4"
    # 주석풀것! - 잠시 뒤에 함수 테스트 하느라 주석
    clip.write_videofile(clip_name)
    return clip_name


def saveOriginVideo(filename,clip, option):
    """
        수정된 영상 폴더별 저장 - 원본이 될 애들 (480p)
        :param filename:
         영상 파일명
         Ex) FULLMETAL ALCHEMIST-01.avi
        :param clip:
         옵션
        :param option:
         옵션
        :return:
         편집영상 이름
         Ex) 
    """
    basename = os.path.basename(filename).split('_')[1]

    dirpath = path.getVideoOriginDirPath(basename, option)
    if not os.path.isdir(dirpath):
        dirpath = path.setVideoOriginDirPath(basename,option)

    clip_name = os.path.join(dirpath, os.path.basename(filename).split('.')[0]) + ".mp4"
    print(clip_name)

    if not os.path.exists(clip_name):
        print("Making...")
        clip.write_videofile(clip_name)
    else:
        print("Already Exist")
    return clip_name


def saveDowngradeVideo(name, filename, clip, option):
    """
        수정된 영상 폴더별 저장 - 저화질 애들
        :param filename:
         영상 파일명
         Ex) FULLMETAL ALCHEMIST-01.avi
        :param clip:
         옵션
        :param option:
         옵션
        :return:
         편집영상 이름
         Ex) 
    """

    dirpath = path.setVideoDowngradeDirPath(name, option)
    if not os.path.isdir(dirpath):
        dirpath = path.setVideoDowngradeDirPath(name, option)

    clip_name = os.path.join(dirpath, filename)
    if not os.path.exists(clip_name):
        print("Making...")
        clip.write_videofile(clip_name)
    else:
        print("Already Exist")
    return clip_name


def editVideoResize(name, filename, width, height, resize):
    """
    영상크기 변환(화질낮추기)
    :param filename:
     영상 파일명
     Ex) FULLMETAL ALCHEMIST-01.avi
    :param width:
     영상 가로값
    :param height:
     영상 세로값
    :return:
     Resize된 클립 명
     Ex) E240FULLMETAL ALCHEMIST-01.mp4
    """

    newFilename = str(height) + "_" + os.path.split(filename)[-1]
    resized = VideoFileClip(filename).resize((width, height))
    clip_name=None
    if resize : clip_name = saveDowngradeVideo(name, newFilename, resized, 'original')
    else : clip_name = saveOriginVideo(name, newFilename, resized, 'original')

    return clip_name


def setImageOriginPath(videoname, option):
    """
        Origin 영상에서 프레임 추출시 경로
        :param videoname:
         영상 파일명
         Ex) FULLMETAL ALCHEMIST-01.avi
        :param option:
         변형방식
        :return dirpath:
         경로 반환
    """
    basename = os.path.basename(videoname).split('_')[1]
    dirpath = path.getImageOriginDirPath(basename, option)

    if not os.path.isdir(dirpath):
        dirpath = path.setImageOriginDirPath(basename, option)

    dirpath = os.path.join(dirpath, os.path.basename(videoname).split('.')[0])

    return dirpath


def setImageDowngradePath(videoname, option):
    """
        Downgrade 영상에서 프레임 추출시 경로
        :param videoname:
         영상 파일명
         Ex) FULLMETAL ALCHEMIST-01.avi
        :param option:
         변형방식
        :return dirpath:
         경로 반환
    """
    basename = os.path.basename(videoname).split('_')[1]
    dirpath = path.getImageDowngradeDirPath(basename, option)

    if not os.path.isdir(dirpath):
        dirpath = path.setImageDowngradeDirPath(basename, option)

    dirpath = os.path.join(dirpath, os.path.basename(videoname).split('.')[0])

    return dirpath


def extractFrame(videoname, option, version):
    """
    Origin 영상에서 프레임 추출
    :param videoname:
     영상 파일명
     Ex) FULLMETAL ALCHEMIST-01.avi
    :param option:
     변형방식
    :param version:
     origin(0) downgrade(1)
    """
    if version==0:
        dirpath = setImageOriginPath(videoname, option)
    else:
        dirpath = setImageDowngradePath(videoname, option)

    print(dirpath)
    if not os.path.exists(dirpath+"_images00000.jpeg"):
        print('Making...')
        vertical_flip = lambda frame: frame[::]  # rotate 180 [::-1]
        clip = VideoFileClip(videoname)
        clip.fl_image(vertical_flip).to_images_sequence(dirpath + "_images%05d.jpeg")
    else:
        print('Already Exist')


if __name__ == '__main__':
    imageio.plugins.ffmpeg.download()
    filename = "D:\metalalchemist_01.avi"
    opening = '00:00:00'
    ending = '00:01:00'
    extractVideoFilename = editVideoCut(filename, opening=opening, ending=ending)  # 영상자르기 후 Edited에 저장

    # originOriginal = editVideoResize(extractVideoFilename, 480)  # 영상resize 후 origin - catoon - original에 저장
    # distort(originOriginal, 0)  # original원본 영상 변형

    #downgradeOriginal = editVideoResize(extractVideoFilename, 320)  # 영상resize 후 downgrade - catoon - original에 저장
    #distort(downgradeOriginal, 1)  # downgrade원본 영상 변형

    #filename2 = "E:\\Project\\GANs_tensorflow\\Video\\origin\\metalalchemist\\lrflip\\LR480_metalalchemist_01.mp4"
    #extractFrame(filename2, 'lrflip',0)  # original영상의 프레임추출
    #filename3 = "D:\\Project\\GANs_tensorflow\\Video\\downgrade\\metalalchemist\\lrflip\\LR320_metalalchemist_01.mp4"
    #extractFrame(filename3, 'lrflip', 1)  # original영상의 프레임추출

    # for문 돌면서 Video/origin/ 의 변형 영상리스트에서 프레임추출
    """
    folder = "D:\\Project\\GANs_tensorflow\\Video\\downgrade\\metalalchemist\\"
    for dir in path.getDirList(folder):
        files = path.getFileList(dir)
        if not files == None:
            for file in path.getFileList(dir):
                print(file)
                option = file.split("\\")[6]
                extractFrame(file, option, 1)
        else:
            print('no files')
    """