#-*- coding: utf-8 -*-
import numpy as np
from moviepy.editor import *
import moviepy.video.fx.all as vfx
import os
import imageio
from . import path

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


def editVideoCut(name, filename, opening, ending):
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
    edit_videodir = path.get_origin_video_path(name, option='edit')
    clip_name = os.path.join(edit_videodir, os.path.basename(filename))

    if not os.path.exists(clip_name):
        clip = VideoFileClip(filename,audio=False).subclip(opening,ending)
        path.make_dir(edit_videodir)
        clip.write_videofile(clip_name)
    return clip_name


def save_video(path,clip):
    dir_path = os.path.dirname(path)
    path.make_dir(dir_path)
    if not os.path.exists(path):
        clip.write_videofile(path)
    else : print("Already Exist")
    return path


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

    resized = VideoFileClip(filename).resize((width, height))
    clip_name=None
    if resize : clip_name = path.get_lr_video_path(name,'origianl')
    else : clip_name =  path.get_hr_video_path(name,'origianl')
    clip_name = os.path.join(clip_name,os.path.basename(filename))
    clip_name = save_video(clip_name,resized)
    return clip_name


def extractFrame(name, videofile, option, resize):
    """
    Origin 영상에서 프레임 추출
    :param videoname:
     영상 파일명
     Ex) FULLMETAL ALCHEMIST-01.avi
    :param option:
     변형방식
    :param resize:
     True : LR
     False : HR
    """
    dirpath =None
    if resize : dirpath = path.get_lr_image_path(name, option)
    else: dirpath = path.get_hr_image_path(name, option)
    path.make_dir(dirpath)

    image_name_format = os.path.join(dirpath, os.path.basename(videofile).split('.')[0])
    firstImage = os.path.join(image_name_format+"_00001.jpeg")
    if not os.path.exists(firstImage):
        print('Making...')
        vertical_flip = lambda frame: frame[::]  # rotate 180 [::-1]
        clip = VideoFileClip(videofile)
        clip.fl_image(vertical_flip).to_images_sequence(image_name_format + "_%05d.jpeg")
    else: print('Already Exist')


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