#-*- coding: utf-8 -*-
import numpy as np
from moviepy.editor import *
import moviepy.video.fx.all as vfx
import os

"""
If you see an error when you import moviepy.editor:
NeedDownloadError: Need ffmpeg exe.

import imageio
imageio.plugins.ffmpeg.download()
"""

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
     잘라낸 클립 명
     Ex) FULLMETAL ALCHEMIST-01_clip.mp4
    """
    original = VideoFileClip(filename)
    print(original.fps)
    clip = original.subclip(opening, ending)
    clip_name = filename.split('.')[0] + "_clip.mp4"
    clip.write_videofile(clip_name)
    return clip_name


def editVideoResize(filename, height, low_height = 90):
    """
    영상크기 변환(화질낮추기)
    :param filename:
     영상 파일명
     Ex) FULLMETAL ALCHEMIST-01.avi
    :param height:
     영상 세로값
    :param low_height:

    :return:
     Resize된 클립 명
     Ex) E240FULLMETAL ALCHEMIST-01_clip.mp4
    """
    width = (height / 3) * 4
    if(height == 320):
        low_width = (low_height/3) * 4
        dummy = VideoFileClip(filename).resize((low_width, low_height))
        original = dummy.resize( (width, height) )
    elif(height == 480):
        original = VideoFileClip(filename).resize((width, height))
    newFilename = "E" + str(height) + filename
    original.write_videofile(newFilename)
    return newFilename


def extractFrame(videoname):
    """
    영상에서 프레임 추출
    :param videoname:
     영상 파일명
     Ex) FULLMETAL ALCHEMIST-01.avi
    """
    vertical_flip = lambda frame: frame[::] # rotate 180 [::-1]
    clip = VideoFileClip(videoname)
    if(not os.path.exists(videoname.split('.')[0])):
        os.makedirs(videoname.split('.')[0])
    clip.fl_image(vertical_flip).to_images_sequence(videoname.split('.')[0]+"/iamges%05d.jpeg")


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


def distort(filename):
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
    """
    original = VideoFileClip(filename)
    BlacknWhite_clip = original.fx(vfx.blackwhite)
    Daken_clip = original.fx(vfx.colorx, 0.3)
    Invert_clip = original.fx(vfx.invert_colors)
    LRflip_clip = original.fx(vfx.mirror_x)
    UDflip_clip = original.fx(vfx.mirror_y)
    LRUDflip_clip = LRflip_clip.fx(vfx.mirror_y)
    LumCon_clip = original.fx(vfx.lum_contrast,100)
    Grennish_clip = original.fl_image(invert_green)
    Reddish_clip = original.fl_image(invert_red)
    Blueish_clip = original.fl_image(invert_blue)

    BlacknWhite_clip.write_videofile("BW "+filename)
    Daken_clip.write_videofile("DK "+filename)
    Invert_clip.write_videofile("IV "+filename)
    LRflip_clip.write_videofile("LR "+filename)
    LRUDflip_clip.write_videofile("FL "+filename)
    UDflip_clip.write_videofile("UD "+filename)
    LumCon_clip.write_videofile("LC "+filename)
    Grennish_clip.write_videofile("GR "+filename)
    Reddish_clip.write_videofile("RD "+filename)
    Blueish_clip.write_videofile("BR "+filename)



if __name__ == '__main__':
    filename = "input_video.avi"
    opening = '00:00:00'
    ending = '00:01:00'

    extractVideoFilename = editVideoCut(filename, opening = opening, ending = ending) #영상자르기
    editVideoResize(extractVideoFilename, 480)  #영상resize
    editVideoResize(extractVideoFilename, 320)  #영상resize

    extractFrame('E320input_video_clip.mp4')    #영상 프레임추출
    extractFrame('E480input_video_clip.mp4')    #영상 프레임추출
    distort('E320input_video_clip.mp4')         #영상 변형
