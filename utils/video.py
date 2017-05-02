import numpy as np
from moviepy.editor import *
import path
import moviepy.video.fx.all as vfx
import os

"""
If you see an error when you import moviepy.editor: 
NeedDownloadError: Need ffmpeg exe.

import imageio
imageio.plugins.ffmpeg.download()  
"""

def editVideoCut(filename, opening, ending):
    original = VideoFileClip(filename)
    print(original.fps)
    clip = original.subclip(opening, ending)
    clip_name = filename.split('.')[0] + "_clip.mp4"
    clip.write_videofile(clip_name)
    return clip_name

def editVideoResize(filename, height, low_height = 90):
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
    vertical_flip = lambda frame: frame[::] # rotate 180 [::-1]
    clip = VideoFileClip(videoname)
    if(not os.path.exists(videoname.split('.')[0])):
        os.makedirs(videoname.split('.')[0])
    clip.fl_image(vertical_flip).to_images_sequence(videoname.split('.')[0]+"/iamges%05d.jpeg")

def editOpenEnd(filename,opening_startT, opening_endT, ending_startT, ending_endT):
    original = VideoFileClip(filename)
    clip1 = original.subclip(0,opening_startT)
    clip2 = original.subclip(opening_endT,ending_startT)
    clip3 = original.subclip(ending_endT)

    final_clip = concatenate_videoclips([clip1,clip2,clip3])

    filename = "Edited "+filename.split('.')[0]+".mp4"
    final_clip.write_videofile(filename)
    #clip1.write_videofile("short.mp4")

def resize(filename, height):
    width = height/3*4
    original = VideoFileClip(filename).resize((width,height))
    original.write_videofile("E"+str(height)+filename)

def invert_green(image):
    return image[:,:,[0,2,1]]

def invert_red(image):
    return image[:,:,[2,0,1]]

def invert_blue(image):
    return image[:,:,[1,0,2]]

def invert_purple(image):
    return image[:,:,[1,0,1]]

def distort(filename):
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
    P_clip = original.fl_image(invert_purple)

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
    P_clip.write_videofile("P "+filename)

opening_startT = '00:00:59'
opening_endT = '00:02:29'
ending_startT = '00:22:27'
ending_endT = '00:23:57'
filename = "input_video.avi"
opening = '00:00:00'
ending = '00:01:00'
#extractVideoFilename = editVideoCut(filename, opening = opening, ending = ending)
#editVideoResize(extractVideoFilename, 480)
#editVideoResize(extractVideoFilename, 320)
#tempname = 'E320input_video_clip.mp4'
#extractFrame(tempname)
#extractFrame('E480input_video_clip.mp4')
#editOpenEnd(filename, opening_startT, opening_endT, ending_startT, ending_endT)
#resize("short.mp4",480)
#resize("short.mp4",240)
#distort("short.mp4")



if __name__ == '__main__':
    print('------video.py-----')


