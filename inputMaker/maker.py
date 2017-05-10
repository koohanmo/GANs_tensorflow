import utils
import os
"""
 특정 option의 Video data를 image data로 변환
"""

HR_res = {'height' : 640, 'width' : 480}
LR_res = {'height' : 320, 'width' : 240}

def makeSRInput(name, option, resize=True):

    videoDir =None
    res = HR_res
    if resize: res = LR_res
    videoDir = utils.path.getVideoOriginDirPath(name,option)

    videofiles = utils.path.getFileList(videoDir)

    # resize
    for file in videofiles:
        utils.video.editVideoResize(name,file,res['height'],res['width'],resize)

    if resize : videoDir = utils.path.getVideoDowngradeDirPath(name,option)

    # cut


if __name__=='__main__':

    # dir = utils.path.getVideoOriginDirPath('onepiece')
    # files = utils.path.getFileList(dir)
    #
    # x=1
    # for file in files:
    #     os.rename(file,os.path.join(dir,'onepiece'+str(x)+'.mp4'))
    #     x+=1

    makeSRInput('conan','original',resize=True)