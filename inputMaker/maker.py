import utils
import os
"""
 특정 option의 Video data를 image data로 변환
"""

HR_res = {'height' : 640, 'width' : 480}
LR_res = {'height' : 320, 'width' : 240}
cut_time = [360,-360]

def makeSRInput(option='original', resize=True):
    """
    SR에 사용될 데이터 셋을 생성
    :param option: 
     Video 처리 옵션
    :param resize: 
     True : Low resolution
     False : High resolution
    :return: 
    """
    for name in utils.path.videos:
        # cut : 앞뒤 영상
        videoDir = utils.path.get_origin_video_path(name,option)
        videofiles = utils.path.getFileList(videoDir)

        for video in videofiles:
            utils.video.editVideoCut(name, video,cut_time[0],cut_time[1])

        # resize
        videoDir = utils.path.get_origin_video_path(name,option='edit')
        videofiles = utils.path.getFileList(videoDir)

        res = HR_res
        if resize: res = LR_res

        for file in videofiles:
            utils.video.editVideoResize(name,file,res['height'],res['width'],resize)

        # extract frame
        if resize : videoDir = utils.path.get_lr_video_path(name,option)
        else :   videoDir = utils.path.get_hr_video_path(name,option)

        videofiles = utils.path.getFileList(videoDir)

        for videofile in videofiles:
            utils.video.extractFrame(name, videofile, option, resize)




if __name__=='__main__':

    # dir = utils.path.getVideoOriginDirPath('onepiece')
    # files = utils.path.getFileList(dir)
    #
    # x=1
    # for file in files:
    #     os.rename(file,os.path.join(dir,'onepiece'+str(x)+'.mp4'))
    #     x+=1+


    while True:
        try:
            makeSRInput('original', resize=False)
            makeSRInput('original', resize=True)
        except : continue