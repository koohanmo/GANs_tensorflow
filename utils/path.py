import os

# path variables about video
videos = ['conan','naruto','onepiece']
videoDir = "D:/Project/VSRGAN/Video"
originVideoDir = os.path.join(videoDir,'origin')
modifiedVideoDir = os.path.join(videoDir,'modified')



if __name__ == '__main__':
    print("------path.py------")
    print (videoDir)
    print(os.listdir(originVideoDir))
    print(os.listdir(modifiedVideoDir))
