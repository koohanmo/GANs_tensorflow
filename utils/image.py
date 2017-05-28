from scipy import misc
from utils import path
import numpy as np
from datetime import datetime
from PIL import Image
from PIL import ImageFilter


def image_path_to_np(path):
    img = misc.imread(path)
    return img

def image_dir_to_np(dir):
    files = path.getFileList(dir)
    if files is None :
        raise FileNotFoundError
    np_images = [image_path_to_np(file) for file in files]
    np_images = np.array(np_images)
    return np_images






def resize(img,factor,intp = Image.NEAREST):
    """
        이미지를 (interpolation을 적용하여) 확대/축소.
        
        :param img:
         Image 타입의 이미지
                   
        :param factor:
         배율. 
         eg. 2는 가로, 세로 각각 2배로 확대
             0.5는 가로, 세로 각각 2배로 축소
             
        :param intp:
         (확대 시) 보간 방법.
         default;       Image.NEAREST 
         candidates;    Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING, Image.BICUBIC, Image.LANCZOS 
          
         
        :return:
         확대/축소된 이미지(Image 타입)
        
        사용 예)  resized_image = resize(I_LR, 2) : NEAREST를 적용하여 확대된 이미지.
                  resized_image = resize(I_LR, 0.5) : 2배로 축소된 이미지.
                  resized_image = resize(I_LR, 2, Image.BICUBIC) : bicubic보간을 적용하여 확대된 이미지.
                  
    """

    w, h = img.size
    img = img.resize((int(w*factor), int(h*factor)), intp)
    #img.save("resized_img.png")
    return img

def apply_filter(img, filt=ImageFilter.GaussianBlur):
    """
        이미지에 필터를 적용하여 반환.
        
        :param img:
        Image 타입의 이미지
        
        :param filt:
        적용할 필터.
        default;        ImageFilter.GaussianBlur        
        candidates;     ImageFilter.BLUR			: 기본 블러
                        ImageFilter.GaussianBlur    : 가우시안 블러
                        ImageFilter.CONTOUR     	: 윤곽
                        ImageFilter.DETAIL			: 디테일살리기
                        ImageFilter.EDGE_ENHANCE	: 경계선 강화
                        ImageFilter.EDGE_ENHANCE_MORE 경계선 더 강화
                        ImageFilter.EMBOSS			: 양각무늬
                        ImageFilter.FIND_EDGES		: 경계선 찾기
                        ImageFilter.SMOOTH			: 부드럽게
                        ImageFilter.SMOOTH_MORE		: 더 부드럽게
                        ImageFilter.SHARPEN			: 날카롭게
                        ImageFilter.UnsharpMask     : 언샵
                        
                        ImageFilter.MedianFilter  	: Median 필터 적용(커널내 중앙값으로 set)
                        ImageFilter.MinFilter		: Min 필터 적용(커널내 최솟값으로 set)
                        ImageFilter.MaxFilter		: Max 필터 적용(커널내 최댓값으로 set)
                        ImageFilter.ModeFilter		: Mode 필터 적용(??)
                        
                        ImageFilter.RankFilter(size ,rank)	: 사이즈 내 n번째 등수로 set
                        ImageFilter.Kernel(size ,kernel ,scale=None ,offset=0) : 커널 직접 설정         
        
        :return:
        필터가 적용된 이미지(Image 타입)
    
        사용 예)  edge_enhanced_image = apply_filter(img,ImageFilter.EDGE_ENHANCE) # 경계선 강화필터 적용
                  gaussian_noised_image = apply_filter(img,ImageFilter.GaussianBlur) # 가우시안 블러 적용
    """

    img = img.filter(filt)
    # img.save(‘filtered_image.png')
    return img

def merge(imgs,
          save_option = False,
          center_option = False,
          view_option = True,
          save_path = "../sample/" ):
    """
        이미지 여러개를 합쳐서 하나의 이미지로 생성.

        :param img:
         Image 타입의 2차원 array
         eg. [ [이미지1-1,이미지1-2,이미지1-3], [이미지2-1,이미지2-2,이미지2-3], [이미지3-1,이미지3-2,이미지3-3] ]
=]지 결정하는 옵션. 
         True : 저장 / False : 저장하지않음 

        :param view_option:
         결과 이미지를 열어 확인할지 말지 결정하는 옵션.
         True : 확인 / False : 확인하지않음

        :return:
         합쳐 통합된 이미지(Image 타입)

        사용 예)  arr = []
                  arr.append([I_LR, I_blur, I_resized, I_bicubic])
                  arr.append([I_LR, I_blur, I_resized, I_bicubic])                
                  merged_image = merge(arr, save_option=True, center_option=True)
    """
    row_num = len(imgs) # row 개수
    column_num = 0      # column 개수
    one_width = 0
    one_height = 0

    # 결과 이미지 사이즈 결정
    for r in range(row_num):
        # 열 개수 결정
        if len(imgs[r]) > column_num:
            column_num = len(imgs[r])

        for c in range(len(imgs[r])):
            w, h = imgs[r][c].size
            if w > one_width:
                one_width = w   # w 결정
            if h > one_height:
                one_height = h  # h 결정

    width = one_width * column_num
    height = one_height * row_num
    result_img = Image.new("RGBA", (width, height))
    for r in range(row_num):
        for c in range(len(imgs[r])):
            if center_option:
                result_img.paste(
                    imgs[r][c],
                    (
                        c * one_width + int((one_width-imgs[r][c].size[0])/2),
                        r * one_height + int((one_height-imgs[r][c].size[1])/2)
                    )
                )
            else:
                result_img.paste(
                    imgs[r][c],
                    (
                        c * one_width,
                        r * one_height
                    )
                )

    if save_option:
        file_name = str(datetime.now().strftime('%Y-%m-%d %H%M%S'))
        result_img.save(save_path + file_name + ".png")
    if view_option:
        result_img.show()
    return result_img











if __name__ == "__main__":
    """
    Test code1
    """
    # imageDir = "D:/Project/GANs_tensorflow/Image/origin"
    # i = image_dir_to_np(imageDir)
    # print(type(i))
    # print(i.shape)Z

    
    
    """
    테스트코드
    """
    input_filename = "LR.png"
    I_LR = Image.open(input_filename)

    I_blur = apply_filter(I_LR)                  # BLUR 필터 적용(default)
    # I_blur.save("BLUR_" + input_filename)
    # I_blur.show()

    I_resized = resize(I_LR, 2)                  # NEAREST로 보간하여 resize(default)
    # I_resized.save("RESIZE_" + input_filename)
    # I_resized.show()

    I_bicubic = resize(I_LR, 2, Image.BICUBIC)   # BICUBIC으로 보간하여 resize
    # I_bicubic.save("BICUBIC_" + input_filename)
    # I_bicubic.show()


    arr = []
    arr.append([I_LR, I_blur, I_resized, I_bicubic])
    arr.append([I_LR, I_blur, I_bicubic])
    arr.append([I_LR, I_blur, I_resized, I_bicubic])

    I_merged = merge(arr, save_option=True, center_option=True)
    # I_merged = merge(arr, save_option=True, center_option=True, view_option=False)

    I_LR.close() # 메모리 관리를 위해 close