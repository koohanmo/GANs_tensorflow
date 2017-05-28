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






def resize(img,factor,intp = Image.NEAREST): # LR이미지를 bicubic interpolation을 이용하여 SR.
    w, h = img.size
    img = img.resize((int(w*factor), int(h*factor)), intp)
    #img.save("resized_img.png")
    return img

def apply_filter(img, filt=ImageFilter.GaussianBlur):
    # eg.  edge_enhanced_image = apply_filter(img,ImageFilter.EDGE_ENHANCE)

    # BLUR			    : 기본 블러
    # GaussianBlur      : 가우시안 블러
    # CONTOUR     		: 윤곽
    # DETAIL			: 디테일살리기
    # EDGE_ENHANCE		: 경계선 강화
    # EDGE_ENHANCE_MORE	: 경계선 더 강화
    # EMBOSS			: 양각무늬
    # FIND_EDGES		: 경계선 찾기
    # SMOOTH			: 부드럽게
    # SMOOTH_MORE		: 더 부드럽게
    # SHARPEN			: 날카롭게
    # UnsharpMask       : 언샵

    # MedianFilter  	: Median 필터 적용(커널내 중앙값으로 set)
    # MinFilter			: Min 필터 적용(커널내 최솟값으로 set)
    # MaxFilter			: Max 필터 적용(커널내 최댓값으로 set)
    # ModeFilter		: Mode 필터 적용(??)

    # RankFilter(size ,rank)	: 사이즈 내 n번째 등수로 set
    # Kernel(size ,kernel ,scale=None ,offset=0) : 커널 직접 설정

    # img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    img = img.filter(filt)
    # img.save(‘filtered_image.png')
    return img

def merge(imgs,
          save_option = False,
          center_option = False,
          view_option = True,
          save_path = "/sample_image/" ):
    # 이미지 여러개를 합쳐서 하나의 이미지로 보여줌.
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
        for c in range(column_num):
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
    Test code...
     """
    # imageDir = "D:/Project/GANs_tensorflow/Image/origin"
    # i = image_dir_to_np(imageDir)
    # print(type(i))
    # print(i.shape)Z


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
    arr.append([I_LR, I_blur, I_resized, I_bicubic])
    arr.append([I_LR, I_blur, I_resized, I_bicubic])

    I_merged = merge(arr, save_option=True, center_option=True)
    # I_merged = merge(arr, save_option=True, center_option=True, view_option=False)