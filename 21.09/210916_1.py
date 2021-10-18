from __future__ import print_function
import cv2 as cv2
import argparse, copy

def standard(src, size):  ## src = 불러온 이미지 소스 , size = 변경할 크기
    max = 0

    if size < src.shape[0] or size < src.shape[1]:
        if src.shape[0] > src.shape[1]:
            max = src.shape[0]
        elif src.shape[1] > src.shape[0]:
            max = src.shape[1]
        else:
            max = src.shape[0]
    else:
        print("확대 불가")
        max = size

    # print(max)

    scale = size / max
    return scale

def imgsize(src, size): ## src = 불러온 이미지 소스 , size = 변경할 크기
    scale = standard(src, size)
    return cv2.resize(src, None, fx=scale, fy=scale)

def imgcolor(src,Colortype=cv2.IMREAD_COLOR): ## src = 불러온 이미지 소스 , Colortype = 변경할 이미지 컬러(미지정시 자동 BGR컬로)
    return cv2.cvtColor(src,Colortype)

def Imgread(url, size, Colortype=cv2.IMREAD_COLOR):  ## url = 이미지 링크 , size = 변경할 크기 , 받아올 이미지 컬러타입 , Colortype = 변경할 이미지 컬러(미지정시 자동 BGR컬로)
    src = cv2.imread(url, Colortype)
    result = imgsize(src, size)
    return result


### 이진화

# img = Imgread("C:/Users/1/Desktop/images (1)/animal-05.jpg",500)
#
#
# img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# ret, img_d_1 = cv2.threshold(img_g,160,255,cv2.THRESH_BINARY)
# ret, img_d_2 = cv2.threshold(img_g,160,255,cv2.THRESH_BINARY_INV)
# ret, img_d_3 = cv2.threshold(img_g,160,255,cv2.THRESH_TRUNC)
#
# result = cv2.hconcat([img_d_1,img_d_2,img_d_3])
#
# cv2.imshow("result",result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

### 흐림효과

# max_value = 255
# thr = 160
# a = []
# b = []
#
# img = Imgread("C:/Users/1/Desktop/images (1)/animal-06.jpg",180)
#
# img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# temp_1 = cv2.blur(img_g,(7,7),anchor=(-1,-1),borderType=cv2.BORDER_DEFAULT)
#
# for i in range(10):
#     a.append(copy.deepcopy(temp_1))
#     temp_1 = cv2.blur(temp_1, (7, 7), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
#
# for i in range(10):
#     ret, c = cv2.threshold(a[i],thr,max_value,cv2.THRESH_BINARY)
#     b.append(c)
#
# # a = [temp_1,temp_d_1]
#
# result = cv2.vconcat([cv2.hconcat(a),cv2.hconcat(b)])
#
# cv2.imshow("result",result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Sobel

# img = Imgread("C:/Users/1/Desktop/images (1)/animal-10.jpg",500)
# img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# img_s_x = cv2.Sobel(img_g,cv2.CV_64F,1,0,ksize=3)
# img_s_x = cv2.convertScaleAbs(img_s_x)
#
# img_s_y = cv2.Sobel(img_g,cv2.CV_64F,1,0,ksize=3)
# img_s_y = cv2.convertScaleAbs(img_s_y)
#
# img_s = cv2.addWeighted(img_s_x,1.0,img_s_y,1.0,0)
#
# result =cv2.hconcat([img_g,img_s])
#
# cv2.imshow("result",result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## laplacian

# img = Imgread("C:/Users/1/Desktop/images (1)/animal-08.jpg",450)
# img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# img_s = cv2.Sobel(img_g,cv2.CV_8U,1,0,ksize=3) ## Soble
#
# img_laplacian = cv2.Laplacian(img_g, cv2.CV_8U, ksize=3)
# img_laplacian1 = cv2.convertScaleAbs(img_laplacian)
#
# a = [img_g,img_laplacian,img_laplacian1,img_s]
#
# result = cv2.hconcat(a)
#
# cv2.imshow("result",result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## canny


# img = Imgread("C:/Users/1/Desktop/images (1)/animal-08.jpg",450)
# img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

def CannyThreshold(val):
   low_threshold = val
   img_blur = cv2.blur(src_gray, (3,3))
   detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
   mask = detected_edges != 0
   dst = src * (mask[:,:,None].astype(src.dtype))
   cv2.imshow(window_name, dst)

parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
print(parser)
parser.add_argument('--input', help='Path to input image.', default="C:/Users/1/Desktop/images (1)/hoya.jpg")
print(parser)
args = parser.parse_args()
src = cv2.imread(cv2.samples.findFile(args.input))

if src is None:
   print('Could not open or find the image: ', args.input)
   exit(0)

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)

CannyThreshold(0)

cv2.waitKey(0)
cv2.destroyAllWindows()


