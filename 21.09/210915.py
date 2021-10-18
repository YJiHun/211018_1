# import cv2 as cv
# import numpy as np
#
# cap = cv.VideoCapture(0)
# while(1):
#     # Take each frame
#     frame = cap.read()
#     # Convert BGR to HSV
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     # define range of blue color in HSV
#     lower_blue = np.array([110,50,50])
#     upper_blue = np.array([130,255,255])
#     # Threshold the HSV image to get only blue colors
#     mask = cv.inRange(hsv, lower_blue, upper_blue)
#     # Bitwise-AND mask and original image
#     res = cv.bitwise_and(frame,frame, mask= mask)
#     cv.imshow('frame',frame)
#     cv.imshow('mask',mask)
#     cv.imshow('res',res)
#     k = cv.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv.destroyAllWindows()

##################

# import cv2 as cv
# import numpy as np
#
# green = np.uint8([[[0,255,0 ]]])
# hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
# print( hsv_green )
# [[[ 60 255 255]]]

##################

# import numpy as np
# import cv2
#
# B = 0
# G = 255
# R = 0
#
# my_color = np.uint8([[[B,G,R]]]) # BGR
# my_hsv = cv2.cvtColor(my_color,cv2.COLOR_BGR2HSV)
#
# print(f"H:{my_hsv[0][0][0]} S:{my_hsv[0][0][1]} V:{my_hsv[0][0][2]}")

##################
import copy,cv2
import numpy as n


def standard(src, size):  ## src = 불러온 이미지 소스 , size = 변경할 크기
    max = 0
    if src.shape[0] > src.shape[1]:
        max = src.shape[0]
    elif src.shape[0] < src.shape[1]:
        max = src.shape[1]
    elif src.shape[0] == src.shape[1]:
        max = src.shape[1]
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

# src_1 = Imgread("C:/Users/1/Desktop/images (1)/kkamong2.jpg", size=500)
# src = Imgread("C:/Users/1/Desktop/images (1)/tomato.jpg",1000,cv2.IMREAD_COLOR)
#
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
#
# h, s, v = cv2.split(hsv)
#
# s_min = 50; s_max = 255
# v_min = 50; v_max = 255
#
# lower_red1 = (0,s_min, v_min)
# upper_red1 = (7,s_max, v_max)
#
# lower_red2 = (165,s_min, v_min)
# upper_red2 = (180,s_max, v_max)
#
# # 116 193
# # mask_h = cv2.inRange(h, 130, 155) # 초록 정보 추출
# # dst = cv2.bitwise_and(src, src, mask=mask_h)
#
# img_mask1 = cv2.inRange(hsv,lower_red1,upper_red1)
# img_mask2 = cv2.inRange(hsv,lower_red2,upper_red2)
# img_mask = cv2.addWeighted(img_mask1, 1.0, img_mask2, 1.0, 0.0)
#
# dst1 = cv2.bitwise_and(src, src, mask=img_mask)

# cv2.imshow("result", src)
# cv2.imshow("result1", dst1)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

############################# 색상(H) , 채도(S) , 명도(V)

# capture = cv2.VideoCapture("C:/Users/1/Desktop/images (1)/color.mp4")
#
# if capture.isOpened() == False:
#     print("동영상을 열수 없습니다.")
#     exit(1)
#
# while True:
#     ret, img = capture.read()
#     img = imgsize(img,500)
#
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     s_min = 50; s_max = 255
#     v_min = 50; v_max = 255
#
#     lower1 = (99, s_min, v_min)
#     upper1 = (109, s_max, v_max)
#
#     lower2 = (0, s_min, v_min)
#     upper2 = (10, s_max, v_max)
#
#     # lower2 = (0, 190, 120)
#     # upper2 = (10, 240, 255)
#
#     img_mask1 = cv2.inRange(hsv, lower1, upper1)
#     img_mask2 = cv2.inRange(hsv, lower2, upper2)
#
#     img_mask = cv2.addWeighted(img_mask1, 1.0, img_mask2, 1.0, 0.0)
#
#     dst1 = cv2.bitwise_and(img, img, mask=img_mask2)
#
#     result = cv2.hconcat([img,dst1])
#     cv2.imshow("result",result)
#
#     key = cv2.waitKey(25)
#
#     if key == 27:
#         break
#
# cv2.destroyAllWindows()

#############################

# img = Imgread("C:/Users/1/Desktop/images (1)/pawns.jpg",500)
#
# img_temp = copy.deepcopy(img)
#
# print(img_temp.shape)
#
#
# temp = img[95:323,17:122,:]
# temp1 = img[95:310,165:250,:]
# temp2 = img[173:310,300:378,:]
#
# temp = imgcolor(temp,cv2.COLOR_BGR2GRAY)
# temp = imgcolor(temp,cv2.COLOR_GRAY2BGR)
#
# temp1 = imgcolor(temp1,cv2.COLOR_BGR2GRAY)
# temp1 = imgcolor(temp1,cv2.COLOR_GRAY2BGR)
#
# temp2 = imgcolor(temp2,cv2.COLOR_BGR2GRAY)
# temp2 = imgcolor(temp2,cv2.COLOR_GRAY2BGR)
#
# m = n.array([[1,0,0],[0,1,0]],dtype=float)
# temp_1 = cv2.warpAffine(temp,m,(500,356))
# temp1_1 = cv2.warpAffine(temp1,m,(500,356))
# temp2_1 = cv2.warpAffine(temp2,m,(500,356))
#
# img_temp[95:323,17:122,:] = temp
# img_temp[95:310,165:250,:] = temp1
# img_temp[173:310,300:378,:] = temp2
#
# result = cv2.hconcat([img_temp,temp_1,temp1_1,temp2_1])
#
# cv2.imshow("result",img_temp)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

########################

# img =Imgread("C:/Users/1/Desktop/images (1)/YJ.jpg",500)
#
# temp = cv2.bitwise_not(img)
#
# a = cv2.hconcat([img,temp])
#
# cv2.imshow("result",a)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

########################

video = cv2.VideoCapture("C:/Users/1/Desktop/images (1)/color.mp4")

if video.isOpened() == False:
    print("동영상을 열수 없습니다.")
    exit(1)

while True:
    ret, vid = video.read()

    vid1 = copy.deepcopy(vid)
    vid1 = cv2.flip(vid1,1)

    temp = vid[:,:int(vid1.shape[1]/2),:] ## 정방향 앞부분 자르기
    temp = cv2.bitwise_not(temp) ## 자른걸 색 반전

    temp1 = vid[:, int(vid1.shape[1] / 2):vid1.shape[1], :] ## 역방향 앞부분 자르기
    temp1 = cv2.bitwise_not(temp1) ## 자른걸 색 반전

    vid1[:, int(vid.shape[1] / 2):vid.shape[1], :] = temp
    vid[:, :int(vid.shape[1] / 2), :] = temp1

    result = cv2.hconcat([vid1,vid])

    cv2.imshow("result",result)

    key = cv2.waitKey(25)

    if  key == 27:
        break

    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.destroyAllWindows()