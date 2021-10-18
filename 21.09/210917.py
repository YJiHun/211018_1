import cv2
import numpy as np


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

    print("원래이미지 h, w",src.shape[0],src.shape[1])

    scale = size / max
    return scale

def imgsize(src, size): ## src = 불러온 이미지 소스 , size = 변경할 크기
    scale = standard(src, size)
    return scale,cv2.resize(src, None, fx=scale, fy=scale)

def imgcolor(src,Colortype=cv2.IMREAD_COLOR): ## src = 불러온 이미지 소스 , Colortype = 변경할 이미지 컬러(미지정시 자동 BGR컬로)
    return cv2.cvtColor(src,Colortype)

def Imgread(url, size, Colortype=cv2.IMREAD_COLOR):  ## url = 이미지 링크 , size = 변경할 크기 , 받아올 이미지 컬러타입 , Colortype = 변경할 이미지 컬러(미지정시 자동 BGR컬로)
    src = cv2.imread(url, Colortype)
    scale,result = imgsize(src, size)
    print("이미지 재지정 h, w", result.shape[0], result.shape[1])
    return scale,result

# img_src = cv2.imread('C:/Users/1/Desktop/images (1)/back.jpg', cv2.IMREAD_COLOR)
#
# #1 - 이미지를 Gray로 변환
# img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
#
# #2 이진화를 진행
# #  이미지의 특성을 파악 : 검출할려고 하는것(도형)이 흰색으로 나와야함
# #  배경이 흰색 : 검출해야하는 물체보다 배경이 밝은 상태
# # 방법 2-1 : 그레이 이미지를 반전하고 Threshold 를 적용
# # img_gray = cv2.bitwise_not(img_gray)
# # ret, img_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
# # 방법 2-2 : cv2.threshold()함수의 옵션사용
# ret, img_binary = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
#
# # 검출하려고 하는 도형의 외곽선 검출 : findContours()함수 사용
# contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST,
#                                       cv2.CHAIN_APPROX_NONE)
#
# my_color = (0,255,0) # (B,G,R)
# text_color = (255,0,0) # (B,G,R)
# thickness = 2
# for i, contour in enumerate(contours):
#    area = cv2.contourArea(contour)
#
#    # contourArea()함수로 객체의 면적을 구하고 면적기준으로 임계값보다
#    # 큰 객체만 외곽선을 그리고 면적정보를 표시한다.
#    if area > 10000:
#        cv2.drawContours(img_src, contours, i, my_color, thickness)
#
#        #모멘트 그리기(무게중심)
#        mu = cv2.moments(contour)
#        cx = int(mu['m10'] / (mu['m00']+1e-5))
#        cy = int(mu['m01'] / (mu['m00']+1e-5))
#        cv2.circle(img_src, (cx,cy), 5, (0,255,255),-1)
#        cv2.putText(img_src, f'{i}: {int(area)}', (cx-50,cy-20),
#                    cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 1)
#
#        # 객체의 외곽에 사각형 그리기
#        # 방법 1: boundingRect(회전 고려 않함) : Cyan
#        x,y,w,h = cv2.boundingRect(contour)
#        cv2.rectangle(img_src, (x,y), (x+w,y+h),(255,255,0), 1)
#
#        # 방법 2: minAreaRect() 사용 : Magenta
#        # 물체의 회전을 고려해서 경계 사각형 그림
#        rect = cv2.minAreaRect(contour)
#        box = cv2.boxPoints(rect)
#        box = np.int0(box)
#        cv2.drawContours(img_src, [box], 0, (255,0,255), 1)
#        cv2.imshow('dst',img_src)
#        cv2.waitKey(0)
#
# cv2.destroyAllWindows()


# # dilation 팽창, 확장
# img_result = cv2.dilate(img_gray, kernel, iterations=1)

# # erosion, 침식, 수축
# img_result = cv2.erode(img_gray, kernel, iterations=1)
# # OR
# img_result = cv2.erode(img_gray,anchor=(-1,-1), kernel, iterations=1)

#########

# img = Imgread('C:/Users/1/Desktop/images (1)/back.jpg',500, cv2.IMREAD_COLOR)
#
# img_g = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# img_g1 = cv2.bitwise_not(img_g)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
#
# img_erode = cv2.erode(img_g1,kernel,iterations=3)
#
# temp = cv2.hconcat([img_g,img_g1,img_erode])
#
# cv2.imshow("result",temp)
# cv2.imshow("result1",img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#########

# img = Imgread('C:/Users/1/Desktop/images (1)/gram.jpg',500, cv2.IMREAD_COLOR)
#
# img_g = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# img_g1 = cv2.bitwise_not(img_g)
#
# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
#
# img_d = cv2.erode(img_g1,kernel,iterations=3)
# img_e = cv2.dilate(img_d,kernel,iterations=3)
# img_e1 = cv2.dilate(img_e,kernel,iterations=3)
# img_d1 = cv2.erode(img_e1,kernel,iterations=3)
#
# temp = cv2.hconcat([img_g1,img_d1])
# # temp1 = cv2.hconcat([img_g,img_g1,img_erode])
#
#
# # cv2.imshow("result",temp)
# # cv2.imshow("result1",temp1)
# cv2.imshow("result2",temp)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#########

# img = Imgread('C:/Users/1/Desktop/images (1)/back1.png',500, cv2.IMREAD_COLOR)
#
# img_g = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# img_g1 = cv2.bitwise_not(img_g)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
#
#
#
# img_m = cv2.morphologyEx(img_g1,cv2.MORPH_CLOSE,kernel,iterations=5) ## morphologyEx = dilate + erode 순서로 한거와같다
# img_e = cv2.dilate(img_g1,kernel,iterations=5)
# img_d = cv2.erode(img_e,kernel,iterations=5)
#
# img_m1 = cv2.morphologyEx(img_g1,cv2.MORPH_OPEN,kernel,iterations=5) ## morphologyEx = erode + dilate 순서로 한거와같다
# img_d1 = cv2.erode(img_g1,kernel,iterations=5)
# img_e1 = cv2.dilate(img_d1,kernel,iterations=5)
#
# temp = cv2.hconcat([img_m,img_d,img_m1,img_e1])
#
# cv2.imshow("result1",temp)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#########

# scale,img = Imgread('C:/Users/1/Desktop/images (1)/gram.jpg',50000, cv2.IMREAD_COLOR)
#
# img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img_g.shape)
#
# ret ,img_b = cv2.threshold(img_g, 150, 255, cv2.THRESH_BINARY_INV)
#
# contours, hierarchy = cv2.findContours(img_b,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#
# my_color = (0,255,0)
# text_color = (255,0,0)
# thickness = 2
#
# for i, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#
#     if area > 10000:
#         cv2.drawContours(img,contours,i,my_color,thickness)
#
#         mu = cv2.moments(contour)
#         cx = int(mu['m10'] / (mu['m00'] + 1e-5))
#         cy = int(mu['m01'] / (mu['m00'] + 1e-5))
#
#         cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
#         cv2.putText(img, f'{i}: {int(area)}', (cx - 50, cy - 20),
#                     cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 1)
#
#         x,y,w,h = cv2.boundingRect(contour)
#         cv2.rectangle(img, (x,y), (x+w,y+h),(255,255,0), 1)
#
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#
#         cv2.drawContours(img, [box], 0, (255,0,255), 1)
#
#
#         cv2.imshow('dst',img)
#
# print(img.shape)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#########

# scale, img = Imgread('C:/Users/1/Desktop/images (1)/circle.jpg',500, cv2.IMREAD_COLOR)
#
# img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# circles = cv2.HoughCircles(img_g,cv2.HOUGH_GRADIENT,1,minDist=int(100*scale),param1=250,param2=10,minRadius=int(80*scale),maxRadius=int(120*scale))
#
# for i in enumerate(circles[0]):
#     cv2.circle(img,(int(i[1][0]),int(i[1][1])),int(i[1][2]),(255,255,255),thickness=3)
#
# result = cv2.hconcat([])
#
# cv2.imshow('result',img)
# # cv2.imshow('result1',img_g)
# # cv2.imshow('result2',result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#########

scale, img = Imgread('C:/Users/1/Desktop/PCB1/6.L50.bmp',5000000, cv2.IMREAD_COLOR)
# scale, img = Imgread('C:/Users/1/Desktop/PCB1/2.L50_OK.bmp',500, cv2.IMREAD_COLOR)
kn = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))

img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) ## 그레이 변환

img_b = cv2.bitwise_not(img_g) ## 이미지 이진화

ret ,img_b = cv2.threshold(img_b, 150, 255, cv2.THRESH_BINARY_INV) ## 이미지 마스킹 처리

img_e = cv2.morphologyEx(img_b,cv2.MORPH_OPEN,kn,iterations=1)
img_e = cv2.erode(img_e,kn,iterations=1)
img_e = cv2.dilate(img_e,kn,iterations=2)
img_e = cv2.erode(img_e,kn,iterations=2)
img_e = cv2.dilate(img_e,kn,iterations=1)
img_e = cv2.morphologyEx(img_e,cv2.MORPH_CLOSE,kn,iterations=2)
img_e = cv2.morphologyEx(img_e,cv2.MORPH_OPEN,kn,iterations=1)
img_e = cv2.dilate(img_e,kn,iterations=2)
img_e = cv2.morphologyEx(img_e,cv2.MORPH_OPEN,kn,iterations=1)
img_e = cv2.erode(img_e,kn,iterations=2)
# img_e = cv2.morphologyEx(img_e,cv2.MORPH_CLOSE,kn,iterations=2)
# img_e = cv2.dilate(img_e,kn,iterations=1)
# img_e = cv2.morphologyEx(img_e,cv2.MORPH_OPEN,kn,iterations=2)
# img_e = cv2.erode(img_e,kn,iterations=2)
# img_e = cv2.dilate(img_e,kn,iterations=1)
# img_e = cv2.morphologyEx(img_e,cv2.MORPH_CLOSE,kn,iterations=2)
# img_e = cv2.erode(img_e,kn,iterations=1)
# img_e = cv2.morphologyEx(img_e,cv2.MORPH_CLOSE,kn,iterations=1)
# img_e = cv2.dilate(img_e,kn,iterations=2)
# img_e = cv2.morphologyEx(img_e,cv2.MORPH_CLOSE,kn,iterations=1)
# img_e = cv2.erode(img_e,kn,iterations=2)
# img_e = cv2.dilate(img_e,kn,iterations=2)
# img_e = cv2.morphologyEx(img_e,cv2.MORPH_OPEN,kn,iterations=2)
# img_e = cv2.erode(img_e,kn,iterations=2)
# img_e = cv2.morphologyEx(img_e,cv2.MORPH_CLOSE,kn,iterations=2)
# img_e = cv2.dilate(img_e,kn,iterations=1)
# img_e = cv2.morphologyEx(img_e,cv2.MORPH_OPEN,kn,iterations=2)


##################

contours, hierarchy = cv2.findContours(img_e,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) ## 이미지 외곽선 검출

my_color = (0,255,0)
text_color = (0,255,0)
thickness = 2

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print(i,contour.shape)

    if area > 5000:
        cv2.drawContours(img,contours,i,(0,0,255),thickness)

        mu = cv2.moments(contour)
        cx = int(mu['m10'] / (mu['m00'] + 1e-5))
        cy = int(mu['m01'] / (mu['m00'] + 1e-5))

        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        x,y,w,h = cv2.boundingRect(contour)
        # cv2.rectangle(img, (x,y), (x+w,y+h),(255,255,0), 1)

        cv2.putText(img, f'{i}: {int(area)}', (x + w - 130 , y + h + 25),
                    cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 1)

        #rect = cv2.minAreaRect(contour)
        #box = cv2.boxPoints(rect)
        #box = np.int0(box)

        #cv2.drawContours(img, [box], 0, (255,0,255), 1)

img = cv2.pyrDown(img)
cv2.imshow('dst',img)

img_e = cv2.pyrDown(img_e)
cv2.imshow('result',img_e)

cv2.waitKey(0)
cv2.destroyAllWindows()