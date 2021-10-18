import cv2
import numpy as np

def Url(link):
    n = np.fromfile(link, dtype=np.uint8)
    return n

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

def Imgread(link, size, Colortype=cv2.IMREAD_COLOR):  ## url = 이미지 링크 , size = 변경할 크기 , 받아올 이미지 컬러타입 , Colortype = 변경할 이미지 컬러(미지정시 자동 BGR컬로)
    url = Url(link)
    src = cv2.imdecode(url, Colortype)
    scale, result = imgsize(src, size)
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
# scale, img = Imgread('C:/Users/user/Desktop/5G․AI자율주행인력양성/파이썬/21.09/PCB1/21번 L50_OK.bmp',5000000, cv2.IMREAD_COLOR)
scale, img = Imgread('PCB1/18번 L50_NG.bmp',5000000, cv2.IMREAD_COLOR)
# scale, img = Imgread('C:/Users/1/Desktop/PCB1/2.L50_OK.bmp',500, cv2.IMREAD_COLOR)
kn = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))

img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) ## 그레이 변환

img_b = cv2.bitwise_not(img_g) ## 이미지 이진화

ret ,img_b = cv2.threshold(img_b, 150, 255, cv2.THRESH_BINARY_INV) ## 이미지 마스킹 처리

img_e = cv2.morphologyEx(img_b,cv2.MORPH_CLOSE,kn,iterations=2)

contours, hierarchy = cv2.findContours(img_e,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) ## 이미지 외곽선 검출

my_color = (0,255,0)
text_color = (0,255,0)
thickness = 2
AA = 0

lists = []
clists = []

bottom = []
right = []
left = []
center = []

move = 330

bottomrange_upper = (380, 1000)
bottomrange_lower = (2170, 1500)
rightrange_upper = (50, 600)
rightrange_lower = (300, 1500)
leftrange_upper = (2250, 600)
leftrange_lower = (2500, 1500)
centerrange_upper = (1150, 600)
centerrange_lower = (1400, 900)

bottomok = (15400, 13900)
rightok = (17200, 14800)
leftok = (17200, 14800)
centerok = (13800, 13000)
carea = ()

NGcount = 0

for i, contour in enumerate(contours):
    temp = []
    area = cv2.contourArea(contour)

    mu = cv2.moments(contour)
    cx = int(mu['m10'] / (mu['m00'] + 1e-5))
    cy = int(mu['m01'] / (mu['m00'] + 1e-5))

    x, y, w, h = cv2.boundingRect(contour)

    temp.append(area)
    temp.append((cx,cy))
    lists.append(temp)

    if area >= 10000:
        temp = []

        temp.append((x, y))
        temp.append((x + w, y + h))
        clists.append(temp)

        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        # cv2.drawContours(img, contours, i, (0, 255, 0), thickness)

# print(lists[0][2],"end")
# print(lists[0][1])
# print(lists[0][1][0])
# print(lists[0][1][1])

for i in range(len(lists)):
    if lists[i][0] >= 1:
        if lists[i][1][0] >= bottomrange_upper[0] and lists[i][1][1] >= bottomrange_upper[1] and lists[i][1][0] <= bottomrange_lower[0] and lists[i][1][1] <= bottomrange_lower[1] and lists[i][0] <= bottomok[0] and lists[i][0] > bottomok[1]: ## OK
            # print("bottom")
            temp = []
            temp.insert(0, int(lists[i][0]))
            temp.insert(1, (lists[i][1][0], lists[i][1][1]))

            cv2.drawContours(img, contours, i, (0, 255, 0), thickness)
            temp.insert(2, "OK")

            bottom.append(temp)

        if lists[i][1][0] >= rightrange_upper[0] and lists[i][1][1] >= rightrange_upper[1] and lists[i][1][0] <= rightrange_lower[0] and lists[i][1][1] <= rightrange_lower[1] and lists[i][0] <= rightok[0] and lists[i][0] > rightok[1]:
            # print("right")
            temp = []
            temp.insert(0, int(lists[i][0]))
            temp.insert(1, (lists[i][1][0], lists[i][1][1]))

            # for ii in range(len(lists)): ## 미완성
            #     if lists[ii][0][0] >= rightrange_upper[0] and clists[ii][0][1] >= rightrange_upper[1] and clists[ii][1][0] <= rightrange_lower[0] and clists[ii][1][1] <= rightrange_lower[1]:
            #         cv2.drawContours(img, contours, i, (0, 255, 0), thickness)
            #         # cv2.rectangle(img, clists[ii][0], clists[ii][1], (255, 0, 255), 1)
            cv2.drawContours(img, contours, i, (0, 255, 0), thickness)
            temp.insert(2, "OK")

            right.append(temp)

        if lists[i][1][0] >= leftrange_upper[0] and lists[i][1][1] >= leftrange_upper[1] and lists[i][1][0] <= leftrange_lower[0] and lists[i][1][1] <= leftrange_lower[1] and lists[i][0] <= leftok[0] and lists[i][0] > leftok[1]:
            # print("left")
            temp = []
            temp.insert(0, int(lists[i][0]))
            temp.insert(1, (lists[i][1][0], lists[i][1][1]))

            # for ii in range(len(clists)): ## 미완성
            #     if clists[ii][0][0] >= leftrange_upper[0] and clists[ii][0][1] >= leftrange_upper[1] and clists[ii][1][0] <= leftrange_lower[0] and clists[ii][1][1] <= leftrange_lower[1]:
            #         cv2.drawContours(img, contours, i, (0, 255, 0), thickness)
            #         cv2.rectangle(img, clists[ii][0], clists[ii][1], (255,0,255), 1)
            #         pass
            # clists[i][0] =
            cv2.drawContours(img, contours, i, (0, 255, 0), thickness)
            temp.insert(2, "OK")

            left.append(temp)

        if lists[i][1][0] >= centerrange_upper[0] and lists[i][1][1] >= centerrange_upper[1] and lists[i][1][0] <= centerrange_lower[0] and lists[i][1][1] <= centerrange_lower[1] and lists[i][0] <= centerok[0] and lists[i][0] > centerok[1]: ## OK
            # print("left")
            temp = []
            temp.insert(0, int(lists[i][0]))
            temp.insert(1, (lists[i][1][0], lists[i][1][1]))

            cv2.drawContours(img, contours, i, (0, 255, 0), thickness)
            temp.insert(2, "OK")

            center.append(temp)

        cv2.rectangle(img, (bottomrange_upper[0], bottomrange_upper[1]), (bottomrange_lower[0], bottomrange_lower[1]), my_color, thickness=2) ## bottom
        cv2.rectangle(img, (rightrange_upper[0], rightrange_upper[1]), (rightrange_lower[0], rightrange_lower[1]), my_color, thickness=2) ## right
        cv2.rectangle(img, (leftrange_upper[0], leftrange_upper[1]), (leftrange_lower[0], leftrange_lower[1]), my_color, thickness=2) ## left
        cv2.rectangle(img, (centerrange_upper[0], centerrange_upper[1]), (centerrange_lower[0],centerrange_lower[1]), my_color, thickness=2) ## center

    elif lists[i][0] < 10000 and lists[i][0] > 1 and not(lists[i][1][0] >= centerrange_upper[0] and lists[i][1][1] >= centerrange_upper[1] and lists[i][1][0] <= centerrange_lower[0] and lists[i][1][1] <=  centerrange_lower[1]):
        cv2.drawContours(img, contours, i, (0, 0, 255), thickness)
        # print(lists[i][1][0],lists[i][1][1])
        NGcount += 1

# print(len(bottom))
# print(bottom)
# print(len(right))
# print(right)
# print(len(left))
# print(left)
# print(len(center))
# print(center)

for i in range(len(bottom)):
    cv2.putText(img,f'{i + 1}.bottom ({bottom[i][2]})',(bottom[i][1][0] - 125, bottom[i][1][1] + 90),cv2.FONT_HERSHEY_COMPLEX, 1,
                text_color, 1)
    cv2.putText(img,f'({bottom[i][0]})',(bottom[i][1][0] - 65, bottom[i][1][1] + 125),cv2.FONT_HERSHEY_COMPLEX, 1,
                text_color, 1)
    if len(right) > i:
        cv2.putText(img, f'{i + 1}.right ({right[i][2]})', (right[i][1][0] + 110, right[i][1][1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                    text_color, 1)
        cv2.putText(img, f'({right[i][0]})', (right[i][1][0] + 110, right[i][1][1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                    1, text_color, 1)
    if len(left) > i: # -180
        cv2.putText(img, f'{i + 1}.left ({left[i][2]})', (left[i][1][0] - 270, left[i][1][1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                    text_color, 1)
        cv2.putText(img, f'({left[i][0]})', (left[i][1][0] - 220, left[i][1][1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                    1, text_color, 1)
    if len(center) > i:
        cv2.putText(img, f'{i + 1}.center ({center[i][2]})', (center[i][1][0] - 125, center[i][1][1] + 90), cv2.FONT_HERSHEY_COMPLEX, 1,
                    text_color, 1) # 왼쪽 : (x - 255, y)
        cv2.putText(img, f'({center[i][0]})', (center[i][1][0] - 65, center[i][1][1] + 125), cv2.FONT_HERSHEY_COMPLEX,
                    1, text_color, 1) # 왼쪽 : (x - 195, y + 35)



# print("NG",NGcount)
img = cv2.pyrDown(img)
cv2.imshow('dst',img)

img_e = cv2.pyrDown(img_e)
cv2.imshow('result',img_e)

cv2.waitKey(0)
cv2.destroyAllWindows()