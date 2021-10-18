import copy

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('C:/Users/1/Desktop/images (1)/kkamong2.jpg',cv2.IMREAD_COLOR)
# img = cv2.imread('C:/Users/1/Desktop/images (1)/hoya.jpg',cv2.IMREAD_COLOR)
img_1 = cv2.imread('C:/Users/1/Desktop/images (1)/animal-03.jpg',cv2.IMREAD_COLOR)
img_2 = cv2.imread('C:/Users/1/Desktop/images (1)/hoya.jpg',cv2.IMREAD_COLOR)
h,w = img.shape[:2] ## 원본 이미지 사진크기
h_1,w_1 = img_1.shape[:2] ## 원본 이미지 사진크기
h_2,w_2 = img_2.shape[:2] ## 원본 이미지 사진크기

scale = 0.25

size = cv2.resize(img,(int(w*scale),int(h*scale)))
h1,w1 = size.shape[:2] ## 원본 이미지 scale(1:4) 사진크기

size1 = cv2.resize(img_2,(int(w_2*0.25),int(h_2*0.25)))
h2,w2 = size1.shape[:2] ## 원본 이미지 scale(1:4) 사진크기

## 1 // img Orignal = h , w / img Orignal / 4 = h1 , w1
rotate = []
dst = []
center = (w1/2,h1/2)
for i in range(8):
    rotate.append(cv2.getRotationMatrix2D(center,i*45,1))
    dst.append(cv2.warpAffine(size,rotate[i],(w1,h1)))

result1_1 = cv2.hconcat([dst[0],dst[1],dst[2],dst[3],dst[4],dst[5],dst[6],dst[7]]) ## 1-1

result1_2_1 = cv2.hconcat([dst[0],dst[1],dst[2],dst[3]]) ## 1-2
result1_2_2 = cv2.hconcat([dst[4],dst[5],dst[6],dst[7]]) ## 1-2
result1_2 = cv2.vconcat([result1_2_1,result1_2_2]) ## 1-2
## 1

## 2 // img Orignal = h , w / img Orignal / 4 = h1 , w1
angle = 45

center1 = (w/2,h/2)
M2 = np.array([[1,0,w1+int(w1/2)],[0,1,h1+int(h1/2)]],dtype=float)

ro2_1 = cv2.getRotationMatrix2D(center1,angle,1)
result2_1 = cv2.warpAffine(size,M2,(w,h)) ## 2-1
result2_2 = cv2.warpAffine(result2_1,ro2_1,(w,h)) ## 2-2
## 2

## 3 // img Orignal = h , w / img Orignal / 4 = h1 , w1
center3_1 = (w1,h1)
M3_1 = np.array([[1,0,w1/2],[0,1,h1/2]],dtype=float)

ro3_1 = cv2.getRotationMatrix2D(center3_1,angle,1)
result3_1 = cv2.warpAffine(size,M3_1,(int(w/2),int(h/2)))  ## 3-1
result3_2 = cv2.warpAffine(result3_1,ro3_1,(int(w/2),int(h/2)))  ## 3-2
## 3

## 4 // img Orignal = h , w / img Orignal / 4 = h1 , w1
M4_1 = np.array([[1,0,0],[0,1,0]],dtype=float)

size4_1 = cv2.resize(size,(int(w1*0.7),int(h1*0.7)))
size4_2 = cv2.resize(size,(int(w1*0.5),int(h1*0.5)))

temp4_1 = cv2.warpAffine(size,M4_1,(w1,h1))
temp4_2 = cv2.warpAffine(size4_1,M4_1,(int(w1*0.7),h1))
temp4_3 = cv2.warpAffine(size4_2,M4_1,(int(w1*0.5),h1))

result4_1 = cv2.hconcat([temp4_1,temp4_2,temp4_3])
## 4

## 5 // img Orignal = h , w / img Orignal / 4 = h1 , w1
scale5 = 0.7
angle5 = 45
size5_1 = cv2.resize(size,(int(w1*scale5),int(h1*scale5)))
h5_1,w5_1 = size5_1.shape[:2]
center5_1 = (w1/2,h1/2)

M5_1 = np.array([[1,0,0],[0,1,0]],dtype=float)
M5_2 = np.array([[1,0,(w1-w5_1)/2],[0,1,(h1-h5_1)/2]],dtype=float)

ro5_1 = cv2.getRotationMatrix2D(center5_1,angle5,1)

temp5_1 = cv2.warpAffine(size,M5_1,(w1,h1))
temp5_2 = cv2.warpAffine(size5_1,M5_1,(w1,h1))
temp5_3 = cv2.warpAffine(size5_1,M5_2,(w1,h1))
temp5_4 = cv2.warpAffine(temp5_3,ro5_1,(w1,h1))

result5_1_1 = cv2.hconcat([temp5_1,temp5_2])
# result5_1_2 = cv2.hconcat([temp5_3,temp5_4])
result5_1_2 = cv2.hconcat([temp5_3,temp5_4])
result5_1 = cv2.vconcat([result5_1_1,result5_1_2])
## 5

## 6 // img_1 Orignal = h_1 , w_1
scale6 = 0.7
size6_1 = cv2.resize(img_1,(int(w_1*scale6),int(h_1*scale6)))
h6_1,w6_1=size6_1.shape[:2]
center6_1 = (w_1/2,h_1/2)

M6_1 = np.array([[1,0,0],[0,1,0]],dtype=float)
M6_2 = np.array([[1,0,(w_1-w6_1)/2],[0,1,(h_1-h6_1)/2]],dtype=float)

ro6_1 = cv2.getRotationMatrix2D(center6_1,45,1)

temp6_1 = cv2.warpAffine(img_1,M6_1,(w_1,h_1))
temp6_2 = cv2.warpAffine(size6_1,M6_1,(w_1,h_1))
temp6_3 = cv2.warpAffine(size6_1,M6_2,(w_1,h_1))
temp6_4 = cv2.warpAffine(temp6_3,ro6_1,(w_1,h_1))

result6_1_1 = cv2.hconcat([temp6_1,temp6_2])
result6_1_2 = cv2.hconcat([temp6_3,temp6_4])
result6_1 = cv2.vconcat([result6_1_1,result6_1_2])
## 6

## 7 // img_2 Orignal = h_2 , w_2 / img_2 Orignal / 4 = h2 , w2
scale7 = 0.7
size7_1 = cv2.resize(size1,(int(w2*scale7),int(h2*scale7)))
h7_1,w7_1=size7_1.shape[:2]
center7_1 = (w2/2,h2/2)

M7_1 = np.array([[1,0,0],[0,1,0]],dtype=float)
M7_2 = np.array([[1,0,(w2-w7_1)/2],[0,1,(h2-h7_1)/2]],dtype=float)

ro7_1 = cv2.getRotationMatrix2D(center7_1,45,1)

temp7_1 = cv2.warpAffine(size1,M7_1,(w2,h2))
temp7_2 = cv2.warpAffine(size7_1,M7_1,(w2,h2))
temp7_3 = cv2.warpAffine(size7_1,M7_2,(w2,h2))
temp7_4 = cv2.warpAffine(temp7_3,ro7_1,(w2,h2))

result7_1_1 = cv2.hconcat([temp7_1,temp7_2])
result7_1_2 = cv2.hconcat([temp7_3,temp7_4])
result7_1 = cv2.vconcat([result7_1_1,result7_1_2])
## 7

## 8 // img_1 Orignal = h_1 , w_1 // 강사님 코드
scale8 = 0.7
background = np.zeros_like(img_1) # 원본이미지 크기과 같은 백 그라운드 생성

size8_1 = cv2.resize(img_1,None,fx=scale8,fy=scale8)
h8_1 , w8_1 = size8_1.shape[:2]

center8 = [w_1/2,h_1/2]#중심값은 원본에서 변하지않음

background[:h8_1,:w8_1,:] = size8_1 # 원본이미지 크기과 같은 백 그라운드에 스케일을 줄인 사진을 삽입

M = np.array([[1,0,(w_1-w8_1)/2],[0,1,(h_1-h8_1)/2]],dtype=float)

temp8_1 = cv2.warpAffine(size8_1,M,(w_1,h_1))
ro8 = cv2.getRotationMatrix2D(center8,45,1)
temp8_2 = cv2.warpAffine(temp8_1,ro8,(w_1,h_1))

result8_1_1 = cv2.hconcat([img_1,background])
result8_1_2 = cv2.hconcat([temp8_1,temp8_2])
result8_1 = cv2.vconcat([result8_1_1,result8_1_2])
## 8 // 강사님 코드

## 9 // img_1 Orignal = h_1 , w_1
scale9 = 0.7
background9_1 = np.array(img_1) # 원본이미지 크기과 같은 백 그라운드 생성
background9_2 = img_1.copy() # 원본이미지 크기과 같은 백 그라운드 생성
# background9_2 = copy.deepcopy(img_1) # 원본이미지 크기과 같은 백 그라운드 생성
background9_3 = copy.deepcopy(img_1) # 원본이미지 크기과 같은 백 그라운드 생성


h9,w9 = background9_1.shape[:2]

size9_1 = cv2.resize(img_1,None,fx=scale9,fy=scale9)
h9_1 , w9_1 = size9_1.shape[:2]

center9 = [w_1/2,h_1/2]#중심값은 원본에서 변하지않음

background9_1[:h9_1,:w9_1,:] = size9_1 # 원본이미지 크기과 같은 백 그라운드에 스케일을 줄인 사진을 삽입

M = np.array([[1,0,(w_1-w9_1)/2],[0,1,(h_1-h9_1)/2]],dtype=float)

temp9_1 = cv2.warpAffine(size9_1,M,(w9,h9))

# for i in range(temp9_1.shape[2]):
#     for ii in range(temp9_1.shape[1]):
#         for iii in range(temp9_1.shape[0]):
#             if temp9_1[iii][ii][i] != 0:
#                 background9_2[iii][ii][i] = temp9_1[iii][ii][i]

ro9 = cv2.getRotationMatrix2D(center9,45,1)
temp9_2 = cv2.warpAffine(temp8_1,ro9,(w9,h9))

# for i in range(temp9_2.shape[2]):
#     for ii in range(temp9_2.shape[1]):
#         for iii in range(temp9_2.shape[0]):
#             if temp9_2[iii][ii][i] != 0:
#                 background9_3[iii][ii][i] = temp9_2[iii][ii][i]

result9_1_1 = cv2.hconcat([img_1,background9_1])
# result9_1_2 = cv2.hconcat([temp9_1,temp9_2])
result9_1_2 = cv2.hconcat([background9_2,background9_3])
result9_1 = cv2.vconcat([result9_1_1,result9_1_2])
## 9

# cv2.imshow("JIHUN1-1",result1_1) ## 1-1
# cv2.imshow("JIHUN1_2",result1_2) ## 1-2
# cv2.imshow("JIHUN2-1",result2_1) ## 2-1
# cv2.imshow("JIHUN2-2",result2_2) ## 2-2
# cv2.imshow("JIHUN3-1",result3_1) ## 3-1
# cv2.imshow("JIHUN3-2",result3_2) ## 3-2
# cv2.imshow("JIHUN4",result4_1) ## 4
# cv2.imshow("JIHUN5",result5_1) ## 5
# cv2.imshow("JIHUN6",result6_1) ## 6
# cv2.imshow("JIHUN7",result7_1) ## 7
# cv2.imshow("JIHUN8",result8_1) ## 8 // 강사님 코드
# cv2.imshow("JIHUN9",result9_1) ## 9



cv2.waitKey(0)
cv2.destroyAllWindows()
