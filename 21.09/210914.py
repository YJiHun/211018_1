import copy

import cv2
import numpy as np


########################## 이미지
# img = cv2.imread("C:/Users/1/Desktop/images (1)/fruits.jpg",cv2.IMREAD_COLOR)
# h,w = img.shape[:2]
#
# img_r,img_g,img_b = cv2.split(img)
#
# img_z1 = np.zeros((h,w,1),dtype=np.uint8)
#
# img_z = np.zeros_like(img_r)
#
# img_R = cv2.merge((img_r,img_z,img_z))
# img_G = cv2.merge((img_z,img_g,img_z))
# img_B = cv2.merge((img_z,img_z,img_b))
#
# # print(img_rgb.shape)
#
# img_GRAY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) ### 3차원 RGB >>>> 1차원 GRAY
# img_GRAY = cv2.cvtColor(img_GRAY,cv2.COLOR_GRAY2BGR) ### 1차원 GRAY >>>> 3차원 GGG
#
# img_temp = img[int(h*0.25):int(h*0.75),int(w*0.25):int(w*0.75),:]
#
# img_GRAY[int(h*0.25):int(h*0.75),int(w*0.25):int(w*0.75),:] = img_temp
#
# result = cv2.hconcat([img_r,img_g,img_b])
#
# # cv2.imshow("result",img_GRAY)
# # cv2.imshow("result1",result )
# cv2.imshow("result",img_R)
# cv2.imshow("result1",img_G)
# cv2.imshow("result2",img_B)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

########################## 동영상

# capture = cv2.VideoCapture("C:/Users/1/Desktop/images (1)/bad.mp4")
#
# if capture.isOpened() == False:
#     print("동영상을 열수 없습니다.")
#     exit(1)
#
# while True:
#     ret, img_frame = capture.read()
#
#     gray = cv2.cvtColor(img_frame,cv2.COLOR_BGR2GRAY) ## 3차원 RGB >>> 1차원 GRAY 변환
#     gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  ## 1차원 GRAY >>> 3차원 GRAY 변환
#
#     r,g,b = cv2.split(img_frame) ## R,G,B 분할
#
#     z = np.zeros_like(r) ## 검정 화면
#
#     video = cv2.merge((b,z,z)) ## R,G,B 조합
#     video1 = cv2.merge((z, g, z)) ## R,G,B 조합
#     video2 = cv2.merge((z, z, r)) ## R,G,B 조합
#
#     size = cv2.resize(img_frame, None, fx=0.5, fy=0.5)
#
#     h,w = img_frame.shape[:2]
#     h1, w1 = size.shape[:2]
#
#     M = np.array([[1,0,int(w*0.25)],[0,1,int(h*0.25)]],dtype=float)
#
#
#     temp = cv2.warpAffine(size,M,(w,h))
#
#     cv2.rectangle(temp,(int(w*0.25),int(h*0.25)),(int(w*0.75),int(h*0.75)),(0,255,0),thickness=3)
#
#     ro = cv2.getRotationMatrix2D((w/2,h/2),45,1)
#
#     temp = cv2.warpAffine(temp,ro,(w,h))
#
#     temp1 = copy.deepcopy(img_frame)
#
#     temp1[:h1,:w1,:] = size
#
#     cv2.rectangle(temp1, (0,0),(int(w * 0.5), int(h * 0.5)), (0, 255, 0), thickness=3)
#
#     VIDEO = cv2.hconcat([img_frame,gray,video])
#     VIDEO1 = cv2.hconcat([img_frame,video, video1, video2])
#     VIDEO2 = cv2.hconcat([img_frame,temp,temp1])
#
#     # cv2.rectangle(VIDEO2, (int(w * 0.25), int(h * 0.15)), (int(w * 0.25) * 7 + w, int(h * 0.85)), (0, 255, 0),
#     #               thickness=3)
#     # cv2.rectangle(VIDEO2, (int(w * 0.25), int(h * 0.15)), (int(w * 0.25) * 7 + w, int(h * 0.85)), (0, 255, 0),
#     #               thickness=3)
#
#     if ret == False: ## 동영상 재생 끝
#         print("동영상 재생 완료")
#         break
#
#     cv2.imshow("vide",VIDEO2)
#
#     key = cv2.waitKey(27)
#
#     if key == 27: ## 27 == ESC키
#         break
#
#     if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
#         capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# capture.release()
# cv2.destroyAllWindows()

########################## 검정색 화면 만들기

# import random
#
# h = 600
# w = 800
#
# img_z = np.zeros((h,w,3),dtype=np.uint8)
#
# center = (400,300)
# radian = 10
# color = (0,255,0)
#
# # cv2.circle(img_z,center,radian,color,thickness=2)
#
# # for i in range(20): ####### 랜덤 원그리기
# #     center = (random.randint(100,700),random.randint(100,500))
# #     radian = random.randint(5,15)
# #     color = (random.randint(0,256),random.randint(0,256),random.randint(0,256))
# #     cv2.circle(img_z,center,radian,color,thickness=2)
#
# start = (50,50)
# end = (750,50)
#
# # cv2.line(img_z,start,end,color,thickness=3)
# #
# # for i,n in enumerate(range(50,600,50)): ####### 랜덤 선그리기
# #     cv2.line(img_z,(50,n),(750,n),
# #              (random.randint(0,256),random.randint(0,256),random.randint(0,256)),
# #              thickness=i+1)
#
# start = (50,50)
# end = (750,550)
#
# # cv2.rectangle(img_z,start,end,color,thickness=3)
# #
# # for i in range(1,11): ####### 랜덤 사각형그리기
# #     cv2.rectangle(img_z,start,(250+(50*i),50+(50*i)),
# #                   (random.randint(0,256),random.randint(0,256),random.randint(0,256)),
# #                   thickness=3)
#
# start = (50,50)
# end = (750,550)
#
# # cv2.ellipse(img_z,(250,400),(100,50),0,0,360,(255,255,0),2)
# # cv2.ellipse(img_z,(650,400),(50,100),0,0,360,(255,255,0),2)
#
# # for i in range(1,9): ####### 랜덤 타원그리기
# #     cv2.ellipse(img_z, (400, 300), (100+(i*30), 50+(i*30)), i * 45, 0, 360, (255, 255, 0), 2)
#
# # cv2.putText(img_z, "Python hi Good",(100,400),cv2.FONT_HERSHEY_SIMPLEX,
# #             2,(random.randint(0,256),random.randint(0,256),random.randint(0,256)),
# #             thickness=2) ####### 글적기(한글안됨)
#
# center = (400,300)
#
# cv2.circle(img_z,center,10,(0,255,0),thickness=-1)
# cv2.circle(img_z,center,100,(0,0,255),thickness=1)
# cv2.putText(img_z,"Draw Circle",(315,430),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),
#             thickness=2)
#
# cv2.line(img_z,(0,0),(800,600),(0,0,255),thickness=3)
# cv2.line(img_z,(800,0),(0,600),(0,255,0),thickness=3)
# cv2.putText(img_z,"Draw Line",(315,590),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),
#             thickness=2)
#
# cv2.ellipse(img_z,center,(200,200),0,0,360,(0,0,255),thickness=3)
# cv2.ellipse(img_z,center,(200,10),0,0,360,(0,255,255),thickness=3)
# cv2.ellipse(img_z,center,(10,200),0,0,360,(0,255,0),thickness=3)
# cv2.putText(img_z,"Draw EllipseCircle",(240,330),cv2.FONT_HERSHEY_SIMPLEX,0.5,(10,200,200),
#             thickness=2)
#
#
#
#
# cv2.imshow("hehe",img_z)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()