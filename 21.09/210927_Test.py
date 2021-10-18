import numpy as np
import cv2

img_src = cv2.imread("videos/lene.png")

lower_yellow = (15, 100, 100)  # 자료형은 튜플형태로(H,S,V)
upper_yellow = (40, 255, 255)  # 자료형은 튜플형태로(H,S,V)

lower_white = (150, 150, 150)  # 자료형은 튜플형태로(B,G,R)
upper_white = (255, 255, 255)  # 자료형은 튜플형태로(B,G,R)

# 노란색, 흰색 차선 추출
# img_g = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

img_hsv = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)

img_bit_w = cv2.inRange(img_src,lower_white,upper_white)
img_bit_y = cv2.inRange(img_hsv,upper_yellow,upper_yellow)

img_bit = cv2.addWeighted(img_bit_w,1.0,img_bit_y,1.0,0)

img_temp = cv2.bitwise_and(img_src,img_src,mask=img_bit)

img_gray = cv2.cvtColor(img_temp,cv2.COLOR_BGR2GRAY)

# 이미지 블러
img_gray = cv2.GaussianBlur(img_gray,(5,5), 5)

# 이진화 수행
_, img_binary = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
img_canny = cv2.Canny(img_binary, 50, 150)

#사다리꼴 적용하기
img_roi = img_canny
rho = 2
theta = 1 * np.pi / 180
threshold = 15
min_line_length = 10
max_line_gap = 20
lines = cv2.HoughLinesP(img_roi, rho, theta, threshold,
                       minLineLength = min_line_length,
                       maxLineGap = max_line_gap)
for i, line in enumerate(lines):
   cv2.line(img_src, (line[0][0], line[0][1]),
            (line[0][2], line[0][3]), (0, 255, 0), 2)

cv2.imshow("dst", img_temp)

cv2.waitKey(0)
cv2.destroyAllWindows()
