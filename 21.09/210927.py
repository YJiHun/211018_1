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

    # print("원래이미지 h, w",src.shape[0],src.shape[1])

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

def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    # 예외처리
    trap_height = 0.4
    if lines is None:
        return
    if len(lines) == 0:
        return
    draw_right = True
    draw_left = True
    # 모든 선의 기울기 찾기
    # 기울기의 절대값이 임계값 보다 커야 추출됨
    slope_threshold = 0.5
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
        # 기울기 계산
        if x2 - x1 == 0.:  # 기울기의 분모가 0일때 예외처리
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)
        # 조건을 만족하는 line을 new_lines에 추가
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
    lines = new_lines
    # 라인을 오른쪽과 왼쪽으로 구분하기
    # 기울기 및 점의 위치가 영상의 가운데를 기준으로 왼쪽 또는 오른쪽에 위치
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2  # 영상의 가운데 x 좌표
        #기울기 방향이 바뀐이유는 y축의 방향이 아래로 내려오기 때문
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)

    # np.polyfit()함수를 사용하기 위해 점들을 추출
    # Right lane lines
    right_lines_x = []
    right_lines_y = []
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        right_lines_y.append(y1)
        right_lines_y.append(y2)

    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1) # y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False

    # Left lane lines
    left_lines_x = []
    left_lines_y = []
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        left_lines_y.append(y1)
        left_lines_y.append(y2)

    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
        left_m, left_b = 1, 1
        draw_left = False

    # 왼쪽과 오른쪽의 2개의 점을 찾기, y는 알고 있으므로 x만 찾으면됨
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)
    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m
    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m

    # 모든 점은 정수형이어야 함(정수형으로 바꾸기)
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    # 차선그리기
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

name = '도로4'
capture = cv2.VideoCapture(f'videos/{name}.mp4')
video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0

trap_bottom_width_p1 = .85  ## 우측 아래
trap_bottom_width_p2 = .15  ## 좌측 아래
trap_top_width_p1 = .45 ## 좌측 상단
trap_top_width_p2 = .55 ## 우측 상단
trap_height_p1 = .5 ## 상단 높이
trap_height_p2 = .8
## 아래 높이

lower_yellow = (15, 100, 100)  # 자료형은 튜플형태로(H,S,V)
upper_yellow = (40, 255, 255)  # 자료형은 튜플형태로(H,S,V)

lower_white = (170, 170, 170)  # 자료형은 튜플형태로(B,G,R) 180, 0, 100
upper_white = (255, 255, 255)  # 자료형은 튜플형태로(B,G,R) 200, 255, 255

if capture.isOpened() == False:
    print("카메라를 열 수 없습니다.")
    exit(1)

startpoint = 0 ## 4000
capture.set(cv2.CAP_PROP_POS_FRAMES, startpoint)

while True:
    ret, img_src = capture.read()
    img_dst = img_src.copy()

    img_mask = np.zeros_like(img_src)

    height, width = img_src.shape[:2]

    pts = np.array([[(int(width * trap_bottom_width_p1), int(height * trap_height_p2))], ## 우측 아래
                   [(int(width * trap_bottom_width_p2), int(height * trap_height_p2))], ## 좌측 아래
                   [(int(width * trap_top_width_p1), int(height * trap_height_p1))], ## 좌측 상단
                   [(int(width * trap_top_width_p2), int(height * trap_height_p1))]], ## 우측 상단
                  dtype=np.int32)

    src = cv2.fillPoly(img_mask, [pts], (255, 255, 255))

    img_bit = cv2.bitwise_and(img_src, src)

    img_bit_B,img_bit_G,img_bit_R = cv2.split(img_bit)

    img_bit_hsv = cv2.cvtColor(img_bit, cv2.COLOR_BGR2HSV)

    # if ret == False: # 동영상 끝까지 재생
    #     print("동영상 읽기 완료")
    #     break
    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정

    img_hsv = cv2.cvtColor(img_bit,cv2.COLOR_BGR2HSV)

    img_mask_w = cv2.inRange(img_bit,lower_white,upper_white)
    img_mask_y = cv2.inRange(img_bit_hsv, lower_yellow, upper_yellow)

    img_mask_wy = cv2.addWeighted(img_mask_w,1.,img_mask_y,1.,0)

    img_temp = cv2.bitwise_and(img_bit, img_bit, mask=img_mask_wy)

    img_gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

    img_gray1 = cv2.GaussianBlur(img_gray, (5, 5), 10)

    _, img_binary = cv2.threshold(img_gray1, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    get = 165
    up = get ## 165
    down = get + 5 ## 170
    _, img_binary1 = cv2.threshold(img_bit_B, up, down, cv2.THRESH_BINARY + cv2.THRESH_BINARY)
    _, img_binary2 = cv2.threshold(img_bit_G, up, down, cv2.THRESH_BINARY + cv2.THRESH_BINARY)
    _, img_binary3 = cv2.threshold(img_bit_R, up, down, cv2.THRESH_BINARY + cv2.THRESH_BINARY)

    img_binary_m = cv2.merge([img_binary3,img_binary2,img_binary1])

    # img_canny = cv2.Canny(img_binary, 50, 150)

    img_canny = cv2.Canny(img_binary, 100, 250)

    img_canny1 = cv2.Canny(img_binary1, 50, 250)

    # 사다리꼴 적용하기
    img_roi = img_canny1
    rho = 2  ## 2 누산평명 거리
    theta = 1 * np.pi / 180 ## 1 * np.pi / 180 누산평면 각도
    threshold = 5 ## 15 누산평면값 직선 결정 임계값
    min_line_length = 10 ## 10 최소 선 길이 20,15
    max_line_gap = 5 ## 20 최대 허용 선 간격
    lines = cv2.HoughLinesP(img_roi, rho, theta, threshold,
                           minLineLength=min_line_length,
                           maxLineGap=max_line_gap)

    draw_lines(img_src,lines,(0,255,0),5)

    for i, line in enumerate(lines):
       cv2.line(img_src, (line[0][0], line[0][1]),
                (line[0][2], line[0][3]), (0, 255, 0), 2)


    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # cv2.imshow('Video-y', img_temp_y)
    # cv2.imshow('Video-w', img_temp_w)
    # cv2.imshow('Video', img_bit)
    img_mask_y = cv2.merge([img_mask_y,img_mask_y,img_mask_y])
    img_mask_w = cv2.merge([img_mask_w, img_mask_w, img_mask_w])
    img_mask_wy = cv2.merge([img_mask_wy, img_mask_wy, img_mask_wy])
    img_canny = cv2.merge([img_canny, img_canny, img_canny])
    img_binary = cv2.merge([img_binary, img_binary, img_binary])
    # img_gray1 = cv2.merge([img_gray1, img_gray1, img_gray1])

    img_canny1 = cv2.merge([img_canny1, img_canny1, img_canny1])
    img_binary1_1 = cv2.merge([img_binary1, img_binary1, img_binary1])

    _, img_dst = imgsize(img_dst, 500)
    _, img_src = imgsize(img_src, 500)
    _, img_bit = imgsize(img_bit, 500)
    _, img_mask_w = imgsize(img_mask_w, 500)
    _, img_mask_y = imgsize(img_mask_y, 500)
    _, img_mask_wy = imgsize(img_mask_wy, 500)
    _, img_canny = imgsize(img_canny, 500)
    _, img_binary = imgsize(img_binary, 500)
    _, img_gray = imgsize(img_gray, 500)
    _, img_temp = imgsize(img_temp, 500)

    _, img_bit_R = imgsize(img_bit_R, 500)
    _, img_bit_B = imgsize(img_bit_B, 500)
    _, img_bit_G = imgsize(img_bit_G, 500)
    _, img_binary1 = imgsize(img_binary1, 500)
    _, img_binary2 = imgsize(img_binary2, 500)
    _, img_binary3 = imgsize(img_binary3, 500)
    _, img_binary_m = imgsize(img_binary_m, 500)
    _, img_canny1 = imgsize(img_canny1, 500)
    _, img_binary1_1 = imgsize(img_binary1_1, 500)
    _, img_mask = imgsize(img_mask, 500)

    img_gray = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)

    a = cv2.hconcat([img_dst, img_src, img_bit])
    b = cv2.hconcat([img_temp, img_binary, img_canny])
    c = cv2.vconcat([a, b])
    h, w = c.shape[:2]

    A = cv2.hconcat([img_bit_G, img_bit_B, img_bit_R])
    B = cv2.hconcat([img_binary1, img_binary2, img_binary3])
    # AA = cv2.hconcat([img_binary1_1, img_canny1,im])
    C = cv2.vconcat([A, B])

    if video_length != startpoint + count:
        count += 1
        pass
    if video_length == startpoint + count:
        count = 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, startpoint)
        pass
    cv2.putText(c,f'{count} / {video_length - startpoint}',(w-300,h-20),cv2.FONT_ITALIC,.8,(255,255,255),2)
    cv2.putText(c, 'R : stop , S : start point', (w - 300, h - 45), cv2.FONT_ITALIC, .5, (255, 255, 255), 2)
    cv2.imshow('Video', c)
    cv2.imshow('Video1', C)
    # cv2.imshow('Video', img_bit)

    key = cv2.waitKey(25) # 33ms마다
    if key == 27:         # Esc 키
        break
    elif key == 114:
        cv2.waitKey(0)
    elif key == 115:
        start = startpoint + int(input("입력하세요."))
        count = start - startpoint # 300 ~ 380 , 360
        capture.set(cv2.CAP_PROP_POS_FRAMES,start)
        cv2.waitKey(0)

capture.release()
cv2.destroyAllWindows()