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

name = '야간3'
capture = cv2.VideoCapture(f'videos/{name}.avi')
video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
relay = 0
count = 0

trap_bottom_width_p1 = 1  ## 우측 아래 .8
trap_bottom_width_p2 = .0  ## 좌측 아래 .2
trap_top_width_p1 = .45 ## 좌측 상단 .45
trap_top_width_p2 = .55 ## 우측 상단 .55
trap_height_p1 = .65 ## 상단 높이 .65
trap_height_p2 = 0.95 ## 하단 1
## 아래 높이

lower = 90 ## 90
upper = 255 ## 255
lower_white = (lower, lower, lower)  # 자료형은 튜플형태로(B,G,R)
upper_white = (upper, upper, upper)  # 자료형은 튜플형태로(B,G,R)

thr_lower = 0 ## 0
thr_upper = 255 ## 255

if capture.isOpened() == False:
    print("카메라를 열 수 없습니다.")
    exit(1)

startpoint = 0 ## 4000
capture.set(cv2.CAP_PROP_POS_FRAMES, startpoint)

while True:
    ret, img_src = capture.read()
    img_dst = img_src.copy()

    ## 사다리꼴
    img_mask = np.zeros_like(img_src)

    height, width = img_src.shape[:2]

    pts = np.array([[(int(width * trap_bottom_width_p1), int(height * trap_height_p2))], ## 우측 아래
                   [(int(width * trap_bottom_width_p2), int(height * trap_height_p2))], ## 좌측 아래
                   [(int(width * trap_top_width_p1), int(height * trap_height_p1))], ## 좌측 상단
                   [(int(width * trap_top_width_p2), int(height * trap_height_p1))]], ## 우측 상단
                  dtype=np.int32)

    src = cv2.fillPoly(img_mask, [pts], (255, 255, 255))

    img_bit = cv2.bitwise_and(img_src, src)
    ## 사다리꼴
    img_bit1 = cv2.inRange(img_bit,lower_white,upper_white)

    img_bit2 = cv2.bitwise_and(img_bit,img_bit,mask=img_bit1)

    img_bit_B, img_bit_G, img_bit_R = cv2.split(img_bit2)

    img_bit_gray = cv2.cvtColor(img_bit2,cv2.COLOR_BGR2GRAY)

    r = 2
    img_bit_blur = cv2.GaussianBlur(img_bit_gray, (1, 1), r)

    img_bit_blur_B = cv2.GaussianBlur(img_bit_B, (1, 1), r)
    img_bit_blur_G = cv2.GaussianBlur(img_bit_G, (1, 1), r)
    img_bit_blur_R = cv2.GaussianBlur(img_bit_R, (1, 1), r)

    _, img_bit_thr = cv2.threshold(img_bit_blur, thr_lower, thr_upper, cv2.THRESH_TOZERO)

    _, img_bit_thr_B = cv2.threshold(img_bit_blur_B, thr_lower, thr_upper, cv2.THRESH_BINARY)
    _, img_bit_thr_G = cv2.threshold(img_bit_blur_G, thr_lower, thr_upper, cv2.THRESH_BINARY)
    _, img_bit_thr_R = cv2.threshold(img_bit_blur_R, thr_lower, thr_upper, cv2.THRESH_BINARY)

    img_canny = cv2.Canny(img_bit_thr, 50, 150)

    img_roi = img_canny  #
    rho = 1  ## 2 누산평명 거리
    theta = 1 * np.pi / 180 ## 1 * np.pi / 180 누산평면 각도
    # print(theta)
    threshold = 15 ## 15 누산평면값 직선 결정 임계값
    min_line_length = 10 ## 10 최소 선 길이 5
    max_line_gap = 20 ## 20 최대 허용 선 간격

    lines = cv2.HoughLinesP(img_roi, rho, theta, threshold,
                           minLineLength=min_line_length,
                           maxLineGap=max_line_gap)

    # 선따기

    # 선 그리기
    draw_lines(img_src, lines, (0, 255, 0), 12)

    try:
        for i, line in enumerate(lines):
            cv2.line(img_bit, (line[0][0], line[0][1]),
                (line[0][2], line[0][3]), (0, 255, 0), 5)
    except TypeError:
        print(count)
    # 선 그리기


    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    img_src = cv2.polylines(img_src, [pts],True, (255, 0, 255),3)

    ## 사이즈
    _, img_src = imgsize(img_src, 500)
    _, img_bit = imgsize(img_bit, 500)
    _, img_bit1 = imgsize(img_bit1, 500)
    _, img_bit2 = imgsize(img_bit2, 500)
    _, img_mask = imgsize(img_mask, 500)
    _, img_bit_R = imgsize(img_bit_R, 500)
    _, img_bit_B = imgsize(img_bit_B, 500)
    _, img_bit_G = imgsize(img_bit_G, 500)
    _, img_bit_thr_B = imgsize(img_bit_thr_B, 500)
    _, img_bit_thr_G = imgsize(img_bit_thr_G, 500)
    _, img_bit_thr_R = imgsize(img_bit_thr_R, 500)
    _, img_bit_thr = imgsize(img_bit_thr, 500)
    _, img_canny = imgsize(img_canny, 500)
    ## 사이즈

    text_p1 = 0
    step = 500
    step1 = step +step

    a = cv2.hconcat([img_src, img_bit,img_bit2])
    cv2.putText(a, f'src', (text_p1, 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)
    cv2.putText(a, f'trap - img_bit', (text_p1 + step, 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)
    cv2.putText(a, f'white - img_bit2', (text_p1 + step1, 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)

    b = cv2.hconcat([img_bit_B, img_bit_G, img_bit_R])

    c = cv2.hconcat([img_bit_thr_B, img_bit_thr_G, img_bit_thr_R])

    e = cv2.hconcat([img_bit_thr, img_canny])

    # cv2.putText(b, f'BGR', (text_p1, 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)
    # cv2.putText(b, f'Thresholding', (text_p1 + step, 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)
    # cv2.putText(b, f'trap', (text_p1 + step1, 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)

    # c = cv2.hconcat([img_binary_1, img_binary_2, img_binary_3])
    # cv2.putText(c, f'B', (text_p1, 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)
    # cv2.putText(c, f'G', (text_p1 + step, 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)
    # cv2.putText(c, f'R', (text_p1 + step1, 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)

    # d = cv2.vconcat([a])
    h,w = a.shape[:2]

    if video_length != startpoint + count:
        count += 1
        pass
    if video_length == startpoint + count:
        count = 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, startpoint)
        pass

    # 오른쪽 아래
    # cv2.putText(d, f'{count} / {video_length - startpoint}', (w - 300, h - 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)
    # cv2.putText(d, 'R : stop , S : start point', (w - 300, h - 45), cv2.FONT_ITALIC, .5, (255, 255, 255), 2)

    # 왼쪽 아래
    cv2.putText(a, f'{count} / {video_length - startpoint}', (0, h - 20), cv2.FONT_ITALIC, .8, (255, 255, 255), 2)
    cv2.putText(a, 'R : stop , S : start point', (0, h - 45), cv2.FONT_ITALIC, .5, (255, 255, 255), 2)

    cv2.imshow('Video', a)
    cv2.imshow('Video1', b)
    cv2.imshow('Video2', c)
    cv2.imshow('Video3', e)

    key = cv2.waitKey(30) # 33ms마다

    if key == 27:         # Esc 키
        break
    if key == 113:
        relay = 0
        pass
    elif key == 114:
        cv2.waitKey(0)
        pass
    elif key == 115:
        start = startpoint + int(input("입력하세요."))
        count = start - startpoint
        capture.set(cv2.CAP_PROP_POS_FRAMES,start)
        cv2.waitKey(0)
        pass
    elif key == 97 or relay == 1:
        thr_lower = int(input(f"{thr_lower} lower 값 입력."))
        thr_upper = int(input(f"{thr_upper} upper 값 입력."))
        # lower_white = (lower, lower, lower)  # 자료형은 튜플형태로(B,G,R) 180, 0, 100
        # upper_white = (upper, upper, upper)  # 자료형은 튜플형태로(B,G,R) 200, 255, 255
        relay = 1
        pass

capture.release()
cv2.destroyAllWindows()