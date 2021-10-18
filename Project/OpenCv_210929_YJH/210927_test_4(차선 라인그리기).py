### 라인 그리기
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

    # print("원래이미지 h, w",src.shape[0],src.shape[1])

    scale = size / max
    return scale

def imgsize(src, size): ## src = 불러온 이미지 소스 , size = 변경할 크기
    scale = standard(src, size)
    return scale,cv2.resize(src, None, fx=scale, fy=scale)

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

name = 'challenge'
capture = cv2.VideoCapture(f'videos/{name}.mp4')
video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

trap_bottom_width_p1 = 1.
trap_bottom_width_p2 = .0
trap_top_width_p1 = .45
trap_top_width_p2 = .55
trap_height_p1 = .6
trap_height_p2 = 1.

lower = 180
upper = 255
lower_white = (lower, lower, lower)
upper_white = (upper, upper, upper)

lower_yellow = (15, 100, 100)
upper_yellow = (40, 255, 255)

thr_lower = 0
thr = 255

if capture.isOpened() == False:
    print("카메라를 열 수 없습니다.")
    exit(1)

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

    ## 색삭 추출
    img_bit_w = cv2.inRange(img_bit, lower_white, upper_white)
    img_bit_w_b = cv2.bitwise_and(img_bit, img_bit, mask=img_bit_w)

    img_bit_y = cv2.inRange(img_bit, lower_yellow, upper_yellow)
    img_bit_y_b = cv2.bitwise_and(img_bit, img_bit, mask=img_bit_y)

    img_bit_wy = cv2.addWeighted(img_bit_w, 1., img_bit_y, 1., 0.)
    img_bit_wy_b = cv2.bitwise_and(img_bit, img_bit, mask=img_bit_wy)
    ## 색삭 추출 end

    img_gray = cv2.cvtColor(img_bit_wy_b,cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (1, 1), 2)

    _, img_thr = cv2.threshold(img_blur, thr_lower, thr, cv2.THRESH_OTSU
                               + cv2.THRESH_BINARY)

    img_bit_canny = cv2.Canny(img_thr, 50, 150)

    ## 색추출 후 라인 따기
    # 선따기
    img_roi = img_bit_canny
    rho = 2
    theta = 1 * np.pi / 180
    threshold = 30
    min_line_length = 10
    max_line_gap = 20

    lines = cv2.HoughLinesP(img_roi, rho, theta, threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    ## 색추출 후 라인 따기 end

    ## 선 그리기
    draw_lines(img_src, lines, (0, 255, 0), 12)
    ## 선 그리기

    ## 추출 영역 표시
    img_src = cv2.polylines(img_src, [pts],True, (255, 0, 255),3)
    ## 추출 영역 표시

    ## 사이즈
    _, img_src = imgsize(img_src, 1000)
    ## 사이즈

    cv2.imshow('draw_lines', img_src) # 4번답

    key = cv2.waitKey(30) # 33ms마다

    if key == 27:         # Esc 키
        break

capture.release()
cv2.destroyAllWindows()