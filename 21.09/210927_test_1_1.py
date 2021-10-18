### 사다리꼴
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

name = 'challenge'
capture = cv2.VideoCapture(f'videos/{name}.mp4')
video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

trap_bottom_width_p1 = 1.
trap_bottom_width_p2 = .0
trap_top_width_p1 = .45
trap_top_width_p2 = .55
trap_height_p1 = .6
trap_height_p2 = 1.

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

    if capture.get(cv2.CAP_PROP_POS_FRAMES) ==\
            capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ## 사이즈
    _, img_bit = imgsize(img_bit, 1000)
    ## 사이즈

    cv2.imshow('trap ', img_bit)  # 1번답

    key = cv2.waitKey(30) # 33ms마다

    if key == 27:         # Esc 키
        break

capture.release()
cv2.destroyAllWindows()