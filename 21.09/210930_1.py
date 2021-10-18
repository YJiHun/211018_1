from MyLibrary import myopencv as my
import cv2
import numpy as np


def binary(thresold, binary_img):
    binary = np.zeros_like(binary_img)
    binary[(binary_img >= thresold[0]) & (binary_img <= thresold[1])] = 255
    return binary
    pass

def babo(img,src,dst):
    src = src * np.float32([width, height])  ## width, height 비율 값
    dst = dst * np.float32(dst_size)  ## 이미지를 적용할 화면 비율
    M = cv2.getPerspectiveTransform(src, dst)  ## 자를 이미지 좌표값
    img_src = cv2.warpPerspective(img, M, dst_size)  ## 잘라낼 이미지, 잘라낼 이미지 영역값, 잘라낼 이미지를 붙일 영역 사이즈
    return img_src
    pass

name = 'challenge'
Video = cv2.VideoCapture(f'videos/{name}.mp4')

margin = 150
nwindows = 9
minpix = 1

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

left_fit_ = np.empty(3)
right_fit_ = np.empty(3)

A = 0
while True:
    _, img = Video.read()
    height, width = img.shape[:2]

    color_img = np.zeros_like(img)

    if A == 0:
        A = 1
        print(img.shape)

    # ##사다리꼴 영역
    # img_temp = np.zeros_like(img)
    #
    # pts = np.array([[(int(width * trap_bottom_width_p1), int(height * trap_height_p2))], ## 우측 아래
    #                [(int(width * trap_bottom_width_p2), int(height * trap_height_p2))], ## 좌측 아래
    #                [(int(width * trap_top_width_p1), int(height * trap_height_p1))], ## 좌측 상단
    #                [(int(width * trap_top_width_p2), int(height * trap_height_p1))]], ## 우측 상단
    #               dtype=np.int32)
    #
    # img_temp = cv2.fillPoly(img_temp, [pts], (255, 255, 255))
    #
    # img = cv2.bitwise_and(img, img_temp)
    # ##사다리꼴 영역

    img_undist = my.Undistort(img)

    img_hls = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HLS)

    img_hls_h, img_hls_l, img_hls_s = cv2.split(img_hls)

    img_sobel_x = cv2.Sobel(img_hls_l, cv2.CV_64F, 1, 1)
    img_sobel_x_abs = abs(img_sobel_x)
    img_sobel_scaled = np.uint8(img_sobel_x_abs * 255 / np.max(img_sobel_x_abs))

    sx_thresold = (15, 255)
    s_thresold = (100, 255)

    a = binary(sx_thresold, img_sobel_scaled)
    b = binary(s_thresold, img_hls_s)

    img_binary_added = cv2.addWeighted(a, 1., b, 1., 0)

    height, width = img_binary_added.shape[:2]

    dst_size = (1280, 720)

    src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]) # TopLeft, TopRight , BottomLeft, BottomRight
    # src = np.float32([(.6, .45), (.8, .45), (.6, .8), (.8, .8)]) # TopLeft, TopRight , BottomLeft, BottomRight
    dst = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]) ## 이미지를 적용할 화면 비율

    img_warp = babo(img_binary_added,src,dst)

    # ## 이미지 자른 영역표시
    # B = np.float32([(0.43, 0.65), (0.58, 0.65), (1., 1.), (0.1, 1.)])  # TopLeft, TopRight , BottomLeft, BottomRight
    # B = B * np.float32([width, height])  ## width, height 비율 값
    # B = np.int32(B)
    # cv2.polylines(img_binary_added, [B], True, color=(255, 0, 255), thickness=5)
    # ## 이미지 자른 영역표시

    # _, img_binary_added = my.imgsize(img_binary_added, 900)
    # _, img_warp = my.imgsize(img_warp, 900)

    img_A = cv2.merge([img_warp,img_warp,img_warp])

    # img_A = cv2.cvtColor(img_a,cv2.COLOR_BGR2GRAY)

    height_1, width_1 = img_warp.shape[:2]

    histogram = np.sum(img_warp[height_1 // 2:, :], axis=0)

    midpoint = int(histogram.shape[0] / 2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(height_1 / nwindows)

    nonzero = img_warp.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(color_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (100, 255, 255), 3)
        cv2.rectangle(color_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (100, 255, 255), 3)

        good_left_inds = ((nonzero_y >= win_y_low) &
                          (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) &
                          (nonzero_x < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzero_y >= win_y_low) &
                           (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) &
                           (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzero_x[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)

    right_fit = np.polyfit(righty, rightx, 2)

    # left_a.append(left_fit[0])
    # left_b.append(left_fit[1])
    # left_c.append(left_fit[2])
    # right_a.append(right_fit[0])
    # right_b.append(right_fit[1])
    # right_c.append(right_fit[2])
    #
    # left_fit_[0] = np.mean(left_a[-10:])
    # left_fit_[1] = np.mean(left_b[-10:])
    # left_fit_[2] = np.mean(left_c[-10:])
    # right_fit_[0] = np.mean(right_a[-10:])
    # right_fit_[1] = np.mean(right_b[-10:])
    # right_fit_[2] = np.mean(right_c[-10:])

    left_fit_[0] = left_fit[0]
    left_fit_[1] = left_fit[1]
    left_fit_[2] = left_fit[2]
    right_fit_[0] = right_fit[0]
    right_fit_[1] = right_fit[1]
    right_fit_[2] = right_fit[2]

    ploty = np.linspace(0, height_1 - 1, height_1)

    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

    color_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 100]
    color_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 100, 255]

    # color_img = np.zeros_like(img)
    left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((left, right))

    mid_fitx = (left_fitx + right_fitx) // 2
    # print("mid_fitx", mid_fitx)
    # print("ploty", ploty)
    mid_fitxy = np.vstack([mid_fitx, ploty])
    # print("mid_fitxy", mid_fitxy)
    mid_fitxy_t = np.transpose(mid_fitxy)
    # print("mid_fitxy_t", mid_fitxy_t)
    mid = np.array([mid_fitxy_t])
    # print("mid", mid)

    left_fitx_mean = np.int32(np.mean(left_fitx))
    right_fitx_mean = np.int32(np.mean(right_fitx))
    mid_fitx_mean = np.int32(np.mean(mid_fitx))

    line_color = (0, 255, 0)
    line_tick = 10
    line_len = 20

    cv2.fillPoly(color_img, np.int_(points), line_color)
    cv2.polylines(color_img, np.int_(points), False, (0,255,255), 10)
    cv2.polylines(color_img, np.int_(mid), False, (0, 255, 255), 10)
    cv2.line(color_img, (left_fitx_mean, height_1 // 2 - line_len),
             (left_fitx_mean, height_1 // 2 + line_len),
             line_color, line_tick)
    cv2.line(color_img, (right_fitx_mean, height_1 // 2 - line_len),
             (right_fitx_mean, height_1 // 2 + line_len),
             line_color, line_tick)
    cv2.line(color_img, (mid_fitx_mean, height_1 // 2 - line_len),
             (mid_fitx_mean, height_1 // 2 + line_len),
             line_color, line_tick)
    cv2.line(color_img, (width_1 // 2, height_1 // 2),
             (width_1 // 2, height_1),
             (255, 0, 0), line_tick)
    cv2.line(color_img, (width_1 // 2, height_1 // 2),
             (mid_fitx_mean, height_1 // 2),
             (255, 255, 255), line_tick * 2)

    road_width = 2.5 ## 도로 폭
    road_width_pixel = road_width / (right_fitx_mean - left_fitx_mean) ## pixel 1 = m
    error = width_1 // 2 - mid_fitx_mean ## 도로중심과 이미지 중심이 떨어진 픽셀 거리
    dis_error = error * road_width_pixel ## 도로중심과 이미지 중심이 떨어진 거리

    img_D = babo(color_img, dst, src)
    img_A = cv2.addWeighted(img, 1., img_D, 0.4, 0)

    text_size = 0.5
    text_color = (0, 255, 0)
    text_tick = 1
    text_org = (width_1 // 2 + 40, height_1 // 2 + 150)

    if dis_error > 0: # right
        cv2.putText(img_A, 'right : {:.2f}m'.format(abs(dis_error)),
                    text_org, cv2.FONT_ITALIC,
                    text_size, text_color, text_tick)
        pass
    elif dis_error < 0: # left
        cv2.putText(img_A, 'left : {:.2f}m'.format(abs(dis_error)),
                    text_org, cv2.FONT_ITALIC,
                    text_size, text_color, text_tick)
        pass
    else: # 중심
        cv2.putText(img_A, 'center', text_org, cv2.FONT_ITALIC, text_size,
                    (0, 0, 255), text_tick)
        pass

    # img_D = babo(color_img, dst, src)
    # img_A = cv2.addWeighted(img, 1., img_D, 0.4, 0)

    if Video.get(cv2.CAP_PROP_POS_FRAMES) ==\
            Video.get(cv2.CAP_PROP_FRAME_COUNT):
        Video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.imshow('c', img_A)
    # cv2.imshow('c', img_D)
    # cv2.imshow('c', color_img)

    key = cv2.waitKey(30)

    if key == 27:
        # cv2.imwrite('videos/hls.jpg', img_warp)
        # cv2.imwrite('videos/hls_1.jpg', img_binary_added)
        # cv2.imwrite('videos/hls_2.jpg', a)
        cv2.destroyAllWindows()
        break
