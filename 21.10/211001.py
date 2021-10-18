import cv2
import numpy as np
from MyLibrary import myopencv as my

margin = 150
nwindows = 9
minpix = 1

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

left_fit_ = np.empty(3)
right_fit_ = np.empty(3)

# a = cv2.imread('videos/hls.jpg')
a = my.Imgread('videos/hls.jpg', 0)
print(a.shape)
# img_zero = np.zeros_like(a)

img_binary = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
print(img_binary.shape)
height, width = img_binary.shape[:2]

histogram = np.sum(img_binary[height // 2:, :], axis=0)
# print("histogram", histogram)

midpoint = int(histogram.shape[0] / 2)
# print(histogram.shape[0])
# print(len(histogram))
# print("midpoint", midpoint)

leftx_base = np.argmax(histogram[:midpoint])
# print("leftx_base", leftx_base, histogram[leftx_base])
# print("left", histogram[:midpoint])

rightx_base = np.argmax(histogram[midpoint:]) + midpoint
# print("rightx_base", rightx_base, histogram[rightx_base])
# print("right", histogram[midpoint:])

window_height = int(height / nwindows)

nonzero = img_binary.nonzero()
nonzero_y = np.array(nonzero[0])
nonzero_x = np.array(nonzero[1])
# print(nonzero)
# print("x",nonzero[0])
# print("y",nonzero[1])

leftx_current = leftx_base
rightx_current = rightx_base

left_lane_inds=[]
right_lane_inds=[]

for window in range(nwindows):
    win_y_low = height - (window + 1) * window_height
    win_y_high = height - window * window_height

    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin

    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    cv2.rectangle(a, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                  (100,255,255), 3)
    cv2.rectangle(a, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
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

left_a.append(left_fit[0])
left_b.append(left_fit[1])
left_c.append(left_fit[2])
right_a.append(right_fit[0])
right_b.append(right_fit[1])
right_c.append(right_fit[2])

left_fit_[0] = np.mean(left_a[-10:])
left_fit_[1] = np.mean(left_b[-10:])
left_fit_[2] = np.mean(left_c[-10:])
right_fit_[0] = np.mean(right_a[-10:])
right_fit_[1] = np.mean(right_b[-10:])
right_fit_[2] = np.mean(right_c[-10:])

ploty = np.linspace(0, height - 1, height)

left_fitx =left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
right_fitx =right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

a[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 100]
a[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 100, 255]

color_img = np.zeros_like(a)
left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
points = np.hstack((left, right))

cv2.polylines(color_img, np.int_(points), False, (0, 255, 255), 10)

a = cv2.addWeighted(a, 1., color_img, 0.4, 0)

# _, result = my.imgsize(a,900)
print(a.shape)

cv2.imshow("result", a)

cv2.waitKey(0)
cv2.destroyAllWindows()