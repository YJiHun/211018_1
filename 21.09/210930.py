import glob
from MyLibrary import myopencv
import numpy as np
import pickle, cv2

def loding(): ## 이미지 뷰어
    images = glob.glob('images/*.jpg')
    total_images = len(images) - 1
    #
    idx = 0
    while True:
        fname = images[idx]

        img = cv2.imread(fname)

        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        out_str = f'{idx} / {total_images}'

        cv2.putText(gray, out_str, (10, 20), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 2)

        cv2.imshow("dst", gray)

        key = cv2.waitKey(0)

        if key == 27:
            idx -= 1
            pass
        else:
            idx += 1
            pass
        if idx < 0:
            idx = 0
            pass

def undistort_img(): ## 이미지 캘리브레이션 (이미지 보정)
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros((6 * 9, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []
    # Get directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')
    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    # Test undistortion on img
    img_size = (img.shape[1], img.shape[0])
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))


# ##
# #
#
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
#
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d points in real world space
# imgpoints = [] # 2d points in image plane.
# # Make a list of calibration images
#
# # 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
# images = glob.glob('camera_cal/*.jpg')
# # Step through the list and search for chessboard corners
#
# total_images = len(images)
# for idx, fname in enumerate(images):
#    img = cv2.imread(fname)
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    # Find the chessboard corners
#    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
#    # If found, add object points, image points
#    if ret == True:
#        objpoints.append(objp)
#        imgpoints.append(corners)
#        # Draw and display the corners
#        cv2.drawChessboardCorners(img, (9,6), corners, ret)
#        write_name = 'camera_cal/result/corners_found'+str(idx)+'.jpg'
#        cv2.imwrite(write_name, img)
#        out_str = f'{idx}/{total_images}'
#        cv2.putText(img, out_str, (10, 25),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
#        cv2.imshow('img', img)
#        cv2.waitKey(500)
# cv2.destroyAllWindows()
#
# #
#
# img = cv2.imread('camera_cal/test_cal.jpg')
# height, width = img.shape[:2]
# img_size = (width, height)
# # Do camera calibration given object points and image points
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
#                                img_size, None,None)
# dst = cv2.undistort(img, mtx, dist, None, mtx)
# cv2.imwrite('camera_cal/result/test_undist.jpg',dst)
#
# # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# dist_pickle = {}
# dist_pickle["mtx"] = mtx
# dist_pickle["dist"] = dist
# pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )
#
# # img_result = cv2.hconcat([img,dst])
# # img_result = cv2.pyrDown(img_result)
# # cv2.imshow('dst',img_result)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# ##

def Undistort(img, url = 'camera_cal/wide_dist_pickle.p'): ## 보정 파라미터 사용
    with open(url, mode='rb') as f:
        file = pickle.load(f)
        mtx = file['mtx']
        dist = file['dist']

    return cv2.undistort(img, mtx, dist, None, mtx)


if __name__ == '__main__':
    video = cv2.VideoCapture('videos/도로_210930.mp4')

    while True:
        _, Video = video.read()

        Video_c = Undistort(Video)

        key = cv2.waitKey(30)

        result = cv2.hconcat([Video,Video_c])

        cv2.imshow('dst', result)

        if key == 27:
            cv2.destroyAllWindows()
            break
    pass