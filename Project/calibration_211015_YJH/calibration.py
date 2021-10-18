import glob, cv2, pickle
import numpy as np

def undistort_img(cal_img, cal, result):
    obj_pts = np.zeros((6 * 9, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    images = glob.glob(f'{cal_img}/*.jpg')
    total_images = len(images)
    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            write_name = f'{cal}/corners_found' + str(indx) + '.jpg'
            cv2.imwrite(write_name, img)
            out_str = f'{indx}/{total_images}'
            cv2.putText(img, out_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            img = cv2.pyrDown(img)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

    img_size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open(f'{result}/cal_pickle.p', 'wb'))

def Undistort(img, url = 'black_box/wide_dist_pickle.p'): ## 보정 파라미터 사용
    with open(url, mode='rb') as f:
        file = pickle.load(f)
        mtx = file['mtx']
        print('mtx', mtx)
        dist = file['dist']
        print('dist', dist)

    return cv2.undistort(img, mtx, dist, None, mtx)

if __name__ == '__main__':
    undistort_img('img', 'conser', 'result')

    img = cv2.imread('img/9.jpg')
    cal = Undistort(img, 'result/cal_pickle.p')
    show = cv2.hconcat([img, cal])
    show = cv2.pyrDown(show)

    cv2.imshow("show", show)
    cv2.imwrite('result/orginal.jpg', img)
    cv2.imwrite('result/calibration.jpg', cal)
    cv2.imwrite('result/result_cal.jpg', show)

    cv2.waitKey(0)
    cv2.destroyAllWindows()