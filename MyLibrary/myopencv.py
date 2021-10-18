import cv2, pickle
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
        # print("확대 불가")
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
    if size > 0:
        src = cv2.imdecode(url, Colortype)
        scale, result = imgsize(src, size)
        print("이미지 재지정 h, w", result.shape[0], result.shape[1])
        return scale,result
    elif size == 0:
        src = cv2.imdecode(url, Colortype)
        return src

def Undistort(img, url='camera_cal/wide_dist_pickle.p'):  ## 보정 파라미터 사용
    with open(url, mode='rb') as f:
        file = pickle.load(f)
        mtx = file['mtx']
        dist = file['dist']

    return cv2.undistort(img, mtx, dist, None, mtx)

def binary(thresold, binary_img):
    binary = np.zeros_like(binary_img)
    binary[(binary_img >= thresold[0]) & (binary_img <= thresold[1])] = 255
    return binary
    pass

def imgshow(title, src , size = 500):
    _, img = imgsize(src,size)
    return cv2.imshow(title, img)
    pass

def imgmerge(img):
    img_src = cv2.merge([img, img, img])
    return img_src
    pass

if __name__ ==" __main__":
    pass