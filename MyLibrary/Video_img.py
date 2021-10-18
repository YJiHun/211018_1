import cv2 ,os
## view = 0 (이미지 표출X) / view = 1 (이미지 표출 O)
def VideotoImg(rink,type,view):
    count = 0
    name = 'video_img'
    while True:
        print("파일 생성")
        if not os.path.exists(f"{name}"):
            os.makedirs(f'{name}')
            count = 0
            break
        elif not os.path.exists(f"video_img_{count}"):
            os.makedirs(f'video_img_{count}')
            name = f'video_img_{count}'
            count = 0
            break
        else:
            count += 1
    Video = cv2.VideoCapture(rink)
    print("이미지 변환중")
    while True:
        _, a = Video.read()
        key = cv2.waitKey(30)
        cv2.imwrite(f'{name}/{count}.{type}',a)
        count += 1
        if view == 1:
            cv2.imshow("a", a)
        if key == 27:
            cv2.destroyAllWindows()
            break
    print("동영상 이미지 변환 완료")
    print(f'파일위치 {name}')
if __name__ ==  '__main__':
    ## view = 0 (이미지 표출X) / view = 1 (이미지 표출 O)
    print("메인에서 실행")
    VideotoImg('E:/5G․AI자율주행인력양성/파이썬/21.09/videos/도로4.mp4','jpg',0)
    pass