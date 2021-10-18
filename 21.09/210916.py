import sys , threading, cv2, copy
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
import numpy as np

# UI파일 연결
# 단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("opencv_210916.ui")[0]
# 화면을 띄우는데 사용되는 Class 선언

class Thr(threading.Thread):
    def __init__(self,perant):
        super().__init__()
        self.perant = perant
        self.url = self.perant.url.text()
        self.state = 0
        pass

    def display_output_image(self, img, mode):
        h, w = img.shape[:2]  # 그레이영상의 경우 ndim이 2이므로 h,w,ch 형태로 값을 얻어올수 없다

        if img.ndim == 2:
            qImg = QImage(img, w, h, w * 1, QImage.Format_Grayscale8)
        else:
            bytes_per_line = img.shape[2] * w
            qImg = QImage(img, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap(qImg)
        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)  # 이미지 비율유지
        # pixmap = self.pixmap.scaled(600, 450, QtCore.Qt.IgnoreAspectRatio)  # 이미지를 프레임에 맞춤

        if mode == 0:
            self.perant.label_1.setPixmap(pixmap)
            self.perant.label_1.update()  # 프레임 띄우기
        else:
            self.perant.label_2.setPixmap(pixmap)
            self.perant.label_2.update()  # 프레임 띄우기

    def run(self):
        # img = cv2.imread(self.perant.url.text())
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # self.display_output_image(img, 0)

        filename = np.fromfile(self.perant.url.text(), dtype=np.uint8)

        size = 600
        scale = 1
        state = 0

        self.perant.serch.setEnabled(False)
        self.perant.stop.setEnabled(False)
        video = cv2.VideoCapture(self.perant.url.text())

        if video.isOpened() == False:
            self.perant.label_1.setText("파일을열수없습니다.")
            exit(1)

        if state == 0:
            while True:
                try:
                    ret, Video = video.read()

                    Video_1 = cv2.cvtColor(Video, cv2.COLOR_BGR2RGB)
                    Video_2 = cv2.cvtColor(Video_1, cv2.COLOR_RGB2GRAY)
                    H, W = Video.shape[:2]

                    if H > W:
                        if H > size:
                            scale = size / H
                    elif W > H:
                        if W > size:
                            scale = size / W

                    Video_1 = cv2.resize(Video_1, None, fx=scale, fy=scale)
                    Video_2 = cv2.resize(Video_2, None, fx=scale, fy=scale)

                    h, w, c = Video_1.shape

                    qqvideo_1 = QImage(Video_1, w, h, w * c, QImage.Format_RGB888)
                    qqvideo_2 = QImage(Video_2, w, h, w * 1, QImage.Format_Grayscale8)

                    pixmap_1 = QPixmap.fromImage(qqvideo_1)
                    pixmap_2 = QPixmap.fromImage(qqvideo_2)

                    self.perant.label_1.setPixmap(pixmap_1)

                    if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
                        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    cv2.waitKey(0)

                    if self.perant.check.text() == "초기화":
                        self.perant.A = 0
                        self.perant.stop.setEnabled(True)
                        if self.perant.stop.text() == "일시정지":
                            self.perant.label_2.setPixmap(pixmap_2)

                    if self.perant.check.text() == "로드" and self.perant.A == 0:
                        self.perant.A = 1
                        self.perant.label_1.clear()
                        self.perant.label_2.clear()
                        self.perant.serch.setEnabled(True)
                        break

                except cv2.error:
                    state = 1
                    break
                if state == 1:
                    break
                    pass

        if state == 1:
            img = cv2.imdecode(filename, cv2.IMREAD_COLOR)
            H, W = img.shape[:2]

            if H > W:
                if H > size:
                    scale = size / H
            elif W > H:
                if W > size:
                    scale = size / W

            img = cv2.resize(img, None, fx=scale, fy=scale)

            img_t = copy.deepcopy(img)
            img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ###################################

            # ret ,img_b = cv2.threshold(img_1, 150, 255, cv2.THRESH_BINARY_INV)
            #
            # contours, hierarchy = cv2.findContours(img_b,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            #
            # my_color = (0,255,0)
            # text_color = (255,0,0)
            # thickness = 2
            #
            # for i, contour in enumerate(contours):
            #     area = cv2.contourArea(contour)
            #
            #     if area > 100:
            #         cv2.drawContours(img,contours,i,my_color,thickness)
            #
            #         mu = cv2.moments(contour)
            #         cx = int(mu['m10'] / (mu['m00'] + 1e-5))
            #         cy = int(mu['m01'] / (mu['m00'] + 1e-5))
            #
            #         cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
            #         cv2.putText(img, f'{i}: {int(area)}', (cx - 50, cy - 20),
            #                     cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 1)
            #
            #         x,y,w,h = cv2.boundingRect(contour)
            #         cv2.rectangle(img, (x,y), (x+w,y+h),(255,255,0), 1)
            #
            #         rect = cv2.minAreaRect(contour)
            #         box = cv2.boxPoints(rect)
            #         box = np.int0(box)
            #
            #         cv2.drawContours(img, [box], 0, (255,0,255), 1)

            ###################################

            img_2 = cv2.cvtColor(img_1, cv2.COLOR_GRAY2RGB)

            h, w, c = img_t.shape

            qImg_1 = QImage(img_t, w, h, w * c, QImage.Format_RGB888)
            qImg_2 = QImage(img_2, w, h, w * c, QImage.Format_RGB888)

            pixmap_1 = QPixmap.fromImage(qImg_1)
            pixmap_2 = QPixmap.fromImage(qImg_2)

            self.perant.label_1.setPixmap(pixmap_1)

            while True:
                if self.perant.check.text() == "초기화":
                    self.perant.label_2.setPixmap(pixmap_2)
                    self.perant.A = 0

                if self.perant.check.text() == "로드" and self.perant.A == 0:
                    self.perant.A = 1
                    self.perant.stop.setEnabled(True)
                    self.perant.serch.setEnabled(True)
                    self.perant.label_1.clear()
                    self.perant.label_2.clear()
                    print("정지")
                    break
        pass

class DialogClass(QDialog, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # __init__ 함수 밑에 코드를 작성하여야 GUI 폼에서 작동함.
        self.serch.clicked.connect(self.Serch)
        self.check.clicked.connect(self.Check)
        self.stop.clicked.connect(self.Stop)
        self.A = 1

    def Serch(self):
        # filter = "All Images(*.jpg; *.png; *.bmp);;JPG (*.jpg);;PNG(*.png);;BMP(*.bmp)"
        filter = "All Images(*.jpg; *.png; *.bmp; *.mp4; *.avi);;JPG (*.jpg);;PNG(*.png);;BMP(*.bmp)"
        while True:
            if self.check.text() == "로드" and self.stop.text() == "일시정지":
                fname = QFileDialog.getOpenFileName(caption="파일 찾기",directory="C:/Users/1/Desktop/images (1)",filter=filter)
                self.url.setText(fname[0]) ## 텍스트값 넣기.
                break
                # self.A = 0
            else:
                self.check.setText("로드")
                self.stop.setText("일시정지")
        if fname[0] != "":
            Thr(self).start()
            self.serch.setEnabled(False)
        pass

    def Check(self):
        if self.check.text() == "로드":
            self.check.setText("초기화")
        elif self.check.text() == "초기화":
            self.check.setText("로드")
        pass

    def Stop(self):
        if self.stop.text() == "일시정지":
            self.stop.setText("다시시작")
        elif self.stop.text() == "다시시작":
            self.stop.setText("일시정지")
        pass

if __name__ == "__main__":
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # DialogClass의 인스턴스 생성
    myDialog = DialogClass()

    # 프로그램 화면을 보여주는 코드
    myDialog.show()

    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
