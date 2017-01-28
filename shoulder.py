#coding: utf-8

# 肩幅を取得するプログラム

import numpy as np
import cv2

#カスケード分類器のPATHを通して読み込む
CLASSIFIER_PATH = '../haarcascades/haarcascade_frontalface_default.xml'
shoulderCascade = cv2.CascadeClassifier(CLASSIFIER_PATH)

#肩周りだけトリミングして輪郭線を取る関数
def shoulderExtraction():

    #検出処理
    shoulder = shoulderCascade.detectMultiScale(gray, 1.1, 3)

    #顔の周りをトリミングして保存
#    i = 0
#    for rect in face:
#      x = rect[0] * 0.5
#      y = rect[1] * 0.05
#      width  = rect[2] * 1.8
#      height = rect[3] * 1.8
#      dst = img[y:y+height, x:x+width]
#      cv2.imwrite('image/face_img5.jpg', dst)
#      cv2.imshow('face_img', dst)
#      i += 1

    #キャニー法で２値化
    canny = cv2.Canny(gray, 50,150)
    cv2.imwrite('image/face_canny5.jpg', canny)
    cv2.imshow('face_canny', canny)

    #輪郭抽出
    hoge, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite('image/contours_img5.jpg', img)
    cv2.imshow('contours_img5.jpg', img)


if __name__ == "__main__":

    #画像の読み込み
    img = cv2.imread('image/image5.jpg')

    #サイズ変更
    hight = img.shape[0]
    width = img.shape[1]
    if hight > 800 and 800 < width:
        img = cv2.resize(img, (int(hight/2), int(width/2)))
    elif hight < 400 and 400 < width:
        img = cv2.resize(img, ((hight*2), (width*2)))

    #グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    shoulderExtraction()

    #表示終了処理
    cv2.waitKey(0)
    cv2.destroyAllWindows()