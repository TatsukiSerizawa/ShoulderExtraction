#coding: utf-8

# 肩幅を取得するプログラム

import numpy as np
import cv2

#カスケード分類器のPATHを通して読み込む
CLASSIFIER_PATH = '../haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(CLASSIFIER_PATH)

#肩周りだけトリミングして輪郭線を取る関数
def shoulderExtraction():

    #検出処理
    face = faceCascade.detectMultiScale(gray, 1.1, 3)

    #肩の周りをトリミングして保存
    for rect in face:
      x = 0
      y = rect[2]
      width  = img.shape[1]
      height = rect[3] * 1.2
      dst = img[y:y+height, x:x+width]
      cv2.imwrite('image/shoulder_img5.jpg', dst)
      cv2.imshow('shoulder_img5', dst)

    #キャニー法で２値化
    canny = cv2.Canny(dst, 50,150)
#    cv2.imwrite('image/shoulder_canny5.jpg', canny)
#    cv2.imshow('shoulder_canny', canny)

    #輪郭抽出
    hoge, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst, contours, -1, (0, 255, 0), 2)
    cv2.imwrite('image/contours_img5.jpg', dst)
    cv2.imshow('contours_img5.jpg', dst)

    #高さの中央値を抽出
    imgShape = cv2.imread('image/contours_img5.jpg')
    height, width = imgShape.shape[:2]
    heightCenter = height//2
    print(heightCenter, width)
    print(imgShape[heightCenter][250])

    #最初に出てきた[0,255,0]の座標を肩の位置として記憶
    for i in range(width):
        if imgShape([[heightCenter][i]]) == ([0, 255, 0]):
            shoulderRight = imgShape[heightCenter][i]
            break
    
    print(shoulderRight)


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