import cv2
import numpy as np
import keyboard


def MOG_mask(source_img, target_img):
    mog = cv2.createBackgroundSubtractorMOG2(2, 25, detectShadows=False)
    # 参数：
    # history：用于训练背景的帧数，默认为500帧,
    # varThreshold：方差阈值，用于判断当前像素是前景还是背景
    # detectShadows：是否检测影子，设为true为检测，false为不检测

    frame = source_img
    for i in range(2):
        mask_img = mog.apply(frame)
        th = cv2.threshold(np.copy(mask_img), 244, 255, cv2.THRESH_BINARY)[1]     # cv2.threshold二值化函数
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)      # 腐蚀操作
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)    # 膨胀操作
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 查找检测物体的轮廓
        # 参数1：图像   参数2：检索模式，cv2.RETR_EXTERNAL表示只检测外轮廓
        # 参数3：method为轮廓的近似办法 cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
        # https://blog.csdn.net/gaoranfighting/article/details/34877549
        for c in contours:
            if cv2.contourArea(c) > 1000:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imshow("mog", mask_img)
        cv2.imshow("thresh", th)
        cv2.imshow("diff", frame & cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR))
        cv2.imshow("detection", frame)
        frame = target_img
        cv2.waitKey(100)
    cv2.waitKey(5000)
    return th


def drawCnt(fn, cnt):
  if cv2.contourArea(cnt) > 1600:
    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.rectangle(fn, (x, y), (x + w, y + h), (255, 255, 0), 2)


def KNN_mask(source_img, target_img):
    knn = cv2.createBackgroundSubtractorKNN(2, 25, detectShadows=False)
    frame = source_img
    for i in range(2):
        mask_img = knn.apply(frame)
        th = cv2.threshold(np.copy(mask_img), 244, 255, cv2.THRESH_BINARY)[1]

        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(th, es, iterations=2)

        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            drawCnt(frame, c)
        cv2.imshow("knn", mask_img)
        cv2.imshow("thresh", th)
        cv2.imshow("detection", frame)
        frame = target_img
        cv2.waitKey(100)
    cv2.waitKey(5000)
    return th


def GMG_mask(source_img, target_img):
    gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
    frame = source_img
    for i in range(2):
        mask_img = gmg.apply(frame)
        th = cv2.threshold(np.copy(mask_img), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 1000:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.imshow("GMG", mask_img)
        cv2.imshow("thresh", th)
        cv2.imshow("diff", frame & cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR))
        cv2.imshow("detection", frame)
        frame = target_img
        cv2.waitKey(100)
    cv2.waitKey(5000)
    return th

'''
for i in range(1589):
    source_img = cv2.imread("E://image_2/%06d.png"%(i))
    target_img = cv2.imread("E://image_2/%06d.png"%(i+1))
    mask = GMG_mask(source_img, target_img)
    if keyboard.is_pressed('Esc'):
        break

    # print("E://image_2/%06d.png"%(i))
    # cv2.imshow("haha",source_img)

'''
img1_path = "E://3.png"
img2_path = "E://4.png"
source_img = cv2.imread(img1_path)
target_img = cv2.imread(img2_path)
# image=cv2.imread("D:\\picture\\%d.jpg"%(i))
mask = MOG_mask(source_img, target_img)

