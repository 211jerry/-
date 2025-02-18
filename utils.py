"""OpenCV-Python utils"""
import cv2
import numpy as np


def getContours(img, cThr=[30, 50], showCanny=False, minArea=1000, filter=0, draw=False):
    """获取长方形物体的轮廓，以及按照左上 -> 右上 -> 左下 -> 右下顺序排列的轮廓的四个角点"""
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)  # 膨胀
    imgThre = cv2.erode(imgDial, kernel, iterations=2)  # 腐蚀
    if showCanny:
        cv2.imshow('Canny', imgThre)
    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)  # 获取轮廓参数
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # 获取轮廓的四个角点
            bbox = cv2.boundingRect(approx)  # 给物体添加 bounding box
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)  # 将四个角点按照左上 -> 右上 -> 左下 -> 右下顺序排列
    if draw:  # 画出轮廓
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, finalCountours


def reorder(myPoints):
    """由于获取轮廓的角点时是乱序获取的，因此需要将其按照左上 -> 右上 -> 左下 -> 右下的顺序排列"""
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # 左上
    myPointsNew[3] = myPoints[np.argmax(add)]  # 右下
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # 右上
    myPointsNew[2] = myPoints[np.argmax(diff)]  # 左下
    return myPointsNew


def warpImg(img, points, width, height, pad=20):
    """从画面中抓取出A4纸"""
    points = reorder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (width, height))
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]  # 去掉抓取结果中的一些多出来的白边
    return imgWarp


def findDis(pts1, pts2):
    """推算A4纸上的长方形物体的长和宽"""
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5


def getCircles(img, dp=1.2, minDist=50, param1=20, param2=30, minRadius=10, maxRadius=200, draw=False):
    """获取图像中的圆形物体，返回圆心和半径"""

    o = img.copy()  # 复制原图
    o = cv2.medianBlur(o, 5)  # 使用中值滤波进行降噪
    gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)  # 从彩色图像变成单通道灰度图像
    # 霍夫圆变换来检测圆形
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,  # 使用霍夫梯度方法
        dp=dp,  # 分辨率的反比
        minDist=minDist,  # 圆心之间的最小距离
        param1=param1,  # 边缘检测的高阈值
        param2=param2,  # 圆心检测的阈值
        minRadius=minRadius,  # 最小圆半径
        maxRadius=maxRadius  # 最大圆半径
    )

    # 如果找到了圆
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # 将圆的圆心和半径转换为整数
        # 找出半径最大的圆
        largest_circle = max(circles, key=lambda c: c[2])  # 选出半径最大的圆
        # 如果需要绘制圆
        if draw:
            center = (largest_circle[0], largest_circle[1])  # 圆心坐标
            radius = largest_circle[2]  # 圆半径
            cv2.circle(img, center, radius, (0, 255, 0), 4)  # 绘制圆
            cv2.circle(img, center, 2, (0, 0, 255), 5)  # 绘制圆心

        return img, largest_circle  # 返回处理后的图像和最大的圆
    else:
        return img, []  # 如果没有检测到圆形物体，返回空列表
