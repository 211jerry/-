import cv2
import numpy as np
import utils  # 注意更改被调用文件的路径

###############################
# 参数设定
webcam = False  # 是否打开电脑自带摄像头或外置摄像头进行实时尺寸推算，如果想要进行实时，将False修改为True即可
path = "test_circle.jpg"  # 存放调用文件的路径
cap = cv2.VideoCapture(0)  # 摄像头
cap.set(3, 1920)  # 摄像头捕捉画面的宽
cap.set(4, 1080)  # 摄像头捕捉画面的高
cap.set(10, 160)  # 摄像头捕捉画面的亮度
scale = 3  # 用于尺寸的缩放
widthA4 = 210 * scale  # A4纸的宽
heightA4 = 297 * scale  # A4纸的长
###############################

while True:
    # 选择实时与否
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    imgContour1, contours1 = utils.getContours(img, minArea=50000, filter=4)  # 抓到画面中面积最大轮廓（即A4纸）

    if len(contours1) != 0:
        biggest = contours1[0][2]  # 获取A4纸轮廓的四个角点

        # 将画面中的A4纸抓出来
        imgWarp = utils.warpImg(img, biggest, widthA4, heightA4)

        # 抓到A4纸中的圆形物体的轮廓
        imgContour2, contours2 = utils.getCircles(imgWarp, draw=True)
        # 让A4纸中的圆形物体的轮廓看起来更加合适、顺滑
        if len(contours1) != 0:

            # 推算A4纸中的圆形物体的半径
            center = (contours2[0], contours2[1])  # 圆心坐标
            radius = contours2[2]  # 圆半径
            cv2.circle(img, center, radius, (0, 255, 0), 4)  # 绘制圆
            cv2.circle(img, center, 2, (0, 0, 255), 5)  # 绘制圆心

            # 在显示时，对A4纸中的圆形物体的半径和周长进行标识以及赋予数值

            x, y, r = contours2
            cv2.putText(imgContour2, '{}cm'.format(round((radius // scale)/10 , 1)), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow("A4", imgContour2)  # 展示推算结果

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)  # 由于拍摄的图片尺寸较大，将其缩小为原本的 1/2
    cv2.imshow("Original", img)  # 展示用于推算尺寸的图像或摄像机原始镜头所拍摄的情况
    cv2.waitKey(1)