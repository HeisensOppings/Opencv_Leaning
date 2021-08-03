# -*- coding: utf-8 -*-

import cv2
import numpy as np
image = cv2.imread('./image/number_img.png')
img = image

#-------------------------------------------------------minAreaRect最小外接矩形
# cv2.threshold() —— 阈值处理
# cv2.findContours() —— 轮廓检测
# cv2.boundingRect() —— 最大外接矩阵
# cv2.rectangle() —— 画出矩形
# cv2.minAreaRect —— 找到最小外接矩形（矩形具有一定的角度）
# cv2.boxPoints —— 外接矩形的坐标位置
# cv2.drawContours(image, [box], 0, (0, 0, 255), 3) —— 根据点画出矩形

# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
# contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# i = 0

# for c in contours:
#     x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 1)

#     rect = cv2.minAreaRect(c)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
    
#     cv2.drawContours(image, [box], 0, (0, 0, 255), 1)
    
#     minAreaRect = cv2.minAreaRect(c)
#     ((cx, cy), (w, h), theta) = minAreaRect
#     cx = int(cx) 
#     cy = int(cy)
#     wm = int(w)
#     hm = int(h)
    
#     img_file = image[int(cx - w/2): int(cy - h/2) ,int(cx + w/2): int(cy + h/2)]
#     i = i + 1
#     img_name = ("number_minarearect_canvas_%d.png"%i)
#     # cv2.imwrite(img_name, img_file)

#     # 计算最小封闭圆的中心和半径
#     (x, y), radius = cv2.minEnclosingCircle(c)
#     center = (int(x),int(y))
#     radius = int(radius)
#     cv2.circle(image, center, radius, (255, 255, 255), 1)

# cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
# cv2.imshow("img", image)

# =============================================================================
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# contours, hier = cv2.findContours(gray, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)

# canvas = np.copy(img)
# for cidx,cnt in enumerate(contours):
#     minAreaRect = cv2.minAreaRect(cnt)
#     # 转换为整数点集坐标
#     rectCnt = np.int64(cv2.boxPoints(minAreaRect))
#     # 绘制多边形
#     cv2.polylines(img=canvas, pts=[rectCnt], isClosed=True, color=(0,0,255),  thickness=3)

# cv2.imshow("canvas_img", canvas)

#-----------------------------------------------------------------------------
img = cv2.imread("./image/number_org.png")      #读入图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #图片灰度化  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.inRange(gray, 0, 160)      #阈值化法二值化图像
cv2.imshow('org',thresh)                #输出图像
final_img = cv2.medianBlur(thresh,19) 
cv2.imshow('final_img',final_img)                #输出图像

contours, hier = cv2.findContours(final_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#for: 依次对每个由findContours函数取出的轮廓进行操作
for cidx,cnt in enumerate(contours):
    #minAreaRect函数: 取出由findContours函数取出的轮廓的最小可包围的矩形
    minAreaRect = cv2.minAreaRect(cnt)
    ((cx, cy), (w, h), theta) = minAreaRect
    cx = int(cx) #坐标转换成整数
    cy = int(cy)
    w = int(w)
    h = int(h)
    # 获取旋转矩阵
    rotateMatrix = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    # 获取旋转后的图像，即将图像转正，方便下面ROI
    rotatedImg = cv2.warpAffine(img, rotateMatrix, (img.shape[1], img.shape[0]))
    # 绘制矩形区域  pt1,pt2分别为最小矩形的左上角和右下角坐标
    pt1 = (int(cx - w/2), int(cy - h/2))       
    pt2 = (int(cx + w/2), int(cy + h/2))
    #if: 去除受噪点影响所形成的轮廓矩形
    #if ( int(cx - w/2)-int(cx + w/2) > 10 or int(cx - w/2)-int(cx + w/2) < -10):
    cv2.rectangle(rotatedImg, pt1=pt1, pt2=pt2,color=(255, 255, 255), thickness=3)
        #ROI取出兴趣区域，由左上角，右下角坐标取出
    final = rotatedImg[pt1[1]: pt2[1] ,pt1[0]: pt2[0]]
    #cv2.imwrite("minarearect_cidx_{}.png".format(cidx), final)  #保存图片
    cv2.imshow("minarearect_cidx_{}".format(cidx), final)  
#---------------------------------------------------------------------------end
cv2.waitKey(0)
cv2.destroyAllWindows()