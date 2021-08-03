# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:26:52 2021

@author: 30679
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("./image/qizi.png")   #读入图片

# =============================================================================
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 十字结构
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
# 
#-腐蚀erode---膨胀dilate---开运算morphologyEx---闭运算morphologyEx---iterations迭代次数
# =============================================================================

#-----------------------------------------------------------------------------
# 创建 核  即5 * 5 二维数组为uint8类型
kernel = np.ones((5,5), np.uint8)

erorsion_img = cv2.erode(img, kernel, iterations=1)
dilate_img = cv2.dilate(img, kernel, iterations=1)
opening_img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel, iterations=2)
closing_img = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel, iterations=2)

plt.subplot(221),plt.imshow(erorsion_img)
plt.subplot(222),plt.imshow(dilate_img)
plt.subplot(223),plt.imshow(opening_img)
plt.subplot(224),plt.imshow(closing_img)

# # 获取形态学梯度
# gradient = cv2.morphologyEx(img,  cv2.MORPH_GRADIENT, kernel)
# cv2.imshow("gradient_img", gradient)

#--------------------------------------------minAreaRect(InputArray points)函数
# boundingRect读入的参数必须是vector或者Mat点集  用于获取最小面积的矩形
# 返回值minAreaRect ：((cx, cy), (width, height), theta)
# cx 矩形中心点x坐标
# cy 矩形中心点y坐标
# width 矩形宽度
# height 矩形高度
# theta 旋转角度，不是弧度
# 注意：上述值均为小数，不可以直接用于图片索引，或者矩形绘制

#----------------------------rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)函数
# img是原图
# （x，y）是矩阵的左上点坐标
# （x+w，y+h）是矩阵的右下点坐标
# （0,255,0）是画线对应的rgb颜色
# 2是所画的线的宽度

#-------------------------------------------------boxpoints=boxPoints(rect)函数
# 根据minAreaRect的返回值计算矩形的四个点
# rect	minAreaRect的返回值
# boxpoints	矩形的4个点

#-----------------------------------------------------cv2.boundingRect(img)函数
# 用一个最小的矩形，垂直
# 返回值x，y是矩阵左上点的坐标，w，h是矩阵的宽和高

###------------------cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画出矩行

###------------------------------------------------------cv2.findContours()函数
#-----cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])  
#opencv2返回两个值：contours：hierarchy
#opencv3会返回三个值,分别是img, countours, hierarchy
#返回值是轮廓本身，还有一个是每条轮廓对应的属性，其中的元素个数和轮廓个数相同
# 第一个参数是寻找轮廓的图像；
# 第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
#     cv2.RETR_EXTERNAL 表示只检测外轮廓
#     cv2.RETR_LIST 检测的轮廓不建立等级关系
#     cv2.RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
#     cv2.RETR_TREE 建立一个等级树结构的轮廓。
# 第三个参数method为轮廓的近似办法
#     cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
#     cv2.CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
#countours是一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示

###-----------------------drawContours()函数----------------------------------#
# void drawContours(InputOutputArray image, InputArrayOfArrays contours, int contourIdx, const Scalar& color, int thickness=1, int lineType=8, InputArray hierarchy=noArray(), int maxLevel=INT_MAX, Point offset=Point() )
# image表示目标图像，image为三通道才能显示轮廓
# contours表示输入的轮廓组，每一组轮廓由点vector构成
# contourIdx指明画第几个轮廓，如果该参数为负值，则画全部轮廓
# color为轮廓的颜色
# thickness为轮廓的线宽，如果为负值或CV_FILLED表示填充轮廓内部
# lineType为线型
# 轮廓结构信息
# maxLevel

# img = cv2.imread("./image/num_test.png",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("org_img", img)

# # 获取边缘信息
# contours, hierarchy = cv2.findContours(image=img,  mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# # 创建画布
# canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# # 绘制轮廓
# cv2.drawContours(image=canvas, contours=contours, contourIdx=-1,  color=(0,255,0), thickness=3)

# cv2.imshow("canvas_img", canvas)

#---------------------------------------------------------------------------end
cv2.waitKey()
cv2.destroyAllWindows() 
