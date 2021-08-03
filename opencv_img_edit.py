# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 01:47:04 2021

@author: 30679
"""

import numpy as np
import cv2

#---------np.zeros函数
#返回来一个给定形状和类型的用0填充的数组；
# zeros(shape, dtype=float, order=‘C’)
# shape:形状
# dtype:数据类型，可选参数，默认numpy.float64
# order:可选参数，c代表与c语言类似，行优先；F代表列优先
# mask = np.zeros(img.shape[:2], np.uint8)
# mask[100:300, 100:400] = 255

img=np.zeros((512,512,3),np.uint8)   #3 表示RGB三通道  画布 默认0填充（黑色）
img1 = cv2.imread('./image/anime.png')

#-------------------Line
# pt1 直线起始端坐标 (x, y)  pt2 直线结束端坐标 (x, y)   color 颜色  thickness 线宽
cv2.line(img=img,pt1=(100,0), pt2=(0,100),color=(0,0,255),thickness=3)

img = cv2.line(img,(0,0),(300,300),(255,0,0),3)
cv2.imshow('img2',img)

#-------------------Rectangle
# 矩形，矩形的左上角和右下角，图像的右上角绘制一个绿色矩形。
# cv2.rectangle(img,(384,0),(510,128),(0,255,0),-1)
# cv2.imshow('img3',img)

#-------------------Circle
# 圆，中心坐标和半径
# img = cv2.circle(img,(200,200), 100, (0,0,255),1)
# cv2.imshow('img4',img)

#---------------------Ellipse
# cv2.ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, color [, thickness[, lineType[, shift]]])

# img = cv2.ellipse(img,(256,256),(100,50),45,0,360,(255,255,255),4)
# cv2.imshow('img5',img)

#---------------------Polygon
# 将这些点转换为ROWSx1x2形状的数组，其中ROWS是顶点数，它应该是int32类型。

# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# img = cv2.polylines(img,[pts],True,(0,255,255))  # 图像，点集，是否闭合，颜色，线条粗细
# cv2.imshow('img6',img)

#---------------------Text
# cv::Mat& img, // 待绘制的图像
# const string& text, // 待绘制的文字
# cv::Point origin, // 文本框的左下角
# int fontFace, // 字体 (如cv::FONT_HERSHEY_PLAIN)
# double fontScale, // 尺寸因子，值越大文字越大
# cv::Scalar color, // 线条的颜色（RGB）
# int thickness = 1, // 线条宽度
# int lineType = 8, // 线型（4邻域或8邻域，默认8邻域）

# font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
# cv2.putText(img,'OpenCV',(10,400), font, 4,(255,255,255),3)  # cv2.LINE_AA
# cv2.imshow('img7',img)

#----------------------ROI
# line to line, row to row.

# roi = img1[100:150,220:270]
# # img1[0:50,100:150] = roi
# cv2.imshow("image_roi", roi)

#---------------------Img operation
# void add(InputArray src1, InputArray src2, OutputArray dst,InputArray mask=noArray(), int dtype=-1);//dst = src1 + src2
# void subtract(InputArray src1, InputArray src2, OutputArray dst,InputArray mask=noArray(), int dtype=-1);//dst = src1 - src2
# void multiply(InputArray src1, InputArray src2,OutputArray dst, double scale=1, int dtype=-1);//dst = scale*src1*src2
# void divide(InputArray src1, InputArray src2, OutputArray dst,double scale=1, int dtype=-1);//dst = scale*src1/src2
# void divide(double scale, InputArray src2,OutputArray dst, int dtype=-1);//dst = scale/src2
# void scaleAdd(InputArray src1, double alpha, InputArray src2, OutputArray dst);//dst = alpha*src1 + src2
# void addWeighted(InputArray src1, double alpha, InputArray src2,double beta, double gamma, OutputArray dst, int dtype=-1);//dst = alpha*src1 + beta*src2 + gamma
# void sqrt(InputArray src, OutputArray dst);//计算每个矩阵元素的平方根
# void pow(InputArray src, double power, OutputArray dst);//src的power次幂
# void exp(InputArray src, OutputArray dst);//dst = e**src（**表示指数的意思）
# void log(InputArray src, OutputArray dst);//dst = log(abs(src))
# void cvAddWeighted( const CvArr* src1, double alpha,const CvArr* src2, double beta,double gamma, CvArr* dst );
# src1，第一个原数组  alpha，第一个数组元素权重  src2第二个原数组  
# beta，第二个数组元素权重  gamma，图1与图2作和后添加的数值  dst，输出图片

# img3 = cv2.add(img,img1)
# dst = cv2.addWeighted(img,0.7,img1,0.3,0)
# cv2.imshow("image_roi", dst)  

#-----------------------Binary logic operation

# bitwise_and是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，1&1=1，1&0=0，0&1=0，0&0=0
# bitwise_or“或”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“或”操作，1|1=1，1|0=0，0|1=0，0|0=0
# bitwise_xor“异或”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“异或”操作，1^1=0,1^0=1,0^1=1,0^0=0
# bitwise_not“非”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“非”操作，~1=0，~0=1

# bitwiseNOT = cv2.bitwise_not(img)
# cv2.imshow("bitwiseNOT", bitwiseNOT)

# bitwiseAnd = cv2.bitwise_and(img, img1)  
# cv2.imshow("bitwise_and.png", bitwiseAnd)

#--------------------------------------------------------------------------end
cv2.waitKey(0)          
cv2.destroyAllWindows() 







