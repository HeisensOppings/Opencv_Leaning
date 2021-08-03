# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 01:19:50 2021

@author: 30679
"""

# 均值滤波	使用模板内所有像素的平均值代替模板中心像素灰度值	易收到噪声的干扰，不能完全消除噪声，只能相对减弱噪声
# 中值滤波	计算模板内所有像素中的中值，并用所计算出来的中值体改模板中心像素的灰度值	对噪声不是那么敏感，能够较好的消除椒盐噪声，但是容易导致图像的不连续性
# 高斯滤波	对图像邻域内像素进行平滑时，邻域内不同位置的像素被赋予不同的权值	对图像进行平滑的同时，同时能够更多的保留图像的总体灰度分布特征

# 平滑处理“（smoothing）也称“模糊处理”（bluring）
# 图像噪声：图像在产生、传输等过程中被其他因素干扰或由于某些原因出现数据丢失，
# 出现的某些像素点明显异于周围像素点的现象，称作图像噪声；

# 图像平滑：为了降低噪声对图像质量带来的影响，对图像进行区域增强的算法，
# “平滑”可以理解为降低噪点与周围正常点差异，抚平像素值的显著跃迁；

#np.hstack()是把矩阵进行行连接---------------------np.vstack()是把矩阵进行列连接

#-----------------------------------------------------------------matchTemplate
# 	void cv::matchTemplate(
# 		cv::InputArray image, // 用于搜索的输入图像, 8U 或 32F, 大小 W-H
# 		cv::InputArray templ, // 用于匹配的模板，和image类型相同， 大小 w-h
# 		cv::OutputArray result, // 匹配结果图像, 类型 32F, 大小 (W-w+1)-(H-h+1)
# 		int method // 用于比较的方法
# 	);
#method:
# CV_TM_SQDIFF 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
# CV_TM_CCORR 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
# CV_TM_CCOEFF 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
# CV_TM_SQDIFF_NORMED 归一化平方差匹配法
# CV_TM_CCORR_NORMED 归一化相关匹配法
# CV_TM_CCOEFF_NORMED 归一化相关系数匹配法

#-------------------------------------------------------2D滤波器cv2.filter2D
# dst=cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])
# src	原图像
# dst	目标图像，与原图像尺寸和通过数相同
# ddepth	目标图像的所需深度
# kernel	卷积核（或相当于相关核），单通道浮点矩阵;如果要将不同的内核应用于不同的通道，请使用拆分将图像拆分为单独的颜色平面，然后单独处理它们。
# anchor	内核的锚点，指示内核中过滤点的相对位置;锚应位于内核中;默认值（-1，-1）表示锚位于内核中心。
# detal	在将它们存储在dst中之前，将可选值添加到已过滤的像素中。类似于偏置。
# borderType	像素外推法，参见BorderTypes

#----------------------------------------------------------------------均值滤波
###函数cv2.blur()是一个通用的2D滤波函数，它的使用需要一个核模板。该滤波函数是单通道运算的，
###函数cv2.boxFilter()如果不想使用归一化模板，那么应该使用cv2.boxFilter(), 
#并且传入参数normalize=False  -1表示与源图像深度相同

#-----------------------------------------------------------------------高斯滤波
#GaussianBlur( InputArray src, OutputArray dst, Size ksize,double sigmaX, double sigmaY = 0,int borderType = BORDER_DEFAULT );
# src	输入图像，图像可以有任意数量的通道，这些通道被处理独立，但深度应为cv_8u、cv_16u、cv_16s、cv_32f或cv_64f。
# dst	输出图像的大小和类型与src相同
# ksize	ksize高斯核大小。ksize.width和ksize.height可以不同，但它们都必须是正数和奇数。或者，它们可以是零，然后从sigma中计算出来。
# sigmaX	高斯核在x方向的标准差
# sigmaY	高斯核在y方向的标准差；如果sigma y为零，则设置为等于sigma x，如果两个sigma都为零，则从ksize.width和ksize.height计算
# borderType	像素外推方法

#----------------------------------------------------------------------双边滤波
#dst = cv.bilateralFilter( src, d, sigmaColor, sigmaSpace[, dst[, borderType]]
# InputArray类型的src，输入图像，即源图像，需要为8位或者浮点型单通道、三通道的图像。
# OutputArray类型的dst，即目标图像，需要和源图片有一样的尺寸和类型。
# int类型的d，表示在过滤过程中每个像素邻域的直径。如果这个值我们设其为非正数，那么OpenCV会从第五个参数sigmaSpace来计算出它来。
# double类型的sigmaColor，颜色空间滤波器的sigma值。这个参数的值越大，就表明该像素邻域内有更宽广的颜色会被混合到一起，产生较大的半相等颜色区域。
# double类型的sigmaSpace坐标空间中滤波器的sigma值，坐标空间的标注方差。他的数值越大，意味着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。当d>0，d指定了邻域大小且与sigmaSpace无关。否则，d正比于sigmaSpace。
# nt类型的borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_DEFAULT。

#------------------------------------------------------------------Sobel算子求导
# Sobel 算子用来计算图像灰度函数的近似梯度。
# Sobel 算子结合了高斯平滑和梯度

# cv2.Sobel(src, ddepth, dx, dy, ksize, scale, delta, borderType)
# src: 输入灰度图像，ddepth：图像深度； dx: x方向上导数因子； dy: Y
# 方向导数因子；ksize: 核大小，必须为奇数，且小于31；

#-----------------------------------------------------------------Canny边缘检测
# cv.Canny(	image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]
# cv2.Canny()第一个参数是输入图像
# 第二和第三个分别是 minVal 和 maxVal
# 低于阈值1的像素点会被认为不是边缘；
# 高于阈值2的像素点会被认为是边缘；
# 在阈值1和阈值2之间的像素点,若与第2步得到的边缘像素点相邻，则被认为是边缘，否则被认为不是边缘。
# 第三个参数设置用来计算图像梯度的 Sobel卷积核的大小，默认值为 3
# 最后一个参数是 L2gradient，它可以用来设定求梯度大小的方程。如果设为 True，就会使用我们上面提到过的方程，否则使用方程：Edge Gradient(G) = |G2x| + |G2y| 代替，默认值为 False

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
img = cv2.imread('./image/anime_same_size.png')

#================================================================matchTemplate
src = cv2.imread("./image/anime.png",0)
#src = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
template = cv2.imread("./image/anime_same_size.png",0)

h,w = template.shape

res = cv2.matchTemplate(src,template,3)
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
top_left = max_loc
buttom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(src, top_left, buttom_right, 255, 2)
cv2.imshow('input image', src)

#----------------------------------------------------------------------filter2D
# kernel = np.ones((5, 5), np.float32)/25
 
# dst = cv2.filter2D(img, -1, kernel)
# htich1 = np.hstack((img, dst))
# cv2.imshow('filter2D', htich1)

#-------------------or
# src = img
# sobel_x = np.array([[-1,0,1],
#                     [-2,0,2],
#                     [-1,0,1]])    #sobel filter

# kernel = np.array((
#     [0.0625, 0.125, 0.0625],
#     [0.125, 0.25, 0.125],
#     [0.0625, 0.125, 0.0625]), dtype="float32")

# dst = cv2.filter2D(src, -1, sobel_x)
# htich = np.hstack((src, dst))
# cv2.imshow('merged_img', htich)

#--------------------------------------------------------------------boxFilter
# blur = cv2.blur(img, (3, 5))  # 模板大小为3*5, 模板的大小是可以设定的
# box = cv2.boxFilter(img, -1, (3, 5))

# htich = np.hstack((img, blur,box))
# cv2.imshow('img_blur_box', htich)

#------------------------------------------------------------中值滤波medianBlur
# for i in range(2000):  # 加入椒盐噪声
#     _x = np.random.randint(0, img.shape[0])
#     _y = np.random.randint(0, img.shape[1])
#     img[_x][_y] = 255
# blur = cv2.medianBlur(img, 3)  
# htich = np.hstack((img, blur))
# cv2.imshow('medianBlur_img', htich)

#------------------------------------------------------------------GaussianBlur
# for i in range(2000):  # 在图像中加入点噪声
#     _x = np.random.randint(0, img.shape[0])
#     _y = np.random.randint(0, img.shape[1])
#     img[_x, _y] = 255
# blur = cv2.GaussianBlur(img, (9, 9), 0)  #5,5）表示的是卷积模板的大小，0表示的是沿x与y方向上的标准差
# htich = np.hstack((img, blur))
# cv2.imshow('GaussianBlur_img', htich)

#----------------------------------------------------------------------双边滤波
# img = cv2.imread('./girl.png')
# # 9表示的是滤波领域直径，后面的两个数字：空间高斯函数标准差，灰度值相似性标准差
# blur = cv2.bilateralFilter(img, 5 , 75, 75)
# htich = np.hstack((img, blur))
# cv2.imshow('bilateralFilter_img', htich)

#-----------------------------------------------------------------------边缘检测
# 使用图像梯度进行锐化

#-------------------------------------------------------------------------Sobel
# x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3) #1, 0表示x方向
# # 由于导数有正负，所有需要16位有符号的数据类型，即cv2.CV_16S
# y = cv2.Sobel(img,cv2.CV_16S,0,1, ksize=3)	#0, 1表示y方向
# absX =  cv2.convertScaleAbs(x)	# 取绝对值，
# absY =  cv2.convertScaleAbs(y)	# 处理完毕后要转回uint8数值类型
# dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0) #结合x和y方向的梯度
# imgSharp = cv2.addWeighted(img, 0.85, dst, 0.15, 0)
# cv2.imshow("img", img)
# cv2.imshow("absX", absX)
# cv2.imshow("absY", absY)
# cv2.imshow("Result", dst)
# cv2.imshow("imgSharp", imgSharp)
#-----------------------------------------------------------------------------#

#-------------------------------------------------------------------------Canny
# lenna = cv2.imread('girl.png')
# # 图像降噪
# img = cv2.GaussianBlur(lenna, (5, 5), 0)
# # Canny边缘检测，50为低阈值low，150为高阈值high
# canny = cv2.Canny(img, 100, 150)
# cv2.imshow("GaussianBlur", img)
# cv2.imshow("canny", canny)

#--------------------------------------------------------------------Carnny动态
# def nothing(x):
#     pass
# lenna = cv2.imread('girl.png')

# res=cv2.resize(lenna,None,fx=1,fy=2,interpolation=cv2.INTER_CUBIC)

# cv2.namedWindow('image')

# cv2.createTrackbar('maxVal','image',0,1000, nothing)
# cv2.createTrackbar('minVal','image',0,1000, nothing)

# while(1):
#     k=cv2.waitKey(1)&0xFF
#     if k==27:
#         break
#     max=cv2.getTrackbarPos('maxVal','image') 
#     min=cv2.getTrackbarPos('minVal','image') 
#     edges = cv2.Canny(lenna,min,max)
#     cv2.imshow('image',edges)
#     if cv2.waitKey(1) == ord("q"):
#         break



#---------------------------------------------------------------------------end
cv2.waitKey(0)
cv2.destroyAllWindows()
























