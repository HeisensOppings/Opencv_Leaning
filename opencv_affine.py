# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:52:37 2021

@author: 30679
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('./image/yuanshen.png')
img1 = cv2.imread("./image/anime.png")
image = cv2.imread('./image/number_img.png')

#====cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
# src - 输入图像。
# M - 变换矩阵。
# dsize - 输出图像的大小。
# flags - 插值方法的组合（int 类型！）
# borderMode - 边界像素模式（int 类型！）
# borderValue - 边界填充值; 默认为0

#=============================cv2.getRotationMatrix2D(center, angle, scale)函数
# center：图片的旋转中心
# angle：旋转角度
# scale：缩放比例，该例中0.5表示我们缩小一半

#===========================================cv2.Flip (src, dst=None,  flipCode)
# src ------  原始图像矩阵； 　　      
# dst -----   变换后的矩阵；
# flipCode ---- 翻转模式
# flipCode 是旋转类型，0代表x轴旋转，任意正数代表y轴旋转，任意负数代表x和y轴同时旋转

#====================================================================resize缩放
# void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR)  
# src - 原图
# dst - 目标图像。当参数dsize不为0时，dst的大小为size；否则，它的大小需要根据src的大小，参数fx和fy决定。dst的类型（type）和src图像相同
# dsize - 目标图像大小
# 所以，参数dsize和参数(fx, fy)不能够同时为0
# fx - 水平轴上的比例因子。
# fy - 垂直轴上的比例因子。
# 最后一个参数插值方法，是默认值，放大时最好选 INTER_LINEAR ，缩小时最好选 INTER_AREA
# interpolation：这个是指定插值的方式，图像缩放之后，肯定像素要进行重新计算的，就靠这个参数来指定重新计算像素的方式，有以下几种：
#       INTER_NEAREST - 最邻近插值
#       INTER_LINEAR - 双线性插值，如果最后一个参数你不指定，默认使用这种方法
#       INTER_AREA -区域插值 resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
#       INTER_CUBIC - 4x4像素邻域内的双立方插值
#       INTER_LANCZOS4 - 8x8像素邻域内的Lanczos插值

#==================================getPerspectiveTransform获得透视变换矩阵的函数
# CV_EXPORTS_W Mat getPerspectiveTransform( const Point2f src[], const Point2f dst[] );
# CV_EXPORTS_W Mat getPerspectiveTransform( InputArray src, InputArray dst );
# 第一个参数表示输入透视变换前图像四点坐标
# 第二个参数表示输入透视变换后图像四点坐标
# 返回值类型Mat
# 该函数返回透视变换矩阵M大小为3x3

#=======================================================warpPerspective透视变换
# CV_EXPORTS_W void warpPerspective( 
# InputArray src, 
# OutputArray dst,
# InputArray M, 透视变换矩阵(3x3)
# Size dsize, 输出图像大小
# int flags=INTER_LINEAR,插值方法，一般为线性或者最近邻插值
# int borderMode=BORDER_CONSTANT,边缘的处理方法，有默认值一般不用设。
# const Scalar& borderValue=Scalar());填充演示，默认是黑色

#--------------------------------------------warpAffine-and-getRotationMatrix2D
# rows,cols = img.shape[:2]
# # x轴平移200，y轴平移100, 2*3矩阵
# N = np.float32([[1, 0, 200], [0, 1, -100]])
# M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
# M1 = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
# M2 = cv2.getRotationMatrix2D((cols/2,rows/2),60,1)

# img_s = cv2.warpAffine(img, N, (cols, rows), borderValue=(155, 150, 200))
# img_tra = cv2.warpAffine(img,M,(cols,rows))
# img_tra1 = cv2.warpAffine(img,M1,(cols,rows))
# img_tra2 = cv2.warpAffine(img,M2,(cols,rows), borderValue=(155, 100, 155))

# plt.figure(figsize=(8,8))

# plt.subplot(221),plt.imshow(img[:,:,::-1])
# plt.subplot(222),plt.imshow(img_s[:,:,::-1])
# plt.subplot(223),plt.imshow(img_tra[:,:,::-1])
# plt.subplot(224),plt.imshow(img_tra2[:,:,::-1])

# plt.subplots_adjust(top=0.8, bottom=0.08, left=0.10, right=0.95, hspace=0,wspace=0.35)

#--------------------------------------------------------------------------flip
# flip_h = cv2.flip(img, 1)
# flip_v = cv2.flip(img, 0)
# flip_hv = cv2.flip(img, -1)

# plt.subplot(221),plt.imshow(flip_h[:,:,::-1])
# plt.subplot(222),plt.imshow(flip_v[:,:,::-1])
# plt.subplot(223),plt.imshow(flip_hv[:,:,::-1])

#----------------------------------------------------------------------resized
# height, width, channel = img.shape
# new_dimension = (int(0.5 * width), int(0.5 * height))

# # 指定新图片的维度与插值算法（interpolation）
# resized = cv2.resize(img, new_dimension,interpolation = cv2.INTER_AREA)

# cv2.imshow('resize',resized)

#-----------------------------------getPerspectiveTransform-and-warpPerspective
# 坐标 上左右 下左右
# pts1 = np.float32([[391,93],[722,66],[393,323],[737,285]])
# pts2 = np.float32([[0,0],[192,0],[0,108],[192,108]])
# M = cv2.getPerspectiveTransform(pts1, pts2)
# dst = cv2.warpPerspective(img1, M, (192,108))

# cv2.imshow('ins', dst)

#--------------------------------------------------------------------------end
cv2.waitKey(0)
cv2.destroyAllWindows()
