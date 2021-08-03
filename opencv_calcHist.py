# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:58:50 2021

@author: 30679
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./image/yuanshen.png")

# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
# imaes:输入的图像
# channels:选择图像的通道
# mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0
# histSize:使用多少个bin(柱子)，一般为256
# ranges:像素值的范围，一般为[0,255]表示0~255
# hist是256x1的数组，每个值对应于该图像中具有相应像素值的像素数

# def fun1():
#     img = cv2.imread('yuanshen.png',cv2.IMREAD_GRAYSCALE)
#     #bins->图像中分为多少格；range->图像中数字范围
#     plt.hist(img.ravel(), bins=256, range=[0, 256]);
#     plt.show()
 
# def fun2():
#     img = cv2.imread('yuanshen.png',cv2.IMREAD_COLOR)
#     color = ('b', 'g', 'r')
#     for i, col in enumerate(color):
#         histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#         plt.plot(histr, color=col)
#     plt.xlim([0, 256])

#--------------------------------------------------------------------灰度直方图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray, cmap=plt.cm.gray)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])

#--------------------------------------------------------------------颜色直方图
# chans = cv2.split(img)
# colors = ('b', 'g', 'r')

# plt.figure()
# plt.title("’Flattened’ Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")

# for (chan, color) in zip(chans, colors):
#     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#     plt.plot(hist, color = color)
#     plt.xlim([0, 256])

#----------------------or
# for cidx, color in enumerate(['b', 'g', 'r']):  
#     cHist = cv2.calcHist([img], [cidx], None, [256], [0,256])
#     plt.plot(cHist, color=color) # 绘制折线图
# plt.xlim([0, 256]) # 设定画布的范围
# plt.show()

#------------------------------------------绘制BGR彩图的统计直方图--使用mask参数
# plt.subplot作用是把一个绘图区域（可以理解成画布）分成多个小区域，用来绘制多个子图。
# nrows和ncols表示将画布分成（nrows*ncols）个小区域，每个小区域可以单独绘制图形；plot_number表示将图绘制在第plot_number个子区域

# mask = np.zeros(img.shape[:2], np.uint8)
# mask[100:300, 100:400] = 255

# masked_img = cv2.bitwise_and(img[:,:,[2,1,0]], img[:,:,[2,1,0]], mask=mask)

# hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
# hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

#-------RGB变换到BGR
# plt.subplot(221),plt.imshow(img[:,:,[2,1,0]],'gray'),plt.colorbar()
# plt.subplot(222), plt.imshow(mask, 'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0, 256])
# plt.show()

#------------------------------------------------------------------多通道直方图
# 将多维数组转换为一维数组img.ravel()

# Matplotlib有一个绘制直方图的函数：matplotlib.pyplot.hist() 
# plt.hist(img.ravel(),256,[0,256])
# plt.show()

# chans = cv2.split(img)
# colors = ('b', 'g', 'r')

# fig = plt.figure(figsize=(15, 5))
# ax = fig.add_subplot(131)
# hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,
#       [32, 32], [0, 256, 0, 256])
# p = ax.imshow(hist, interpolation = "nearest")
# ax.set_title("2D Color Histogram for G and B")
# plt.colorbar(p)

# ax = fig.add_subplot(132)
# hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
#       [32, 32], [0, 256, 0, 256])
# p = ax.imshow(hist, interpolation = "nearest")
# ax.set_title("2D Color Histogram for G and R")
# plt.colorbar(p)

# ax = fig.add_subplot(133)
# hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,
#   [32, 32], [0, 256, 0, 256])
# p = ax.imshow(hist, interpolation = "nearest")
# ax.set_title("2D Color Histogram for B and R")
# plt.colorbar(p)

# print("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))

# hist = cv2.calcHist([img], [0, 1, 2],
#   None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
# print("3D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))