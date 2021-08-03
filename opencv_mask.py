# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:26:45 2021

@author: 30679
"""
#-------img.shape[:2] 取彩色图片的长、宽。
# 如果img.shape[:3] 则取彩色图片的长、宽、通道。
# 关于img.shape[0]、[1]、[2]
# img.shape[0]：图像的垂直尺寸（高度）
# img.shape[1]：图像的水平尺寸（宽度）
# img.shape[2]：图像的通道数
# 在矩阵中，[0]就表示行数，[1]则表示列数

import cv2    
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('./image/yuanshen.png')    
img1 = cv2.imread('./image/anime.png')
img2 = cv2.imread('./image/yuanshen.png')

# print("图像宽度: {} pixels".format(img.shape[1]))
# print("图像高度: {} pixels".format(img.shape[0]))

# # 创建掩膜图像
# #------.shape[]函数可以快速读取矩阵的形状，例如使用shape[0]读取矩阵第一维度的长度
# #------np.zeros函数返回来一个给定形状和类型的用0填充的数组
# #------np.shape可以获取矩阵的形状,例如二维数组的行列,获取的结果是一个元组
# mask_img=np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)   # mask大小
# # mask_img[50:200,100:400]=255  #兴趣区域赋值白色
# mask_img = cv2.circle(mask_img, (200, 100), 50, (255, 255, 255), -1)

# # add means &&
# mask_image=cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask_img)

# cv2.imshow("mask_img", mask_img)     

# cv2.imshow("mask_image", mask_image)    

#--------------------------------------------------------mask掩膜
rows,cols,channels = img2.shape
# 选择区域
roi = img1[0:rows,0:cols]
# plt.imshow(roi[...,::-1])

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# THRESH_BINARY=0，THRESH_BINARY_INV,THRESH_TRUNC,THRESH_TOZERO,THRESH_TOZERO_INV,THRESH_OTSU,THRESH_TRIANGLE,THRESH_MASK
# 二值化函数threshold(src, thresh, maxval, type[, dst]),thresh:阈值，maxval：最大值，type：阈值类型
# ret:暂时就认为是设定的thresh阈值，mask：二值化的图像
ret,mask = cv2.threshold(img2gray,175,256,cv2.THRESH_BINARY)

plt.imshow(mask,cmap='gray')

mask_inv = cv2.bitwise_not(mask)
plt.imshow(mask_inv,cmap='gray')

img1_bg = cv2.bitwise_and(roi,roi,mask=mask)
plt.imshow(img1_bg[...,::-1])

img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
plt.imshow(img2_fg[...,::-1])

dst = cv2.add(img1_bg,img2_fg)
plt.imshow(dst[...,::-1])

img1[0:rows,0:cols] = dst
plt.imshow(img1[...,::-1])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()  