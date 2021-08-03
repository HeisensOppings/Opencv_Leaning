# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 01:05:42 2021

@author: 30679
"""
'''
This file include:
    Aisle change with function and plit
    Binarization with formula and Thresholding
    Grayscale with function and formula
    
'''

import cv2
import numpy as np
img = cv2.imread('./image/yuanshen.png')

#=============================================================================
# show some img's information
#-----------------------------------------------------------------------------
#img = cv2.imread('yuanshen.png',cv2.IMREAD_COLOR)        # default BGR
#img = cv2.imread('yuanshen.png',cv2.IMREAD_GRAYSCALE)    # gray

#--------------------------------Convert BGR to HLS
# imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# cv2.imshow("output", imgHLS)

# Image的属性，本质上是numpy的多维数组数据格式(ndarray)的属性
# img.shape[:2] 取彩色图片的长、宽。
# img.shape[:3] 则取彩色图片的长、宽、通道。
# img.shape[0]：图像的垂直尺寸（高度）
# img.shape[1]：图像的水平尺寸（宽度）
# img.shape[2]：图像的通道数

px = img [100,100]   # get value with x-y coordinate of pixels
print (px)

img[100,100] = [0,0,0]   # can change pixels value

b = img[50,50,0]      #x-y-R(aisle)
g = img[50,50,1]      #G
r = img[50,50,2]      #R
print(b,g,r)

print("图像对象的类型 {}".format(type(img)))
print(img.shape)
print("图像宽度: {} pixels".format(img.shape[1]))
print("图像高度: {} pixels".format(img.shape[0]))
# print("通道: {}".format(img.shape[2]))


print("图像大小: {}".format(img.size))  #uchar = uint8
print("数据类型: {}".format(img.dtype))

cv2.imshow('image', img)  #显示图像

#=============================================================================
# aisle change & use function
#-----------------------------------------------------------------------------
# def BGRRGB(img):
    
# # cv2.imread() # default as BGR

#   b = img[:, :, 0].copy()
#   g = img[:, :, 1].copy()
#   r = img[:, :, 2].copy()

# # RGB>BGR
#   img[:, :, 0] = r
#   img[:, :, 1] = g
#   img[:, :, 2] = b
#   return img

# or newImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

# img = BGRRGB(img)   #调用函数

# cv2.imshow("result1", img)

#---------------------------------or plit
# b, g, r = cv2.split(img)

# img = cv2.merge((r, g, b))    # merge aisle
# cv2.imshow("result2", img)

#=============================================================================
# Binarization 
#-----------------------------------------------------------------------------
# img_bin = cv2.inRange(img, lowerb=(9, 16, 84), upperb=(255, 251, 255))

# cv2.imshow('img',img_bin)

# img = cv2.imread('yuanshen.png')

#=============================================================================
# Grayscale
#-----------------------------------------------------------------------------
# print("数据类型: {}".format(img.dtype))

# out=0.2126*img [:,:,0] +0.7152*img [:,:,1]+0.0722*img [:,:,2]  #通道代入公式
# out=out.astype(np.uint8)    #转换格式图像由unit8类型，即0～255的整数部分构成。

# cv2.imshow("result_1", out)   # 输出结果图片

#--------------------------------------------------------------------------end
cv2.waitKey(0)
cv2.destroyAllWindows()










   














