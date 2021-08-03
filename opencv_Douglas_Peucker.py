# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:53:56 2021

@author: 30679
"""

#------------------------------------------approxPolyDP函数采用道格拉斯-普克算法

#---------------------------------------------------------------------minMaxLoc
# minMaxLoc(src[, mask]，minVal, maxVal, minLoc, maxLoc) 
# src参数表示输入单通道图像 
# mask参数表示用于选择子数组的可选掩码 
# minVal参数表示返回的最小值，如果不需要，则使用NULL
# maxVal参数表示返回的最大值，如果不需要，则使用NULL
# minLoc参数表示返回的最小位置的指针（在2D情况下）； 
# 如果不需要，则使用NULL。 maxLoc参数表示返回的最大位置的指针（在2D情况下）； 
# 如果不需要，则使用NULL
# 使用函数cvMinMaxLoc(result,&min_val,&max_val,&min_loc,&max_loc,NULL);从result中提取最大值（相似度最高）以及最大值的位置（即在result中该最大值max_val的坐标位置max_loc，即模板滑行时左上角的坐标，类似于图中的坐标（x,y）。）

# 由此得到：rect=cvRect(max_loc.x,max_loc.y,tmp->width,tmp->height);rect表示的矩形区域即是最佳的匹配区域

###--------------------------------------------------approxPolyDP多边形逼近函数
# void approxPolyDP(InputArray curve, OutputArray approxCurve, double epsilon, bool closed)；

# 例如：approxPolyDP(contourMat, approxCurve, 10, true);//找出轮廓的多边形拟合曲线
# 第一个参数 InputArray curve：输入的点集
# 第二个参数OutputArray approxCurve：输出的点集，当前点集是能最小包容指定点集的。画出来即是一个多边形。
# 第三个参数double epsilon：指定的精度，也即是原始曲线与近似曲线之间的最大距离,epsilon越小，折线的形状越“接近”曲线
# 第四个参数bool closed：若为true，则说明近似曲线是闭合的；反之，若为false，则断开。

###-------------------------------------------------------轮廓长度函数arcLength
# arcLength 函数用于计算封闭轮廓的周长或曲线的长度。
# cv2.arcLength(curve, closed)
# curve，输入的轮廓顶点
# closed，用于指示曲线是否封闭
# 返回值，轮廓的周长

###------------------------------------------------------轮廓面积contourArea函数
# double contourArea(InputArray contour, bool oriented = false);
# contour，输入的二维点集（轮廓顶点），可以是 vector 或 Mat 类型。
# oriented，面向区域标识符。有默认值 false。若为 true，该函数返回一个带符号的面积值，正负取决于轮廓的方向（顺时针还是逆时针）。若为 false，表示以绝对值返回。

###---------------------------------------------------------convexHull图像的凸包
#void convexHull(InputArray points,OutputArray hull,bool clockwise =  false, bool returnPoints = true)
# InputArray points: 得到的点集，一般是用图像轮廓函数求得的轮廓点
# OutputArray hull: 输出的是凸包的二维xy点的坐标值，针对每一个轮廓形成的
# bool clockwise = false: 表示凸包的方向，顺时针或者逆时针
# bool returnPoint = true: 表示返回点还是点地址的索引

###---------------------------------------------------------多边形polylines函数
# void cv::polylines  (   
#         Mat &                 mg,作为画布的矩阵
#         const Point *const *  pts,折线顶点数组
#         const int *           npts,折线顶点个数
#         int                   ncontours,待绘制折线数
#         bool                  isClosed,是否是闭合折线(多边形)
#         const Scalar &        color,折线的颜色
#         int     thickness = 1,折线粗细
#         int     lineType = LINE_8,线段类型
#         int     shift = 0 缩放比例(0是不缩放,4是1/4)
#     )
# void cv::polylines  (   InputOutputArray    img,作为画布的矩阵
#         InputArrayOfArrays      pts,折线顶点数组
#         bool    isClosed,是否是闭合折线(多边形)
#         const Scalar &      color,折线的颜色
#         int     thickness = 1,折线粗细
#         int     lineType = LINE_8,线段类型
#         int     shift = 0 缩放比例(0是不缩放,4是1/4)
#     )
#与定义1不同,定义2中 pts 的类型为 InputArrayOfArrays, 而 InputArrayOfArrays 的本源是 vector

import cv2
img = cv2.imread("number_org.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh_old = cv2.inRange(gray, 0, 160)      #阈值化法二值化图像
thresh = cv2.medianBlur(thresh_old,19) 
binary = cv2.medianBlur(thresh,19) 
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# 寻找凸包并绘制凸包（轮廓）
# 寻找物体的凸包并绘制凸包的轮廓
for cnt in contours:
    hull = cv2.convexHull(cnt)
    length = len(hull)
    # 如果凸包点集中的点个数大于5
    if length > 1:
        # 绘制图像凸包的轮廓
        for i in range(length):
            cv2.line(img, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,0,255), 2)

cv2.imshow('line', img)
# ----------------------------------------------------------------------------#
# contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

# length = len(contours) # 轮廓数量
# print (length)
# for i in range(length):
#     cnt = contours[i]
#     epsilon = 0.01 * cv2.arcLength(cnt,True)
#     #print(epsilon)#周长每个轮廓
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     #print(approx)#轮廓点坐标
#     cv2.drawContours(img, approx, -1, (0, 255, 0), 15)
#     cv2.polylines(img, [approx],True, (0, 0, 255), 2)
#     # print (i)//轮廓数量 
# cv2.imshow("approx",img)

#-----------------------------------------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
