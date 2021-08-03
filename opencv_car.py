# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:42:54 2021

@author: 30679
"""

import cv2 as cv

vc = cv.VideoCapture(r"./output.flv")
#createBackgroundSubtractorKNN去背景,前一帧即后一帧的背景
BS = cv.createBackgroundSubtractorKNN(detectShadows=True)
while (vc.isOpened()):
    ret, frame = vc.read()
    
    status = 0          #car's number
    
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #前后两帧灰度值相减吧，即移动的物体为有值，其余为0黑色
    fgmask = BS.apply(gray)
    ##中值滤波模板就是用卷积框中像素的中值代替中心值，达到去噪声的目的
    image = cv.medianBlur(fgmask,7)   
    element = cv.getStructuringElement(cv.MORPH_RECT,(3, 3));#创建结构体  
    #    MORPH_ERODE    = 0, //腐蚀    MORPH_DILATE   = 1, //膨胀
    #iterations	叠代; 重复进行
    image2 = cv.morphologyEx(image, cv.MORPH_ERODE,element,iterations=2);
    image3 = cv.morphologyEx(image2, cv.MORPH_DILATE, element, iterations=3)
    
    contours, hierarchy = cv.findContours(image3, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for cnt in contours:       
        x, y, w, h = cv.boundingRect(cnt)      
        #排除人行道的干扰 即轮廓中心点在车道范围内
        if 120<int(y+(h/2))<600 and 0<int(x+(w/2))<1260  :
            #排除小矩形的干扰contourArea：轮廓面积
            if cv.contourArea(cnt) < 100 or w < 50 or h < 20:
                continue
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)        
            status += 1#画出即车数量加一

    cv.putText(frame, str(status), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv.imshow("frame",frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break   
vc.release()
cv.destroyAllWindows()










