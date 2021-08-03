# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:53:58 2021

@author: sziit
"""

import cv2
import cv2 as cv
import numpy as np

##---------------------------------------------------------------cornerHarris
img = cv2.imread("anime.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.04 )
dst = cv2.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0,0,225]

cv2.imshow("img",img)

##------------------------------------------------------------------------------

# camera = cv2.VideoCapture("./people.mp4")
# while True:
#     (grabbed, frame) = camera.read()

#     hog = cv.HOGDescriptor()
#     hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
#     (rects, weights) = hog.detectMultiScale(
#         frame,
#         winStride = (4 ,4),
#         padding = (8, 8),
#         scale = 1.25,
#         useMeanshiftGrouping = False 
#         )
#     for (x,y,w,h) in rects:
#         cv.rectangle(frame, (x,y), (x+w,y+h),(0,225,0),2)

#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break

#----------------------------------------------------------------------------
cv2.waitKey(0)
# camera.release()
cv2.destroyAllWindows()