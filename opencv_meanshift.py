# -*- coding: utf-8 -*-

#---------------------------------------------------------------------meanshift

import numpy as np
import cv2 as cv
cap = cv.VideoCapture('./output.flv')
ret, frame = cap.read()
r, h, c, w = 200, 40, 455, 100 # simply hardcoded the values
track_window = (c,r,w,h)
roi = frame[r:r+h, c:c+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv.imshow('img2',img2)
        k = cv.waitKey(60) & 0xff
        if k == ord("q"):
            break
    else:
        break
cv.destroyAllWindows()
cap.release()