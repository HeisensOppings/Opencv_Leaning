# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 01:05:42 2021

@author: 30679
"""

#=============================================================================
# Opencv_Camera(Use app of camera or ipCamera show in HighGUI)
#-----------------------------------------------------------------------------
import cv2

url = 'http://admin:admin@192.168.1.108:8081/'                
# cap = cv2.VideoCapture(url)                       # input the url
cap = cv2.VideoCapture("./image/BadApple.mp4")      # or local mp4
# cap = cv2.VideoCapture(0)                         # or use windowns camcra(Instantiating objects)

img_count = 1                                       # save img name

# set camera width 1920
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# set camera high+ 1080
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while (cap.isOpened()):                    
    
    ret, frame = cap.read()                         # if read success, ret=True
    
    #our operation on the frame come here
    #frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
    # Filp horizontal flip flipCode = 1
    # Flip vertically flipCode = 0
    # Flip sametime   flipCode = -1
    # frame = cv2.flip(frame, -1)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):           # key 'q' to quit
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):         # ket 'c' to save
        cv2.imwrite("{}.png".format(img_count), frame)   # set img name as img_count.png
        print("img name save as {}.png".format(img_count))
        img_count += 1                              # img name +1

    
# When everything done, release the capture  
cap.release()
cv2.destroyAllWindows()
#=============================================================================


