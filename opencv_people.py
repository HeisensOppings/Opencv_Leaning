# -*- coding: utf-8 -*-
"""
Created on Mon May 31 00:12:25 2021

@author: 30679
"""



# -*- coding: utf-8 -*-
import cv2
import time
import threading
#-----------------------------------------------------------------------多线程
#接收摄影机串流影像，采用多线程的方式，降低缓冲区栈图帧的问题。
# class ipcamCapture:
#     def __init__(self, URL):
#         self.Frame = []
#         self.status = False
#         self.isstop = False
# 		
#  	# 摄影机连接。
#         self.capture = cv2.VideoCapture(URL)

#     def start(self):
#  	# 把程序放进子线程，daemon=True 表示该线程会随着主线程关闭而关闭。
#         print('ipcam started!')
#         threading.Thread(target=self.queryframe, daemon=True, args=()).start()

#     def stop(self):
#  	# 记得要设计停止无限循环的开关。
#         self.isstop = True
#         print('ipcam stopped!')
   
#     def getframe(self):
#  	# 当有需要影像时，再回传最新的影像。
#         return self.Frame

#     def queryframe(self):
#         while (not self.isstop):
#             self.status, self.Frame = self.capture.read()

#         self.capture.release()

# url = 'http://admin:admin@192.168.1.106:8081/'

# # 连接摄影机
# ipcam = ipcamCapture(0)
# # 启动子线程
# ipcam.start()
# # 暂停1秒，确保影像已经填充
# time.sleep(2)

import cv2 as cv                # 导入库文件   
from bypy import ByPy
import time
#-----------------------------------------------------------------------百度api
class BaiduNetdisk(object):     # 创建百度api的类
    def __init__(self):
        self.bp = ByPy()
    def upload(self,filepath):  # 上传文件路径文件路径
        self.bp.upload(filepath)
        
bp = BaiduNetdisk()             # 实例化BaiduNetdisk类
#----------------------------------------------------------------------检测模型
hog = cv.HOGDescriptor()  # hog特征描述
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector()) # 创建SVM检测器

# url = 'http://admin:admin@192.168.1.106:8081/'
# cap = cv.VideoCapture(url)
cap = cv.VideoCapture('four.mp4')  # 创建一个video capture的实例，读进视频
while True:
    ret, frame = cap.read()   # 逐帧捕获视频画面
    # frame = ipcam.getframe()  
    
    # 返回行人轮廓左上点坐标 长 宽 rects
    (rects, weights) = hog.detectMultiScale(frame, 
                                            winStride=(4, 4),
                                            padding=(8, 8),
                                            scale=1.4,
                                            useMeanshiftGrouping=False)
    
    for (x, y, w, h) in rects:  # 获取轮廓的坐标，以及长宽
        # 原始图片；2坐标点；3.矩形宽高 4.颜色值(RGB)；5.线框    
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        # 获取当前系统时间
        time_real = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(time.time()))
        roi = frame[y:y + h,x:x + w]  # #截取y行到y + h行，列为x到x + w列的整块区域     
        # 将获取到的时间变量设为到每一帧的保存名子
        cv.imwrite(r'./' + str(time_real) + '.jpg', roi)
        # 调用百度网盘api将每一帧发送到网盘中
        bp.upload(r'./' + str(time_real) + '.jpg')
    # 输出视频流
    cv.imshow("hog-people", frame)
    
    k = cv.waitKey(30) & 0xff # 按下Esc退出
    if k == 27:
        break

cap.release()           # 释放摄像头
cv.destroyAllWindows()  # 关闭窗口  



#==========================================================================人脸
# image = cv2.imread('face7.jpg')
# #加载已经训练好的OpenCV人脸检测器
# face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# faces = face_model.detectMultiScale(gray) #多尺度人脸检测
# #用方框标记检测出的人脸

# for (x, y, w, h) in faces:
# cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#=======================================================================百度api
# from bypy import ByPy

# class BaiduNetdisk(object):
#     def __init__(self):
#         self.bp = ByPy()
#     def upload(self):
#         filepath = r'G:\opencv\project\six\2.png'
#         self.bp.upload(filepath)
# if __name__ == '__main__':
#     bp = BaiduNetdisk()
#     bp.upload()



