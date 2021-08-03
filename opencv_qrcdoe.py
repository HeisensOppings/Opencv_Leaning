# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:31:15 2021

@author: sziit
"""
import qrcode
from MyQR import myqr
import os

# =============================================================================
img = qrcode.make('Heisenberg')
img.save('qr.png')

# =============================================================================
# input_data = "heisenberg"
# qr = qrcode.QRCode(
#     version=1,
#     box_size=10,
#     border=5)

# qr.add_data(input_data)
# qr.make(fit=True)

# img = qr.make_image(fill='black', back_color='white')
# img.save('qr.png')

# =============================================================================
# version, level, qr_name = myqr.run(
#   words="https://www.cnblogs.com/Poppings/",     # 可以是字符串，也可以是网址(前面要加http(s)://)
#   version=1,               # 设置容错率为最高
#   level='H',               # 控制纠错水平，范围是L、M、Q、H，从左到右依次升高
#   picture="morty.gif",              # 将二维码和图片合成
#   colorized=True,             # 彩色二维码
#   contrast=1.0,              #用以调节图片的对比度，1.0 表示原始图片，更小的值表示更低对比度，更大反之。默认为1.0
#   brightness=1.0,             #用来调节图片的亮度，其余用法和取值同上
#   save_name="qr.gif",           # 保存文件的名字，格式可以是jpg,png,bmp,gif
#   save_dir=os.getcwd()          #控制位置
# )

















