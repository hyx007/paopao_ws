#!/usr/bin/env python
#coding:utf-8
#说明：本脚本用来生成测试傅里叶变换以及phase-correlation算法的图片

import cv2
import numpy as np

img_raw = np.ones((1000,1000),dtype=np.uint8)
rows,cols = img_raw.shape

def GeneratePicture():
	#生成原始图像
	for i in range(cols):
		if i > 250 and i < 750:
			img_raw[500,i] = 255
	cv2.imwrite("img_raw.png",img_raw)

	#先将原始图像旋转45度
	M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
	img_r = cv2.warpAffine(img_raw,M,(cols,rows))
	cv2.imwrite("img_r.png",img_r)

	#将旋转过的图像在x,y方向各平移200,300个像素
	M = np.float32([[1,0,200],[0,1,300]])
	img_r_t = cv2.warpAffine(img_r,M,(cols,rows))
	cv2.imwrite("img_r_t.png",img_r_t)

print "图像已生成"

if __name__ == '__main__':
	GeneratePicture()
