#!/usr/bin/env python
#coding:utf-8
#说明：本脚本用来查看图像傅里叶变换后的在频域中的“振幅谱”，以下统称“频谱图”

import cv2
import numpy as np
import matplotlib.pyplot as plt
img_raw_path = "img_raw.png"
img_r_path = "img_r.png"
img_r_t_path = "img_r_t.png"



def GetFFT(img):
	fft2 = np.fft.fft2(img)
	shift2center = np.fft.fftshift(fft2)
	log_fft2 = np.log(1 + np.abs(fft2))
	log_shift2center = np.log(1 + np.abs(shift2center))
	return log_shift2center

if __name__ == '__main__':
	img_raw = plt.imread(img_raw_path)
	img_r = plt.imread(img_r_path)
	img_r_t = plt.imread(img_r_t_path)

	#画出原始图像的频谱图像
	plt.figure()
	plt.subplot(121),plt.imshow(img_raw,'gray'),plt.title('img_raw')
	img_raw_fft = GetFFT(img_raw)
	plt.subplot(122),plt.imshow(img_raw_fft,'gray'),plt.title('img_raw_fft')
	
	#画出经过旋转后图像的频谱图像
	plt.figure()
	plt.subplot(121),plt.imshow(img_r,'gray'),plt.title('img_r')
	img_r_fft = GetFFT(img_r)
	plt.subplot(122),plt.imshow(img_r_fft,'gray'),plt.title('img_r_fft')

	#画处经过旋转与平移后图像的频谱图像
	plt.figure()
	plt.subplot(121),plt.imshow(img_r_t,'gray'),plt.title('img_r_t')
	img_r_t_fft = GetFFT(img_r_t)
	plt.subplot(122),plt.imshow(img_r_t_fft,'gray'),plt.title('img_r_t_fft')

	plt.show()
