#!/usr/bin/env python
#coding:utf-8
#说明：本脚本用来测试opencv中集成的phaseCorrelation算法
import cv2
import numpy as np




if __name__ == '__main__':
	img_src = cv2.imread('img_raw.png',0)
	img_dst = cv2.imread("img_r_t.png",0)
	rows,cols = img_src.shape

	polar_src = img_src
	polar_dst = img_dst

	polar_src = cv2.logPolar(img_src,(img_src.shape[0]/2,img_src.shape[1]/2),70,cv2.WARP_FILL_OUTLIERS+cv2.INTER_LINEAR)
	polar_dst = cv2.logPolar(img_dst,(img_dst.shape[0]/2,img_dst.shape[1]/2),70,cv2.WARP_FILL_OUTLIERS+cv2.INTER_LINEAR)
	polar_src = np.float32(polar_src)
	polar_dst = np.float32(polar_dst)
	r = cv2.phaseCorrelate(polar_src,polar_dst)
	yaw = r[0][1] * 180/(img_src.shape[1]/2)
	print "yaw:",yaw
	M = cv2.getRotationMatrix2D((cols/2,rows/2),yaw,1)
	img_r = cv2.warpAffine(img_src,M,(cols,rows))
	img_r = np.float32(img_r)
	img_dst = np.float32(img_dst)
	t = cv2.phaseCorrelate(img_r,img_dst)
	print "translation:",t
