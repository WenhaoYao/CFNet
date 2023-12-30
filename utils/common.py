import os

import numpy as np
import cv2
from PIL import Image


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.psnr =0
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename):
	img = cv2.imread(filename) # cv2 --> BGR
	img = img[:,:,::-1] / 255.0
#img = Image.open(filename).convert("RGB") / 255.0
	img = np.array(img).astype('float32')
	return img

# def read_img(filename):
# 	flen = len(filename)
# 	i = 0
# 	for i in range(flen):
# 		img = Image.open(filename + str(i) +'.png')
# 		img[i] = img[i]/255.0
# 		img[i] = np.array(img[i]).astype('float32')
# 		return img

def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).astype('float32')

def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).astype('float32')

# # 单张照片
# img = cv2.imread('/home/xjc/PycharmProjects/CBDNet/data/NC12/__192x192__bears.png')
# print(img.shape)
# img = img[:,:,::-1] / 255.0
#
# img = np.array(img).astype('float32')

#重命名
#
# class BatchRename():
#     '''
#     批量重命名文件夹中的图片文件
#
#     '''
#     def __init__(self):
#         self.path = '/home/xjc/PycharmProjects/CBDNet/data/NC12_test'
#
#     def rename(self):
#         filelist = os.listdir(self.path)
#         total_num = len(filelist)
#         i = 0
#         for item in filelist:
#             if item.endswith('.png'):
#                 src = os.path.join(os.path.abspath(self.path), item)
#                 dst = os.path.join(os.path.abspath(self.path), str(i) + '.png')
#                 try:
#                     os.rename(src, dst)
#                     print ('converting %s to %s ...' % (src, dst))
#                     i = i + 1
#                 except:
#                     continue
#         print ('total %d to rename & converted %d png' % (total_num, i))
#
# if __name__ == '__main__':
#     demo = BatchRename()
#     demo.rename()

# images = np.array(images).astype('float32')

# import os
# import numpy as np
# import math
# from PIL import Image
# import time
# # 当中是你的程序
#
# def psnr(img1, img2):
# 	mse = np.mean((img1- img2) ** 2)
# 	if mse < 1.0e-10:
# 		return 100 * 1.0
# 	return 10 * math.log10(255.0 * 255.0 / mse)
#
#
# def mse(img1, img2):
# 	mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
# 	return mse
#
#
# def ssim(y_true, y_pred):
# 	u_true = np.mean(y_true)
# 	u_pred = np.mean(y_pred)
# 	var_true = np.var(y_true)
# 	var_pred = np.var(y_pred)
# 	std_true = np.sqrt(var_true)
# 	std_pred = np.sqrt(var_pred)
# 	c1 = np.square(0.01 * 7)
# 	c2 = np.square(0.03 * 7)
# 	ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
# 	denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
# 	return ssim / denom
#
#
# path1 = '/home/xjc/PycharmProjects/CBDNet/data/test/test_nc_out1/'  # 指定输出结果文件夹
# path2 = '/home/xjc/PycharmProjects/CBDNet/data/test/test_nc_out/'  # 指定原图文件夹
#
# f_nums = len(os.listdir(path1))
# list_psnr = []
# list_ssim = []
# list_mse = []
# for i in range(f_nums):
# 	img_a = Image.open(path1 + str(i)+'.png')
# 	img_b = Image.open(path2 + str(i)+'.png')
# 	img_a = np.array(img_a)
# 	img_b = np.array(img_b)
# 	psnr_num = psnr(img_a, img_b)
# 	ssim_num = ssim(img_a, img_b)
# 	mse_num = mse(img_a, img_b)
# 	list_ssim.append(ssim_num)
# 	list_psnr.append(psnr_num)
# 	list_mse.append(mse_num)
# print("平均PSNR:", np.mean(list_psnr),list_psnr)
# print("平均SSIM:", np.mean(list_ssim),list_ssim)
# print("平均MSE:", np.mean(list_mse),list_mse)

