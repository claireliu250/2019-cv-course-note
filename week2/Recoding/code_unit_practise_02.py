import cv2
import numpy as np

img = cv2.imread('../ImageSet/lenna.jpg')
cv2.imshow('lenna', img)

kernel = cv2.getGaussianKernel(7, 5)
print(kernel)
# [[0.12895603]
#  [0.14251846]
#  [0.15133131]
#  [0.1543884 ]
#  [0.15133131]
#  [0.14251846]
#  [0.12895603]]

g1_img = cv2.GaussianBlur(img,(7,7),5)

# sepFilter2D() 用分解的核函数对图像做卷积。首先，图像的每一行与一维的核kernelX做卷积；然后，运算结果的每一列与一维的核kernelY做卷积
g2_img = cv2.sepFilter2D(img, -1, kernel, kernel) # ori, depth, kernelX, kernelY
# cv2.imshow('g1_blur_lenna', g1_img)
# cv2.imshow('g2_blur_lenna', g2_img)

# 2nd derivative: laplacian （双边缘效果）
# 强制生成一个 float32 类型的数组
kernel_lap = np.array([[0, 1,0], [1,-4, 1],[0,1,0]], np.float32)
print(kernel_lap)

# [[ 0.  1.  0.]
#  [ 1. -4.  1.]
#  [ 0.  1.  0.]]

# filter2D 利用内核实现对图像的卷积运算
lap_img = cv2.filter2D(img, -1, kernel=kernel_lap)
cv2.imshow('lap_lenna', lap_img)


# 应用： 图像锐化 = edge+ori
# app: sharpen
# 图像+edge=更锐利地图像，因为突出边缘
kernel_sharp = np.array([[0, 1, 0], [1, -3, 1], [0, 1, 0]], np.float32)
print('------kernel_sharp---------')
print(kernel_sharp)
# [[ 0.  1.  0.]
#  [ 1. -3.  1.]
#  [ 0.  1.  0.]]

lap_img = cv2.filter2D(img, -1, kernel=kernel_sharp)
cv2.imshow('sharp_lenna', lap_img)


# 这样不对，因为，周围有4个1，中间是-3，虽然有边缘效果，但是周围得1会使得原kernel有滤波效果，使图像模糊；
# 解决：所以取kernel_lap得相反数，再加上原图像，这样突出了中心像素，效果类似于小方差的高斯，所以
#      可以既有边缘效果，又保留图像清晰度
kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
lap_img = cv2.filter2D(img, -1, kernel=kernel_sharp)
cv2.imshow('sharp_lenna2', lap_img)

# 更“凶猛”的边缘效果
# 不仅考虑x-y方向上的梯度，同时考虑了对角线方向上的梯度
kernel_sharp = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
lap_img = cv2.filter2D(img, -1, kernel=kernel_sharp)
cv2.imshow('sharp_lenna3', lap_img)

key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()