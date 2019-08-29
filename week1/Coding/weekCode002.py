import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

# 伽马校正
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    print('invGamma:', invGamma)
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    print("table before:", table)
    table = np.array(table).astype(image.dtype)
    print("table np after:", table)
    return cv2.LUT(image, table)


file_path = "../ImageSet/pexels-photo-1097456.jpeg"

img_dark = cv2.imread(file_path)

file_path2 = "../ImageSet/lenna.jpg"

img = cv2.imread(file_path2)
cv2.imshow('img', img)
#
img_brighter = adjust_gamma(img_dark, 2)  # 1:no change. 2:brighter. 3:something wrong.
# cv2.imshow('img_brighter', img_brighter)
#
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()


###############################
# histogram 直方图

img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0]*0.5), int(img_brighter.shape[1]*0.5)))

# 使用plt.hist()，进行绘制
#
# plt.hist(img,ravel(),hitsizes,ranges,color=)
#
# img.ravel()将原图像的array数组转成一维的数组
# hitsizes为直方图的灰度级数
# ranges为灰度范围[0,255]
# color是参数，需要使用color=''来指定颜色
plt.hist(img_brighter.flatten(), 256, [0, 256], color='r')
# YUV是一种颜色编码方法
# Y 分量表示颜色的亮度（luminance），单取出 Y 分量就是图像的灰度图；U、V 分量表示颜色色度或者浓度（Chrominance）
# TODO 为什么要先转为YUV?
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
# 图像的直方图是对图像对比度效果上的一种处理，旨在使得图像整体效果均匀，黑与白之间的各个像素级之间的点更均匀一点。 
# 通过这种方法，亮度可以更好地在直方图上分布。这样就可以用于增强局部的对比度而不影响整体的对比度，直方图均衡化通过有效地扩展常用的亮度来实现这种功能。
# 这种方法对于背景和前景都太亮或者太暗的图像非常有用，这种方法尤其是可以带来X光图像中更好的骨骼结构显示以及曝光过度或者曝光不足照片中更好的细节。

img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)   # y: luminance(明亮度), u&v: 色度饱和度
#
# cv2.imshow('Color input image', img_small_brighter)
# cv2.imshow('Histogram equalized', img_output)


##########################
# rotation
# 在opencv中提供了cv2.getRotationMatrix2D函数获得变换矩阵。第一参数指定旋转圆点；第二个参数指定旋转角度；第二个参数指定缩放比例

rows, cols, channel = img_dark.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 40, 1)

# warpAffine 平移变换 TODO 为什么要做这步？
img_ratation = cv2.warpAffine(img_dark, M, (cols, rows))
# cv2.imshow('img_ratation', img_ratation)

print(M)

# M[0][2] = M[1][2] = 0
print(M)

img_ratation2 = cv2.warpAffine(img_dark, M, (cols, rows))
# cv2.imshow('img_ratation2', img_ratation2)

M = cv2.getRotationMatrix2D((cols/2, rows/2), 40, 0.5)

img_rotate = cv2.warpAffine(img_dark, M, (cols, rows))
# cv2.imshow('rotated lenna', img_rotate)




##############################
# Affine Transform
# 图像的旋转加上拉升就是图像仿射变换，仿射变化也是需要一个M矩阵就可以，但是由于仿射变换比较复杂，一般直接找很难找到这个矩阵，opencv提供了根据变换前后三个点的对应关系来自动求解M。这个函数是
# M=cv2.getAffineTransform(pos1,pos2),其中两个位置就是变换前后的对应位置关系。输出的就是仿射矩阵M。然后在使用函数cv2.warpAffine()。
#
# 作者：深思海数_willschang
# 链接：https://www.jianshu.com/p/ef67cacf442c
# 来源：简书
# 简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。

rows, cols, channel = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

print('pts1:', pts1, 'pts2', pts2)

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('affine lenna', dst)




###############################
# 透视　Perspective
# 视角变换，需要一个3*3变换矩阵。在变换前后要保证直线还是直线。
# 构建此矩阵需要在输入图像中找寻 4个点，以及在输出图像中对应的位置。这四个点中的任意三个点不能共线
# 有点类似文字扫描

def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)

    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp
M_warp, img_warp = random_warp(img, img.shape[0], img.shape[1])
cv2.imshow('lenna_warp', img_warp)


key = cv2.waitKey(0)
if key == 27:
    exit()