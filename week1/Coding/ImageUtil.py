import cv2
from matplotlib import pyplot as plt
import random
import numpy as np

# 作业要求：Please combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script.
#    Your code need to be completed in Python/C++ in .py or .cpp file with comments and readme file to indicate how to use.

file_path = "../ImageSet/lena.png"

img = cv2.imread(file_path)

################  images crop (图片裁剪) ###############
img_crop = img[0:20, 0:100]
plt.imshow(img_crop)
plt.title('img_crop')
plt.show()

################ color shift (颜色偏移) ################
B, G, R = cv2.split(img)
plt.imshow(B)
plt.title('B')
plt.show()

plt.imshow(G)
plt.title('G')
plt.show()

plt.imshow(R)
plt.title('R')
plt.show()

################ rotation（旋转） ###################
rows, cols, channel = img.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 40, 1)
# warpAffine 平移变换
img_ratation = cv2.warpAffine(img, M, (cols, rows))

plt.imshow(img_ratation)
plt.title('img_ratation')
plt.show()


################ perspective transform（透视变换） ###################
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

plt.imshow(M_warp)
plt.title('M_warp')
plt.show()

plt.imshow(img_warp)
plt.title('img_warp')
plt.show()

