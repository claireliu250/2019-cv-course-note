import cv2
import numpy as np

img = cv2.imread('../ImageSet/lenna.jpg')
cv2.imshow('lenna', img)


######## Edge #########
# x轴
edgex = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
print(edgex)

sharp_img = cv2.filter2D(img, -1, kernel=edgex)
# cv2.imshow('edgex_lenna', sharp_img)

# y轴
edgey = np.array([[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]], np.float32)
sharpy_img = cv2.filter2D(img, -1, kernel=edgey)
# cv2.imshow('edgey_lenna_y', sharpy_img)

img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
print(img_gray)

img_harris = cv2.cornerHarris(img_gray, 2, 3, 0.05)    # 2： blockSize: window size; 3: Sobel kernel size
# cv2.imshow('img_harris ', img_harris)

# 没法看原因：1. float类型； 2. img_harris本质上是每个pixel对于Harris函数的响应值
# 没有看的价值
print(img_harris)

# 为了显示清楚
# img_harris = cv2.dilate(img_harris , None)

thres = 0.05 * np.max(img_harris)
img[img_harris > thres] = [0, 0, 255]
# cv2.imshow('img_harris2', img)

########### SIFT ###########
# create sift class
sift = cv2.xfeatures2d.SIFT_create()
# detect SIFT
kp = sift.detect(img,None)   # None for mask
# compute SIFT descriptor
kp,des = sift.compute(img,kp)
print(des.shape)
img_sift= cv2.drawKeypoints(img,kp,outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('lenna_sift.jpg', img_sift)

key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
