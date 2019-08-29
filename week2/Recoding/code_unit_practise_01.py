import cv2
import numpy as np

img = cv2.imread('../ImageSet/lenna.jpg')
cv2.imshow('lenna', img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()


g_img = cv2.GaussianBlur(img, (7,7), 5)
cv2.imshow('gaussian_blur_lenna', g_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()


g_img = cv2.GaussianBlur(img, (17,17), 5)
cv2.imshow('gaussian_blur_lenna2', g_img)


g_img = cv2.GaussianBlur(img,(7,7),1)
cv2.imshow('gaussian_blur_lenna3', g_img)


key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()