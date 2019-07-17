import cv2
import random

from matplotlib import pyplot as plt

file_path = "/Users/claireliu/PycharmProjects/firstPythonDome001/lena.png"

img_gray = cv2.imread(file_path, 0)
# cv2.imshow("img", img_gray)

print(img_gray)

# key = cv2.waitKey()
#
# if key == 27:
#     cv2.destroyAllWindows()

# to show image data type. result is : uint8
print(img_gray.dtype)

# to show image shape(#h #w #channel)
print(img_gray.shape)

#read the old
img = cv2.imread(file_path);


#images corp
img_crop = img[0:20, 0:100]
# cv2.imshow('img_crop', img_crop)


# color split
B, G, R = cv2.split(img)
# cv2.imshow('B', B)
# cv2.imshow('G', G)
# cv2.imshow('R', R)




# change color
def random_light_color(img):
    # brightness
    B, G, R = cv2.split(img)

    # random.randint(a, b)，用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b
    b_rand = random.randint(-50, 50)

    print('b_rand:', b_rand)

    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255 # 如果B > lim，刚将B=255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype) # ? 暂时没有看懂这段话
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)

    print('g_rand:', g_rand)

    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255  # 如果G > lim，刚将B=255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)  # ? 暂时没有看懂这段话
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)

    print('r_rand:', r_rand)

    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255  # 如果R > lim，刚将B=255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)  # ? 暂时没有看懂这段话
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_marge = cv2.merge((B, G, R))
    return img_marge


img_random_color = random_light_color(img)

cv2.imshow('img_random_color', img_random_color)

key = cv2.waitKey()

if key == 27:
    cv2.destroyAllWindows()