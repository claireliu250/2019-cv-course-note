import cv2
import numpy as np
from matplotlib import pyplot as plt

def medianBlur(img, kernel, padding_way):

    # 检测传入的kernel是否为一个合法数组
    if kernel % 2 == 0 or kernel is 1:
        print("kernel must 3,5,7,9....")
        return None

    # 算paddingSize
    paddingSize = kernel // 2

    # 获取图片的通道数
    layerSize = len(img.shape)

    # 获取图片大小
    height, width = img.shape[:2]

    # 多通道处理方式（对每个通道进行递归）
    if layerSize == 3:
        matMutbase = np.zeros_like(img)
        for l in range(matMutbase.shape[2]):
            matMutbase[:,:, l] = medianBlur(img[:,:,l], kernel,padding_way)
        return matMutbase
    # 单通道
    elif layerSize == 2:
        # 创建一个padding矩阵
        matBase = np.zeros((height + paddingSize * 2, width + paddingSize *2), dtype=img.dtype)
        print("create padding:" , matBase)
        # 将原值写入padding矩阵中
        matBase[paddingSize:-paddingSize, paddingSize:-paddingSize] = img
        print("padding after insert default value:", matBase)

        if padding_way is 'ZERO':
            pass
        # 复制模式，即padding的值从原img矩阵相邻的值复制而来。
        elif padding_way is 'REPLICA':
            for r in range(paddingSize):
                matBase[r, paddingSize:-paddingSize] = img[0,:]
                matBase[-(1+r), paddingSize:-paddingSize] = img[-1,:]
                matBase[paddingSize:-paddingSize, r] = img[:,0]
                matBase[paddingSize:-paddingSize, -(1+r)] = img[:, -1]

            print("replica :", matBase)
        else:
            print('padding_way error. must ZERO OR REPLICA.')
            return None

        matOut = np.zeros((height, width), dtype=img.dtype)

        for x in range(height):
            for y in range(width):
                line = matBase[x:x + kernel, y:y+kernel].flatten()
                line = np.sort(line)
                matOut[x, y] = line[(kernel * kernel) // 2]
        return matOut
    else:
        print('image layers error.')
        return None

def main():
    # read img
    img = cv2.imread("../ImageSet/2.png")

    # myself = medianBlur(img, 5, 'REPLICA')
    myself = medianBlur(img, 5, 'ZERO')
    if myself is None:
        return

    opencv = cv2.medianBlur(img, 5)

    # marge
    img = np.hstack((img, myself))
    img = np.hstack((img, opencv))

    plt.imshow(img)
    plt.title('median_blur')
    plt.show()


if __name__ == '__main__':
    main()



