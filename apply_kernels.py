import cv2
import numpy as np
import math, matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from kernals import *


path = 'images\\Fig0636(woman_baby_original).tif'
ori_image = cv2.imread(path)                            # reading image,
image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)     # converting color to gray_scale
# print(image)

def convolve(image, kernel):
    (iH, iW), (kH, kW) = image.shape[:2], kernel.shape[:2]
    pad = (kW - 1) // 2
    output = np.zeros((iH, iW), dtype="float32")

    for i in range(pad):                                      # padding of zeros
        image = np.insert(image, 0, 0, axis=1)
        image = np.insert(image, iW+i+1, 0, axis=1)
        h_pad = [[0 for _ in range(iW+2+i*2)]]
        image = np.vstack((h_pad, image))
        image = np.vstack((image, h_pad))

    for y in np.arange(pad, iH + pad):                          # convolving
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))       # rescaling
    return output

i = 2
plt.subplot(231)
plt.imshow(ori_image, 'gray')
plt.title('original image')
for (kernelName, kernel) in kernelBank:
    sub = '23' + str(i)
    print("applying {} kernel".format(kernelName))
    convoleOutput = convolve(image, kernel)
    # opencvOutput = cv2.filter2D(image, -1, kernel)    # inbuilt function

    # cv2.imshow("original", image)
    plt.subplot(sub)
    plt.imshow(convoleOutput, cmap='gray')
    plt.title(kernelName)
    i = i + 1
plt.suptitle('kernel convolution')
plt.show()
# plt.imsave(fname='kernel_convolution', arr='luminance')
