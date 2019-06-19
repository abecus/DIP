import cv2
import numpy as np
import math, matplotlib.pyplot as plt


path = 'images\\z.tif'
img = cv2.imread(path, 0)

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.hist(img.flatten(),256,[0,256])
plt.xlim([0,256])
plt.title('Original Histogram')
plt.xticks([])
plt.yticks([])

equ = cv2.equalizeHist(img)
plt.subplot(223)
plt.imshow(equ, cmap='gray')
plt.title('equalized image')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.hist(equ.flatten(),256,[0,256])
plt.xlim([0,256])
plt.title('equalized Histogram')
plt.xticks([])
plt.yticks([])

plt.suptitle('Histogram Equalisation')
plt.show()
