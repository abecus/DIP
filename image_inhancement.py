import cv2
import numpy as np
import math, matplotlib.pyplot as plt


path = 'images\\a.tif'
image = cv2.imread(path, 0)

negative = 255 - image
plt.subplot(221)
plt.imshow(negative, cmap='gray')
plt.title('negative')
plt.xticks([]), plt.yticks([])

log_trans = np.uint8(np.log1p(image)*.5)
log_trans = cv2.threshold(log_trans, 1, 255, cv2.THRESH_BINARY)[1]
plt.subplot(222)
plt.imshow(log_trans, cmap='gray')
plt.title('log_transformation')
plt.xticks([]), plt.yticks([])

pl_trans = np.power(image, 2)
plt.subplot(223)
plt.imshow(pl_trans, cmap='gray')
plt.title('power_law_transformation')
plt.xticks([]), plt.yticks([])

min = np.amin(image)
max = np.amax((image))

for i in range(len(image)):
    for j in range(len(image[i])):  
        image[i][j] = ((image[i][j] - min)/ (max - min)) * 255

plt.subplot(224)
plt.imshow(image, cmap='gray')
plt.title('contrast_streching')
plt.xticks([]), plt.yticks([])
plt.suptitle('image inhancement')
plt.show()
