import cv2
import math, matplotlib.pyplot as plt

path = 'images\\circuit.tif'
image = cv2.imread(path, 0)

gauss = cv2.GaussianBlur(image, (5, 5), 0)
median = cv2.medianBlur(image, 5)
weiner = cv2.bilateralFilter(image, 7, 1,1)

plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(gauss, cmap='gray')
plt.title('Gaussian filter')
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(median, cmap='gray')
plt.title('Median filtetr')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.imshow(weiner, cmap='gray')
plt.title('Weiner filter')
plt.xticks([])
plt.yticks([])

plt.suptitle('Noise Reduction')
plt.show()
