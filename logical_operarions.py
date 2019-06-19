import cv2
import math, matplotlib.pyplot as plt


path1 = 'images\\horizontal.tif'
path2 = 'images\\vertical.tif'

img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)
img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]))

anded = cv2.bitwise_and(img1, img2)
ored = cv2.bitwise_or(img1, img2)
noted = cv2.bitwise_not(img2)
ex_ored = cv2.bitwise_xor(img1, img2)

plt.subplot(221)
plt.imshow(anded, cmap='gray')
plt.title('AND')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(ored, cmap='gray')
plt.title('OR')
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(noted, cmap='gray')
plt.title('NOT')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.imshow(ex_ored, cmap='gray')
plt.title('EX-OR')
plt.xticks([])
plt.yticks([])
plt.suptitle('logical operation')

plt.show()
