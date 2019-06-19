import cv2
import math, matplotlib.pyplot as plt


path1 = 'images\\a.tif'; 
path2 = 'images\\z.tif'

img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)
img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]))

add = img1 + img2
plt.subplot(221)
plt.imshow(add, cmap='gray')
plt.title('addition')
plt.xticks([])
plt.yticks([])

sub = img1 - img2
plt.subplot(222)
plt.imshow(sub, cmap='gray')
plt.title('substraction')
plt.xticks([])
plt.yticks([])

mul = img1 * img2
plt.subplot(223)
plt.imshow(mul, cmap='gray')
plt.title('multiplication')
plt.xticks([])
plt.yticks([])

div = img1 / img2
plt.subplot(224)
plt.imshow(div, cmap='gray')
plt.title('division')
plt.xticks([])
plt.yticks([])
plt.suptitle('arithematic operation')
plt.show()

