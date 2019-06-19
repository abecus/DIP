import cv2
import math, matplotlib.pyplot as plt
from kernals import *


path = 'images\\ori.tif'
image = cv2.imread(path, 0)

highpass3 = cv2.filter2D(image, -1, laplacian)
highpass5 = cv2.filter2D(image, -1, laplacian2)
lowpass3 = cv2.filter2D(image, -1, smallBlur)
lowpass5 = cv2.filter2D(image, -1, largeBlur)

plt.subplot(221)
plt.imshow(highpass3, cmap='gray')
plt.title('Highpass 3*3 filter')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(highpass5, cmap='gray')
plt.title('Highpass 5*5 filter')
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(lowpass3, cmap='gray')
plt.title('Lowpass 3*3 filter')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.imshow(lowpass5, cmap='gray')
plt.title('Lowpass 5*5 filter')
plt.xticks([])
plt.yticks([])

plt.suptitle('Noise Removal')
plt.show()
