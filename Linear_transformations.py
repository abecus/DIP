import cv2
import numpy as np
import math, matplotlib.pyplot as plt


path = 'images\\z.tif'
img = cv2.imread(path, 0)

st_scaled = cv2.resize(img, None, fx=2, fy=1)            # streaching by 2
sh_scaled = cv2.resize(img, None, fx=.5, fy=.5)          # shrinking by 2

plt.subplot(231)
plt.imshow(st_scaled, cmap='gray')
plt.title('streaching')
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.imshow(sh_scaled, cmap='gray')
plt.title('shrinking')
plt.xticks([])
plt.yticks([])

rows, cols = img.shape
M = np.float32([[1, 0, 100], [0, 1, 150]])
trans = cv2.warpAffine(img, M, (cols, rows))
plt.subplot(233)
plt.imshow(trans, cmap='gray')
plt.title('translation')
plt.xticks([])
plt.yticks([])

M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
rotation = cv2.warpAffine(img,M,(cols, rows))
plt.subplot(234)
plt.imshow(rotation, cmap='gray')
plt.title('rotation by 90_degree')
plt.xticks([])
plt.yticks([])

M = np.float32([[1, 1, 0],[0, 1, 0 ]])                         # horizontal shear
h_shear = cv2.warpAffine(img, M, (cols, rows))
plt.subplot(235)
plt.imshow(h_shear, cmap='gray')
plt.title('horizontal shear')
plt.xticks([])
plt.yticks([])

M = np.float32([[1, 0, 0],[-1, 1, 0 ]])                        # vertical shear
v_shear = cv2.warpAffine(img, M, (cols, rows))
plt.subplot(236)
plt.imshow(v_shear, cmap='gray')
plt.title('vertical shear')
plt.xticks([])
plt.yticks([])

plt.suptitle('Linear Transformations')
plt.show()
