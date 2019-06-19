import cv2
import numpy as np
import math, matplotlib.pyplot as plt


path = 'images\\chalk.tif'
image = cv2.imread(path, 0) 

img_float32 = np.float32(image)
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

mag_spectrum = 20*np.log(np.abs(fshift))

rows, cols = image.shape
crow, ccol = rows//2 , cols//2     # center

# create a mask first, center square is 1, remaining all zeros
LowpassMask = np.zeros((rows, cols, 2), np.uint8)
LowpassMask[crow-30:crow+30, ccol-30:ccol+30] = 1
HighpassMask = 1 - LowpassMask

# apply mask and inverse DFT
fshift1 = dft_shift*LowpassMask
f_ishift1 = np.fft.ifftshift(fshift1)

fshift2 = dft_shift*HighpassMask
f_ishift2 = np.fft.ifftshift(fshift2)
img_back1 = cv2.idft(f_ishift1)
img_back2 = cv2.idft(f_ishift2)
img_back1 = cv2.magnitude(img_back1[:,:,0],img_back1[:,:,1])
img_back2 = cv2.magnitude(img_back2[:,:,0],img_back2[:,:,1])

plt.subplot(221)
plt.imshow(image, cmap = 'gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(mag_spectrum, cmap = 'gray')
plt.title('fourier trans. spectrum')
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(img_back1, cmap = 'gray')
plt.title('lowpass')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.imshow(img_back2, cmap = 'gray')
plt.title('highpass')
plt.xticks([])
plt.yticks([])

plt.suptitle('filtering in frequency domain')
plt.show()
