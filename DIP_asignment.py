import cv2
import numpy as np
import math, matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from PIL import Image

# electron opposess electric field hence as momrntum increases its resistance to electric field increases

# cv2.namedWindow('a', cv2.WINDOW_AUTOSIZE)
# s = os.chdir('C:\Users\ABDUL BASID\Desktop\dtu\scholarship')

# im = cv2.imread('48000inc.jpg', 1)
# print(len(im))
# c = cv2.VideoCapture(0)
# while True:
#     # c = cv2.VideoCapture(0)
#     tf, frame = c.read(cv2.IMREAD_GRAYSCALE)
#     cv2.imshow('capture', frame)
#     # print(frame)
#
#     if cv2.waitKey(0):
#         break

# cv2 .release()



# path = 'images\\kuku.png'
# ori_image = cv2.imread(path)                            # reading image,
# image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)     # converting color to gray_scale
# print(image)

# def convolve(image, kernel):
#     (iH, iW), (kH, kW) = image.shape[:2], kernel.shape[:2]
#     pad = (kW - 1) // 2
#     output = np.zeros((iH, iW), dtype="float32")

#     for i in range(pad):                                      # padding of zeros
#         image = np.insert(image, 0, 0, axis=1)
#         image = np.insert(image, iW+i+1, 0, axis=1)
#         h_pad = [[0 for _ in range(iW+2+i*2)]]
#         image = np.vstack((h_pad, image))
#         image = np.vstack((image, h_pad))

#     for y in np.arange(pad, iH + pad):                          # convolving
#         for x in np.arange(pad, iW + pad):
#             roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
#             k = (roi * kernel).sum()
#             output[y - pad, x - pad] = k

#     output = rescale_intensity(output, in_range=(0, 255))       # rescaling
#     return output


# i = 2
# plt.subplot(231)
# plt.imshow(ori_image, 'gray')
# plt.title('original image')
# for (kernelName, kernel) in kernelBank:
#     sub = '23' + str(i)
#     print("applying {} kernel".format(kernelName))
#     convoleOutput = convolve(image, kernel)
#     # opencvOutput = cv2.filter2D(image, -1, kernel)

#     # cv2.imshow("original", image)
#     plt.subplot(sub)
#     plt.imshow(convoleOutput, cmap='gray')
#     plt.title(kernelName)
#     i = i + 1
# plt.suptitle('kernel convolution')
# plt.show()
# plt.imsave(fname='kernel_convolution', arr='luminance')


# #  02
#
# path = 'E:\\dtu\\5th sem\\dip\\a.tif'
# image = cv2.imread(path, 0)

# negative = 255 - image
# plt.subplot(221); plt.imshow(negative, cmap='gray'); plt.title('negative')
# plt.xticks([]), plt.yticks([])
# log_trans = np.uint8(np.log1p(image)*.5)
# log_trans = cv2.threshold(log_trans, 1, 255, cv2.THRESH_BINARY)[1]
# plt.subplot(222); plt.imshow(log_trans, cmap='gray'); plt.title('log_transformation')
# plt.xticks([]), plt.yticks([])

# pl_trans = np.power(image, 2)
# plt.subplot(223); plt.imshow(pl_trans, cmap='gray'); plt.title('power_law_transformation')
# plt.xticks([]), plt.yticks([])

# min = np.amin(image); max = np.amax((image))
# for i in range(len(image)):
#     for j in range(len(image[i])):  image[i][j] = ((image[i][j] - min)/ (max - min)) * 255
# plt.subplot(224); plt.imshow(image, cmap='gray'); plt.title('contrast_streching')
# plt.xticks([]), plt.yticks([])
# plt.suptitle('image inhancement')
# plt.show()

#03
# path1 = 'E:\\dtu\\5th sem\\dip\\a.tif'; path2 = 'E:\\dtu\\5th sem\\dip\\z.tif'
# img1 = cv2.imread(path1, 0);    img2 = cv2.imread(path2, 0)
# img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]))
#
# add = img1 + img2
# plt.subplot(221); plt.imshow(add, cmap='gray'); plt.title('addition')
# plt.xticks([]), plt.yticks([])
#
# sub = img1 - img2
# plt.subplot(222); plt.imshow(sub, cmap='gray'); plt.title('substraction')
# plt.xticks([]), plt.yticks([])
#
# mul = img1 * img2
# plt.subplot(223); plt.imshow(mul, cmap='gray'); plt.title('multiplication')
# plt.xticks([]), plt.yticks([])
#
# div = img1 / img2
# plt.subplot(224); plt.imshow(div, cmap='gray'); plt.title('division')
# plt.xticks([]), plt.yticks([])
# plt.suptitle('arithematic operation')
# plt.show()


#04
# path1 = 'E:\\dtu\\5th sem\\dip\\a.tif'; path2 = 'E:\\dtu\\5th sem\\dip\\z.tif'
# img1 = cv2.imread(path1, 0);    img2 = cv2.imread(path2, 0)
# img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]))
#
# anded = cv2.bitwise_and(img1, img2)
# ored = cv2.bitwise_or(img1, img2)
# noted = cv2.bitwise_not(img2)
# ex_ored = cv2.bitwise_xor(img1, img2)
#
# plt.subplot(221); plt.imshow(anded, cmap='gray'); plt.title('AND')
# plt.xticks([]), plt.yticks([])
# plt.subplot(222); plt.imshow(ored, cmap='gray'); plt.title('OR')
# plt.xticks([]), plt.yticks([])
# plt.subplot(223); plt.imshow(noted, cmap='gray'); plt.title('NOT')
# plt.xticks([]), plt.yticks([])
# plt.subplot(224); plt.imshow(ex_ored, cmap='gray'); plt.title('EX-OR')
# plt.xticks([]), plt.yticks([])
# plt.suptitle('logical operation')
# plt.show()

# 05
path = 'E:\\dtu\\5th sem\\dip\\z.tif'
image = cv2.imread(path, 0)
#
# st_scaled = cv2.resize(img, None, fx=2, fy=1)            # streaching by 2
# sh_scaled = cv2.resize(img, None, fx=.5, fy=.5)          # shrinking by 2
# plt.subplot(231); plt.imshow(st_scaled, cmap='gray'); plt.title('streaching')
# plt.xticks([]), plt.yticks([])
# plt.subplot(232); plt.imshow(sh_scaled, cmap='gray'); plt.title('shrinking')
# plt.xticks([]), plt.yticks([])
#
# rows,cols = img.shape
# M = np.float32([[1, 0, 100], [0, 1, 150]])
# trans = cv2.warpAffine(img, M, (cols, rows))
# plt.subplot(233); plt.imshow(trans, cmap='gray'); plt.title('translation')
# plt.xticks([]), plt.yticks([])
#
# M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
# rotation = cv2.warpAffine(img,M,(cols, rows))
# plt.subplot(234); plt.imshow(rotation, cmap='gray'); plt.title('rotation by 90_degree')
# plt.xticks([]), plt.yticks([])
#
# M = np.float32([[1, 1, 0],[0, 1, 0 ]])                         # horizontal shear
# h_shear = cv2.warpAffine(img, M, (cols, rows))
# plt.subplot(235); plt.imshow(h_shear, cmap='gray'); plt.title('horizontal shear')
# plt.xticks([]), plt.yticks([])
#
# M = np.float32([[1, 0, 0],[-1, 1, 0 ]])                        # vertical shear
# v_shear = cv2.warpAffine(img, M, (cols, rows))
# plt.subplot(236); plt.imshow(v_shear, cmap='gray'); plt.title('vertical shear')
# plt.xticks([]), plt.yticks([])
# plt.suptitle('geometrical transformations')
# plt.show()


# 06
# img = cv2.imread(path, 0)
# plt.subplot(221); plt.imshow(img, cmap='gray'); plt.title('Original image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(222); plt.hist(img.flatten(),256,[0,256]); plt.xlim([0,256]); plt.title('Original Histogram')
# plt.xticks([]), plt.yticks([])
#
# equ = cv2.equalizeHist(img)
# plt.subplot(223); plt.imshow(equ, cmap='gray'); plt.title('equalized image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(224); plt.hist(equ.flatten(),256,[0,256]); plt.xlim([0,256]); plt.title('equalized Histogram')
# plt.xticks([]), plt.yticks([])
# plt.suptitle('Histogram Equalisation')
# plt.show()


# 07
# gauss = cv2.GaussianBlur(image, (5, 5), 0)
# median = cv2.medianBlur(image, 5)
# weiner = cv2.bilateralFilter(image, 7, 1,1)
#
# plt.subplot(221); plt.imshow(image, cmap='gray'); plt.title('Original image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(222); plt.imshow(gauss, cmap='gray'); plt.title('Gaussian filter')
# plt.xticks([]), plt.yticks([])
# plt.subplot(223); plt.imshow(median, cmap='gray'); plt.title('Median filtetr')
# plt.xticks([]), plt.yticks([])
# plt.subplot(224); plt.imshow(weiner, cmap='gray'); plt.title('Weiner filter')
# plt.xticks([]), plt.yticks([])
# plt.suptitle('Noise Reduction')
# plt.show()

#08

path = 'E:\\dtu\\5th sem\\dip\\ori.tif'
image = cv2.imread(path, 0)

# highpass3 = cv2.filter2D(image, -1, laplacian)
# highpass5 = cv2.filter2D(image, -1, laplacian2)
# lowpass3 = cv2.filter2D(image, -1, smallBlur)
# lowpass5 = cv2.filter2D(image, -1, largeBlur)
#
# plt.subplot(221); plt.imshow(highpass3, cmap='gray'); plt.title('Highpass 3*3 filter')
# plt.xticks([]), plt.yticks([])
# plt.subplot(222); plt.imshow(highpass5, cmap='gray'); plt.title('Highpass 5*5 filter')
# plt.xticks([]), plt.yticks([])
# plt.subplot(223); plt.imshow(lowpass3, cmap='gray'); plt.title('Lowpass 3*3 filter')
# plt.xticks([]), plt.yticks([])
# plt.subplot(224); plt.imshow(lowpass5, cmap='gray'); plt.title('Lowpass 5*5 filter')
# plt.xticks([]), plt.yticks([])
# plt.suptitle('Noise Removal')
# plt.show()

# 09
# img_float32 = np.float32(image)
# dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# f = np.fft.fft2(image)
# fshift = np.fft.fftshift(f)
# mag_spectrum = 20*np.log(np.abs(fshift))
#
# rows, cols = image.shape
# crow, ccol = rows//2 , cols//2     # center
#
# # create a mask first, center square is 1, remaining all zeros
# LowpassMask = np.zeros((rows, cols, 2), np.uint8)
# LowpassMask[crow-30:crow+30, ccol-30:ccol+30] = 1
# HighpassMask = 1 - LowpassMask
#
# # apply mask and inverse DFT
# fshift1 = dft_shift*LowpassMask
# f_ishift1 = np.fft.ifftshift(fshift1)
#
# fshift2 = dft_shift*HighpassMask
# f_ishift2 = np.fft.ifftshift(fshift2)
# img_back1 = cv2.idft(f_ishift1)
# img_back2 = cv2.idft(f_ishift2)
# img_back1 = cv2.magnitude(img_back1[:,:,0],img_back1[:,:,1])
# img_back2 = cv2.magnitude(img_back2[:,:,0],img_back2[:,:,1])
#
# plt.subplot(221),plt.imshow(image, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(222),plt.imshow(mag_spectrum, cmap = 'gray')
# plt.title('fourier trans. spectrum'), plt.xticks([]), plt.yticks([])
# plt.subplot(223),plt.imshow(img_back1, cmap = 'gray')
# plt.title('lowpass'), plt.xticks([]), plt.yticks([])
# plt.subplot(224),plt.imshow(img_back2, cmap = 'gray')
# plt.title('highpass'), plt.xticks([]), plt.yticks([])
# plt.suptitle('filtering in frequency domain')
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()