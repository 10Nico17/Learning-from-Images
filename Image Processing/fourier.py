
import cv2
import numpy as np
np.seterr(divide = 'ignore') 

image = cv2.imread('graffiti.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


fft_img = np.fft.fft2(img)
fft_img_shift = np.fft.fftshift(fft_img)
fft_img_shift_log = 20 * np.log(fft_img_shift)
fft_img_shift_log = np.abs(fft_img_shift_log)
#cv2.imshow("Image after FFT:", fft_img_shift_log)
cv2.imwrite('Image_FFT.png', fft_img_shift_log) 


# IFT for testing
img_reshifft = np.fft.ifftshift(fft_img_shift)
img_ifft = np.fft.ifft2(img_reshifft).real
img = np.abs(img_ifft)
#cv2.imshow("IFFT unfiltered image:", img)
#cv2.imwrite('Image_after_IFFT_without_filtering.png', img) 


# Hochpassfilter (square or circle in the middle (low frequences blocked))
rows, cols = img.shape
# find the center 
crow = int(rows/2)
ccol = int(cols/2)
# set all mask elemnts to one 
mask = np.ones((rows, cols), np.uint8)
#print('mask: ', mask)
# square mask
size_filter=40
mask[crow - size_filter:crow + size_filter, ccol - size_filter:ccol + size_filter] = 0
filtered_image = fft_img_shift * mask
log_filtered_image = 20 * np.log(filtered_image)
log_filtered_image = np.abs(log_filtered_image)
#cv2.imshow("Image after FFT and filtered:", log_filtered_image)
cv2.imwrite('Image_FFT_HP.png', log_filtered_image)


filtered_img_reshifft = np.fft.ifftshift(filtered_image)
ifft_img_filtered = np.fft.ifft2(filtered_img_reshifft).real
ifft_img_filtered = np.abs(ifft_img_filtered) 
#cv2.imshow("IFFT after filtering:", ifft_img_filtered)
cv2.imwrite('Image_IFFT_HP.png', ifft_img_filtered)


# Lowpass filter
mask = np.zeros((rows, cols), np.uint8)
size_filter=40
mask[crow - size_filter:crow + size_filter, ccol - size_filter:ccol + size_filter] = 1
filtered_image = fft_img_shift * mask
log_filtered_image = 20 * np.log(1+filtered_image)
log_filtered_image = np.abs(log_filtered_image)
cv2.imwrite('Image_FFT_TP.png', log_filtered_image)

filtered_img_reshifft = np.fft.ifftshift(filtered_image)
ifft_img_filtered = np.fft.ifft2(filtered_img_reshifft).real
ifft_img_filtered = np.abs(ifft_img_filtered) 
#cv2.imshow("IFFT after filtering:", ifft_img_filtered)
cv2.imwrite('Image_IFFT_TP.png', ifft_img_filtered)


#cv2.waitKey(0)


