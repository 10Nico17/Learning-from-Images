import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Load image and convert to gray and floating point
img = cv2.imread('./images/graffiti.png')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

gray = np.float32(gray)

# Define sobel filter and use cv2.filter2D to filter the grayscale image
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Compute G_xx, G_yy, G_xy and sum over all G_xx etc. 3x3 neighbors to compute
# entries of the matrix M = \sum_{3x3} [ G_xx Gxy; Gxy Gyy ]
# Note1: this results again in 3 images sumGxx, sumGyy, sumGxy
# Hint: to sum the neighbor values you can again use cv2.filter2D to do this efficiently
sumGxx = cv2.filter2D(sobel_x * sobel_x, -1, np.ones((3, 3)))
sumGyy = cv2.filter2D(sobel_y * sobel_y, -1, np.ones((3, 3)))
sumGxy = cv2.filter2D(sobel_x * sobel_y, -1, np.ones((3, 3)))

det_M = (sumGxx * sumGyy) - (sumGxy ** 2)
trace_M = sumGxx + sumGyy

# Define parameter
k = 0.04
threshold = 0.01

harris = det_M - k * (trace_M ** 2)

harris_thres = np.zeros(harris.shape)
harris_thres[harris > threshold * harris.max()] = 255

# The OpenCV implementation looks like this - please do not change
harris_cv = cv2.cornerHarris(gray,3,3,k)

# intialize in black - set pixels with corners in with
harris_cv_thres = np.zeros(harris_cv.shape)
harris_cv_thres[harris_cv>threshold*harris_cv.max()]=[255]

# just for debugging to create such an image as seen
# in the assignment figure.
img[harris>threshold*harris.max()]=[255,0,0]


cv2.imwrite("Harris_own.png", harris_thres)
cv2.imwrite("Harris_cv.png", harris_cv_thres)
cv2.imwrite("Image_with_Harris.png", img)

cv2.imshow('Harris Corner (Own)', harris_thres)
cv2.imshow('Harris Corner (OpenCV)', harris_cv_thres)
cv2.imshow('Harris Corner Grafity', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# please leave this - adjust variable name if desired
print("====================================")
print("DIFF:", np.sum(np.absolute(harris_thres - harris_cv_thres)))
print("====================================")


'''
cv2.namedWindow('Interactive Systems: Harris Corner')
while True:
     ch = cv2.waitKey(0)
     if ch == 27:
         break
#
     cv2.imshow('harris',harris_thres)
     cv2.imshow('harris_cv',harris_cv_thres)
'''