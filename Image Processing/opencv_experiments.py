# Learning from Images: Assigment 1 

import cv2 as cv
import numpy as np
import sys

# Exercise1:

img = cv.imread("graffiti.png")
if img is None:
 sys.exit("Could not read the image.")

mode = 0
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

while(True):
    
    ch = cv.waitKey(1) & 0xFF  

    if ch == ord('0'):
        mode = 0
    if ch == ord('1'):
        mode = 1
    elif ch == ord('2'):
        mode = 2
    elif ch == ord('3'):
        mode = 3
    if ch == ord('q'):
        break

    if mode == 0:

        # way 1
        
        rows_rgb, cols_rgb, channels = img.shape
        rows_gray, cols_gray = gray_img.shape
        rows_comb = max(rows_rgb, rows_gray)
        cols_comb = cols_rgb + cols_gray
        comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)
        comb[:rows_rgb, :cols_rgb] = img
        comb[:rows_gray, cols_rgb:] = gray_img[:, :, None]
        #print('Shape final image: ', comb.shape)
        cv.imshow('Exercise 1', comb)
        cv.setWindowTitle('Exercise 1', 'Exercise 1.1: Concenate image')         

        '''
        # way 2
        rows, cols = img.shape[:2]
        final_image = np.zeros((rows, cols * 2, 3), dtype=np.uint8)
        final_image[:, :cols, 0] = gray_img  # Red channel
        final_image[:, :cols, 1] = gray_img  # Green channel
        final_image[:, :cols, 2] = gray_img  # Blue channel
        final_image[:, cols:, :] = img
        cv.imshow('Exercise 1 (2)', final_image)
        cv.setWindowTitle('Exercise 1 (2)', 'Exercise 1.1 (2): Concenate image') 
        '''    

    if mode == 1:
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv_img = cv.resize(hsv_img, (0,0), fx=0.75, fy=0.75) 
        brightLAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        brightLAB = cv.resize(brightLAB, (0,0), fx=0.75, fy=0.75) 
        img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
        img_yuv = cv.resize(img_yuv, (0,0), fx=0.75, fy=0.75) 
        im_h = cv.hconcat([hsv_img, brightLAB, img_yuv])
        cv.imshow('Exercise 1', im_h)
        cv.setWindowTitle('Exercise 1', 'Exercise 1.2: HSV, LAB, YUV') 

    # source: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    if mode==2:
        
        '''
        th3 = cv.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        ret2,th2 = cv.threshold(gray_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        adthres = cv.hconcat([th3, th2])
        cv.imshow("Exercise 1", adthres)
        cv.setWindowTitle('Exercise 1', 'Adaptive thresholding variants Gaussian and Otsu-Thresholding')    
        '''

        th2 = cv.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        # Otsu's thresholding after Gaussian filtering
        blur = cv.GaussianBlur(gray_img,(5,5),0)
        ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        concat = cv.hconcat([th2, th3])
        cv.imshow("Exercise 1", concat)
        cv.setWindowTitle('Exercise 1', 'Adaptive thresholding variants Gaussian and Otsu') 


    if mode==3:
        edges = cv.Canny(img,100,200)
        cv.imshow("Exercise 1", edges)
        cv.setWindowTitle('Exercise 1', 'Canny Edge Detector') 


#cv.imwrite('OpenCV_concenate.png', comb)
#cv.imwrite('OpenCV_HSV_LAB_YUV.png', im_h)
#cv.imwrite('OpenCV_Adaptive_Threshold_Otsu.png', concat)
#cv.imwrite('OpenCV_Canny_Edge.png', edges)


cv.destroyAllWindows()