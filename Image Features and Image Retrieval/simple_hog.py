import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins):
    print('bins: ', bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.title('Histogramm vert own')
    plt.bar(center, hist, align='center', width=width)
    plt.show()

def normalize_histogram(hist):
    hist_sum = np.sum(hist)
    if hist_sum > 0:
        normalized_hist = hist / hist_sum
        return normalized_hist
    return hist


def compute_simple_hog(imgcolor, keypoints):

    # Convert color to grayscale image and extract features in grayscale
    imggray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)

    # Compute x and y gradients (Sobel kernel size 5)
    gradient_x = cv2.Sobel(imggray, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(imggray, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = cv2.magnitude(gradient_x, gradient_y)
    angle = cv2.phase(gradient_x, gradient_y)


    # Compute magnitude and angle of the gradients
    #magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        
        #print('kp.pt: ', kp.pt)
        #print('kp.size: ', kp.size)
        x, y = kp.pt
        keypoint_size = int(kp.size / 2)        
        # print kp.pt, kp.size
        # extract angle in keypoint sub window
        sub_window_magnitude = magnitude[int(y - keypoint_size):int(y + keypoint_size),
                              int(x - keypoint_size):int(x + keypoint_size)]
        #print('sub_window_magnitude: ', sub_window_magnitude)

        # extract gradient magnitude in keypoint subwindow
        sub_window_angle = angle[int(y - keypoint_size):int(y + keypoint_size),
                                int(x - keypoint_size):int(x + keypoint_size)]
        #print('sub_window_angle: ', sub_window_angle)

        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        mask = (sub_window_magnitude > 0)
        hist, bins = np.histogram(sub_window_angle[mask], bins=8, range=(0, 2 * np.pi))

        #(hist, bins) = np.histogram(...)
        normalized_hist = normalize_histogram(hist)

        plot_histogram(normalized_hist, bins)
        
        descr[count] = hist

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
test = cv2.imread('./images/hog_test/vert.jpg')
descriptor = compute_simple_hog(test, keypoints)
