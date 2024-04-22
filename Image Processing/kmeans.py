import numpy as np
import cv2
import math
import sys
import warnings

############################################################
#
#                       KMEANS
#
############################################################

# implement distance metric - e.g. squared distances between pixels
def distance(a, b):
    return np.sum((a - b) ** 2)


# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

def update_mean(img, clustermask):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""
    reshaped_array = clustermask.reshape((clustermask.shape[0]*clustermask.shape[1], 1))
    image_reshape = img.reshape((img.shape[0]*img.shape[1],3))
    
    for num in range(numclusters):
        pixel_counter=0
        ch1_sum, ch2_sum, ch3_sum = 0, 0, 0 
        for cluster in range(len(reshaped_array)):         
            if (num==reshaped_array[cluster]):
                pixel_counter+=1
                ch1_sum += image_reshape[cluster][0]
                ch2_sum += image_reshape[cluster][1]
                ch3_sum += image_reshape[cluster][2]
        if pixel_counter>0:
            current_cluster_centers[num][0][0]=ch1_sum/pixel_counter
            current_cluster_centers[num][0][1]=ch2_sum/pixel_counter
            current_cluster_centers[num][0][2]=ch3_sum/pixel_counter
        else: 
            current_cluster_centers[num][0][0]=0
            current_cluster_centers[num][0][1]=0
            current_cluster_centers[num][0][2]=0

 

def assign_to_current_mean(img, result, clustermask):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    overall_dist = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j]
            min_dist = sys.float_info.max
            cluster_id = 0
            for k in range(numclusters):
                dist = distance(pixel, current_cluster_centers[k])
                if dist < min_dist:
                    min_dist = dist
                    cluster_id = k
            clustermask[i, j] = cluster_id
            
            result[i, j] = cluster_colors[cluster_id]
            #result[i, j] = current_cluster_centers[cluster_id]             #change colour to mean
            
            overall_dist += min_dist
    #print('clustermask: ', clustermask)

    return overall_dist



def initialize(img):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    
    for i in range(numclusters):
        rand_x = np.random.randint(0, img.shape[0])
        rand_y = np.random.randint(0, img.shape[1])
        current_cluster_centers[i] = img[rand_x, rand_y]    


    #print('Initialize Cluster Centers: ', current_cluster_centers)


def kmeans(img):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max

    clustermask = np.zeros((h1, w1, 1), np.uint8)

    result = np.zeros((h1, w1, 3), np.uint8)
    
    initialize(img)

    # calculate distance in each cluster ad sum 
    dist = assign_to_current_mean(img, result, clustermask)
    print('Total error after random init without k-means: ', dist)   

    # optimization of the algorithm, check that distances of the initialized pixel are far enough away 
    '''
    distance_control = 2000000000                                                                                           # Opti
    while dist<distance_control:
        print('New init')
        initialize(img)
        dist = assign_to_current_mean(img, result, clustermask)
    '''    
    
    prev_dist = dist
    dist = assign_to_current_mean(img, result, clustermask)
    update_mean(img, clustermask)

    for iteration in range(max_iter):
        prev_dist = dist
        dist = assign_to_current_mean(img, result, clustermask)
        update_mean(img, clustermask)

        change_rate = abs(prev_dist - dist) / prev_dist
        if change_rate < max_change_rate:
            #print('Iteration process stopped after:  ', iteration)
            break
    
    # total intra-cluster variance. Intra-cluster variance is the measure of how much are the points in a given cluster spread
    print('Total error after k-means: ', dist)                    
    return result

# num of cluster
numclusters = 3
# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128], [64, 64, 64]]

# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 1, 3), np.float32)

# load image
imgraw = cv2.imread("graffiti.png")
#scaling_factor = 0.005
scaling_factor = 0.5
imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)


# compare different color spaces and their result for clustering
# YOUR CODE HERE or keep going with loaded RGB colorspace img = imgraw
image = imgraw
h1, w1 = image.shape[:2]

# execute k-means over the image
# it returns a result image where each pixel is color with one of the cluster_colors
# depending on its cluster assignment
res = kmeans(image)


h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1 + w2] = image


cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
title = 'K-Means_'+str(numclusters)+'_clusters'
cv2.setWindowTitle('Color-based Segmentation Kmeans-Clustering', title) 
file_name=title+'.png'

#cv2.imwrite(file_name, vis)
cv2.waitKey(0)
cv2.destroyAllWindows()


