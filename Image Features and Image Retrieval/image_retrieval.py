import cv2
import glob
import numpy as np
from queue import PriorityQueue
import matplotlib.pyplot as plt

############################################################
#
#              Simple Image Retrieval
#
############################################################


# implement distance function
def distance(a, b):
    return np.linalg.norm(a - b)


def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11
    
    spacing = 12                        # Hyperparameter to create grid 
    
    image_with_grid = image.copy()
    raster = 0

    for y in range(0, image.shape[0], spacing):
        for x in range(0, image.shape[1], spacing):
            keypoint = cv2.KeyPoint(x, y, keypointSize)
            keypoints.append(keypoint)
            '''
            image_with_grid = cv2.drawMarker(image_with_grid, (int(keypoint.pt[0]), int(keypoint.pt[1])), (0, 0, 255),
                                            markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
            raster+=1
            '''
    #image_with_keypoints = cv2.drawKeypoints(image, keypoints, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)     
    #cv2.imshow('Bild mit Raster', image_with_grid)
    #cv2.waitKey(0)
    #cv2.imshow('Bild mit Keypoints', image_with_keypoints)
    #cv2.waitKey(0)
    # please sample the image uniformly in a grid
    # find the keypoint size and number of sample points
    # as hyperparameters
    # YOUR CODE HERE
    return keypoints


############################################################
#       Create Descriptor
############################################################

# 1. preprocessing and load
images = glob.glob('./images/db/train/*/*.jpg')

descriptors = []
# save the images
train_images = []
i = 0
sift = cv2.SIFT_create()

for image_path in images:
    # Read the image using OpenCV    
    image = cv2.imread(image_path)
    train_images.append(image)  
    keypoints = create_keypoints(256, 256)
    keypoints, descriptors_image = sift.compute(image, keypoints)
    descriptors.append(descriptors_image)

############################################################
#       Test the images 
############################################################

images_test = glob.glob('./images/db/test/*.jpg')

for image_path in images_test:
    test_image = cv2.imread(image_path)
    keypoints_test_image = create_keypoints(256, 256)
    keypoints, descriptors_test_image = sift.compute(test_image, keypoints_test_image)

    distances = []

    # Schleife durch alle Deskriptoren in Ihrer Liste und berechnen Sie die Distanzen
    for descriptor in descriptors:
        dist = distance(descriptors_test_image, descriptor)
        distances.append(dist)


    sorted_values_with_indices = sorted(enumerate(distances), key=lambda x: x[1])

    concatenated_image = np.zeros((256, 1280, 3), dtype=np.uint8)
    new_size = 128
    x, y = 0, 0 
    counter = 0 

    for index, value in sorted_values_with_indices:
        resized_image = cv2.resize(train_images[index], (new_size, new_size))
        concatenated_image[y:y+new_size, x:x+new_size, :] = resized_image
        x+=new_size
        if x + new_size >concatenated_image.shape[1]:
            x = 0
            y += new_size

    split_path = image_path.split("/")
    last_word = split_path[-1]

    # Entfernen Sie die ".jpg"-Erweiterung, wenn sie vorhanden ist
    if last_word.endswith(".jpg"):
        last_word = last_word[:-4]

    cv2.imwrite('./images/db/result/'+last_word+'.jpg', concatenated_image)
    cv2.imshow('Query image', test_image)
    cv2.imshow('Best matches', concatenated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


