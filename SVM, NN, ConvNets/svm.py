import numpy as np
import cv2
import glob
from sklearn import svm
import os

############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################
# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use ~15x15 keypoints on each image with subwindow of 21px (diameter)
def get_class_name(label, int_to_class):
    return int_to_class[label]


def train_svm(img_path):
    # Bild laden
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Fehler beim Laden des Bildes von {img_path}")
    keypoints = []
    keypointSize = 15    
    spacing = 21                       # Hyperparameter to create grid     
    image_with_grid = image.copy()
    for y in range(0, image.shape[0], spacing):
        for x in range(0, image.shape[1], spacing):
            keypoint = cv2.KeyPoint(x, y, keypointSize)
            keypoints.append(keypoint)
    
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    '''
    cv2.imshow(img_path, img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    '''
    print('keypoints: ', len(keypoints))
    keypointsSIFT, descriptor = sift.compute(image, keypoints)
    #print('descriptor: ', descriptor)
    flattened_vector = np.ravel(descriptor)
    #print('flattened_vector: ', len(flattened_vector))
    #flattened_vector = np.ravel(subwindows)
    #print('len subwindows: ', len(subwindows))
    #print('len flattened_vector: ', len(flattened_vector))
    return flattened_vector
    


# Definiere den Pfad zu deinem Trainingsdatensatz
train_data_path = 'images/db/train'
class_to_int = {class_folder: idx for idx, class_folder in enumerate(os.listdir(train_data_path))}
int_to_class = {idx: class_folder for class_folder, idx in class_to_int.items()}

print('class_to_int: ', class_to_int)
print('int_to_class: ', int_to_class)

# Initialisiere leere Listen für Bilder und Labels
X_train = []
y_train = []
sift = cv2.SIFT_create()
num_keypoints=15
#subwindow_diameter=21
all_flattened_vector= []

# Gehe durch jeden Unterordner im Trainingsdatensatz
for class_folder in os.listdir(train_data_path):
    class_path = os.path.join(train_data_path, class_folder)    
    #print('class_path: ', class_path)
    # Ignoriere Dateien, die keine Ordner sind
    if not os.path.isdir(class_path):
        continue    
    label = class_to_int[class_folder]
    #print('label: ', label)    
    # Lese jede Bilddatei im Ordner
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)        
        y_train.append(label)
        #print('img_path: ', img_path)
        X_train.append(train_svm(img_path))
        my_array = np.array(X_train)
        print("Dimension der Liste:", np.shape(my_array))

print('y_train: ', y_train)
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
test_data_path = 'images/db/test'
bilder_dateien = [f for f in os.listdir(test_data_path) if f.lower().endswith('.jpg')]

for bild_datei in bilder_dateien:
    bild_pfad = os.path.join(test_data_path, bild_datei)
    print('bild_pfad: ', bild_pfad)
    image_name = os.path.splitext(os.path.basename(bild_pfad))[0]
    print('image_name: ', image_name)    
    
    test_features = train_svm(bild_pfad)
    predicted_label = svm_classifier.predict([test_features]) 
    predicted_class = get_class_name(predicted_label[0], int_to_class)
    print('Predicted class: ', predicted_class)
    image = cv2.imread(bild_pfad)
    
    cv2.imwrite('./images/db/predictions_svm/'+'image_'+image_name+'_predicted_'+predicted_class+'.jpg', image)
    cv2.imshow('image: '+image_name+', predicted: '+predicted_class, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    

