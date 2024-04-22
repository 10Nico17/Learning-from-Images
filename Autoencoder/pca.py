import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


def load_images(path: str, file_ending: str = ".png") -> np.ndarray:
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
        path: Path of directory containing image files that can be assumed to have all the same dimensions
        file_ending: string that image files have to end with, if not->ignore file

    Return:
        images: A 3-D Numpy Array representing multiple images
                Dim 1 = Number of images
                Dim 2 = Height of images
                Dim 3 = Width of images
    """

    images = []

    files = os.listdir(path)
    files.sort()
    for cur in files:
        if not cur.endswith(file_ending):
            continue

        try:
            image = mpl.image.imread(path + cur)
            img_mtx = np.asarray(image, dtype="float64")
            images.append(img_mtx)
        except:
            continue

    return np.array(images)


if __name__ == '__main__':

    print(f'---------- TRAIN ----------')
    # Load images as 3-D Numpy Array.
    images = load_images('./data/train/')
    y, x = images.shape[1:3]
    print(f'Loaded image matrix of shape {images.shape}')

    # Flatten last two dimensions by reshaping the array
    # A matrix
    images = images.reshape(images.shape[0], x * y)
    print(f'Image matrix shape after flattening {images.shape}')



        
    # 1.1 Calculate mean values for each pixel across all images
    # 1.2 Subtract mean values from images to center the data
    # TODO YOUR CODE HERE
    mean_values = np.mean(images, axis=0)
    print('mean_values: ', mean_values.shape)
    centered_images = images - mean_values

    
    # Calculate PCA
    # 2. Compute Eigenvectors of the image data
    #   and find the best linear mapping (eigenbasis)
    #   use the np.linalg.svd with the parameter 'full_matrices=False'
    #   pcs contains the singular vectors ~ eigen vectors
    U, svals, pcs = np.linalg.svd(images, full_matrices=False)
    print(
        f"U shape: {U.shape}, PCS shape: {pcs.shape}, SVALS shape: {svals.shape}, ",
        U.shape, pcs.shape, svals.shape)
    
    '''
    #check if i can build old images matrix
    reconstructed_images = np.dot(U, np.dot(np.diag(svals), pcs))
    '''       


    #Use k=10/75/150 first eigenvectors for reconstruction    
    k = 150

    print('pcs: ', pcs.shape)
    pcs_new = pcs[:k, :]
    print('pcs new: ', pcs_new.shape)

    
    # 4. Load, flatten and center the test images (as in 1.1 and 1.2)
    images_test = load_images('./data/test/')
    print('##################### images_test: ', images_test.shape)
    y, x = images_test.shape[1:3]
    # TODO YOUR CODE HERE - hint: see auto-encode.py
    # images_test_normalized = ...
    images_test_normalized1 = images_test.reshape(images_test.shape[0], x * y)
    images_test_normalized = images_test_normalized1 - mean_values

    #print('mean_values: ', mean_values.shape)
    print('images_test_normalized: ', images_test_normalized.shape)


    # List for reconstructed images to plot it later
    reconstructed_images = []

    
    # 5. Loop through all normalized test images
    errors = []
    
    for i, test_image_normalized in enumerate(images_test_normalized):
        print('######## test_image_normalized: ', test_image_normalized.shape)
        print('pcs.T: ', (pcs_new.T).shape)        
        coeff_test_image = np.dot(test_image_normalized, pcs_new.T)
        print(f'Encoded / compact shape: {coeff_test_image.shape}')

    

    # 5.2 Reconstruct image from coefficient vector and add mean
        reconstructed_image = np.dot(coeff_test_image, pcs_new) + mean_values
        print(reconstructed_image.shape)
        reconstructed_image = reconstructed_image.reshape(images_test[0].shape)
        reconstructed_images.append(reconstructed_image)
    
    
   
    # TODO UNCOMMENT
    # Measure error between loaded original image and reconstructed image
    for i in range(len(reconstructed_images)):
        difference = images_test[i] - reconstructed_images[i]
        error = np.linalg.norm(difference)
        errors.append(error)
        

    # Plot Results
    if len(images_test) != 0 and len(reconstructed_images) != 0:
        plot_img_original = images_test[-1]
        plot_img_reconstructed = reconstructed_images[-1]

        grid = plt.GridSpec(2, 9)

        plt.subplot(grid[0, 0:3])
        plt.imshow(plot_img_original, cmap='Greys_r')
        plt.xlabel('Original person')

        plt.subplot(grid[0, 3:6])
        plt.imshow(plot_img_reconstructed, cmap='Greys_r')
        plt.xlabel('Reconstructed image')

        plt.subplot(grid[0, 6:])
        plt.plot(np.arange(len(images_test)), errors)
        plt.xlabel('Errors all images')

        print("Mean error", np.asarray(errors).mean())

        plt.suptitle("k: "+str(k))  

        plt.savefig("pca_solution_k"+str(k)+".png")
        plt.show()
    else:
        print(
            '#################### Make sure to fill image_test and reconstructed_images lists with images to show.'
        )

