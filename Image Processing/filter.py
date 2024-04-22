import numpy as np
import cv2


def im2double(im):
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.

    :param im:
    :return: normalized image
    """
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def make_gaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return k / np.sum(k)


def convolution_2d(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    # TODO write convolution of arbritrary sized convolution here
    # Hint: you need the kernelsize

    offset = int(kernel.shape[0]/2)
    newimg = np.zeros(img.shape)
       
    for i in range(img.shape[0] - (offset+1)):
        for j in range(img.shape[1] - (offset+1)):
            newimg [i + 1, j + 1] = np.sum(np.multiply(kernel, img[i:i + 3, j:j + 3]))  # x direction
            #gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))  # y direction
            #newimg[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"
    return newimg

if __name__ == "__main__":

    # 1. load image in grayscale
    img = cv2.imread("graffiti.png", cv2.IMREAD_GRAYSCALE)   

    # 2. convert image to 0-1 image (see im2double)
    image = im2double(img)
    #image = img

    gaussian_blur_kernel = make_gaussian(size=3)
    gaussian_image= convolution_2d(image, gaussian_blur_kernel)

    # kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  

    # gradients
    sobel_x = convolution_2d(gaussian_image, sobelmask_x)
    sobel_y = convolution_2d(gaussian_image, sobelmask_y)
    
    # 4. compute magnitude of gradients
    sobel_filtered= np.sqrt(sobel_x ** 2 + sobel_y ** 2)  # calculate the "hypotenuse"
    
    # MOG


    cv2.imshow("gaussian_blur", gaussian_image)
    #cv2.imwrite('filer_gaussian_blur.png', gaussian_image)

    cv2.imshow("sobel_x", sobel_x)
    #cv2.imwrite('filter_sobel_x.png', sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    #cv2.imwrite('filter_sobel_y.png', sobel_y)
    cv2.imshow("Sobel: ", sobel_filtered)
    #cv2.imwrite('filter_sobel_magnitude.png', sobel_filtered)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

        