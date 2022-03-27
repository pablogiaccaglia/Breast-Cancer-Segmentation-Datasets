# Import stuff
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import albumentations as A
import dicom_utils
import augmentation as aug

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state = None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    shape = image.shape

    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size = pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode = cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode = "constant", cval = 0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode = "constant", cval = 0) * alpha

    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing = 'ij')

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distored_image = map_coordinates(image, indices, order = 1, mode = 'reflect')

    return distored_image.reshape(shape)

def visualizeImageAndMask(image, mask, original_image = None, original_mask = None, cmap = None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize = (8, 8))

        ax[0].imshow(image, cmap = cmap)
        ax[1].imshow(mask, cmap = cmap)
    else:
        f, ax = plt.subplots(2, 2, figsize = (8, 8))

        ax[0, 0].imshow(original_image, cmap = cmap)
        ax[0, 0].set_title('Original image', fontsize = fontsize)

        ax[1, 0].imshow(original_mask, cmap = cmap)
        ax[1, 0].set_title('Original mask', fontsize = fontsize)

        ax[0, 1].imshow(image, cmap = cmap)
        ax[0, 1].set_title('Transformed image', fontsize = fontsize)

        ax[1, 1].imshow(mask, cmap = cmap)
        ax[1, 1].set_title('Transformed mask', fontsize = fontsize)

    plt.show()

# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color = (255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color = (255,))

def applyHistogramEqualization(img, cv2ColorIn: int, cv2ColorOut: int):
    # since equalizeHist equalizes the histogram of a grayscale image, we need this color space conversion
    grayImage = cv2.cvtColor(img, cv2ColorIn)
    grayImageEqualizedHistogram = cv2.equalizeHist(grayImage)

    # convert back the image to the original color space
    img = cv2.cvtColor(grayImageEqualizedHistogram, cv2ColorOut)
    return img

# Load images
im = cv2.imread("lenna.jpeg", -1)

equa = applyHistogramEqualization(im, cv2.COLOR_RGB2GRAY, cv2.COLOR_GRAY2RGB)

# Draw grid lines
draw_grid(im, 50)

# Apply transformation on image
transformedImage = elastic_transform(im, alpha = 2000, sigma = 30, alpha_affine = 20)

# transformedImage2 = applyHistogramEqualization(img = im, cv2ColorIn = cv2.COLOR_RGB2GRAY, cv2ColorOut = cv2.COLOR_GRAY2RGB)

# Display result

image = dicom_utils.getData("/Users/pablo/Desktop/nl2-project/ct.dcm")
mask = dicom_utils.getData("/Users/pablo/Desktop/nl2-project/mask.dcm")


augmented = aug.augment(image = image, mask = mask)

visualizeImageAndMask(image = augmented['image'], mask = augmented['mask'], original_image = image, original_mask = mask, cmap = plt.cm.gray)



