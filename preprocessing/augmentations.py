import random

import numpy as np
import albumentations as A
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

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

def imageAndMaskHFlip(image: np.ndarray, mask: np.ndarray) -> dict:
    aug = A.HorizontalFlip(p = 1)
    augmented = aug(image = image, mask = mask)
    return augmented

def transposeImageAndMask(image: np.ndarray, mask: np.ndarray) -> dict:
    aug = A.Transpose(p = 1)
    augmented = aug(image = image, mask = mask)
    return augmented

def elasticTransform(image: np.ndarray, mask: np.ndarray) -> dict:
    aug = A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
    random.seed(7)
    augmented = aug(image=image, mask=mask)
    return augmented

def gridDistort(image: np.ndarray, mask: np.ndarray) -> dict:
    aug = A.GridDistortion(p=1)
    random.seed(7)
    augmented = aug(image=image, mask=mask)
    return augmented

def opticalDistort(image: np.ndarray, mask: np.ndarray) -> dict:
    aug = A.OpticalDistortion(distort_limit = 2, shift_limit = 0.5, p = 1)
    random.seed(7)
    augmented = aug(image = image, mask = mask)
    return augmented

def CLAHE(image: np.ndarray) -> np.ndarray:
    aug = A.CLAHE(p = 1)
    image = image.astype(np.uint8)
    augmented = aug(image = image)
    return augmented['image']

def randomRotate90(image: np.ndarray) -> dict:
    aug = A.RandomRotate90(p = 1)
    augmented = aug(image = image)
    return augmented

def randomRotate(image: np.ndarray) -> dict:
    aug = A.Rotate(p = 1, limit = 180)
    augmented = aug(image = image)
    return augmented

def randomBrightness(image: np.ndarray) -> dict:
    aug = A.RandomBrightnessContrast(p=1, contrast_limit = 0)
    augmented = aug(image = image)
    return augmented

def randomContrast(image: np.ndarray) -> dict:
    aug = A.RandomBrightnessContrast(p=1, brightness_limit = 0)
    augmented = aug(image = image)
    return augmented

def augment(image: np.ndarray, mask: np.ndarray, targetSize: int = 256) -> tuple:

    try:
        height, width, c = image.shape
    except:
        height, width = image.shape
    
    image = image.astype(dtype = np.uint8)

    image = pad(img = image)
    mask = pad(img = mask)

    aug = A.Compose([
        A.Crop(x_min = 0, y_min = 0, y_max = int(height*2/3), x_max = int(width*2/3), p = 0.4),
        A.RandomRotate90(p = 0.8),
        A.Transpose(p = 0.3),
        A.RandomBrightnessContrast(p = 0.5, contrast_limit = 0.08, brightness_limit=0.08),
        A.Resize(height = targetSize, width = targetSize),
        A.OneOf([
            A.HorizontalFlip(p = 0.8),
            A.VerticalFlip(p = 0.8)
        ], p = 1)])

    augmented = aug(image = image, mask = mask)

    return augmented['image'], augmented['mask']

def rotateFlipData(image: np.ndarray, mask: np.ndarray) -> tuple:
    aug = A.Compose([
        A.RandomRotate90(p = 0.5),
        A.OneOf([
            A.HorizontalFlip(p = 0.8),
            A.VerticalFlip(p = 0.8)
        ], p = 0.5)])

    augmented = aug(image = image, mask = mask)

    return augmented['image'], augmented['mask']

def pad(img: np.ndarray, logger = None, dtype = np.float64) -> np.ndarray:
    """
    This function pads a given image with black pixels,
    along its shorter side, into a square and returns
    the square image.
    If the image is portrait, black pixels will be
    padded on the right to form a square.
    If the image is landscape, black pixels will be
    padded on the bottom to form a square.
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to pad.
    Returns
    -------
    paddedImage : {numpy.ndarray}
        The padded square image, if padding was required
        and done.
    img : {numpy.ndarray}
        The original image, if no padding was required.
    """

    paddedImg = img

    try:
        channels = None
        if len(img.shape) == 2:
            nRows, nCols = img.shape
        else:
            nRows, nCols, channels = img.shape[0], img.shape[1], img.shape[2]

        # If padding is required...
        if nRows != nCols:
            # Take the longer side as the target shape.
            if nCols < nRows:
                if channels:
                    targetShape = (nRows, nRows, channels)
                else:
                    targetShape = (nRows, nRows)
            elif nRows < nCols:
                if channels:
                    targetShape = (nCols, nCols, channels)
                else:
                    targetShape = (nCols, nCols)

            # pad.
            paddedImg = np.zeros(shape = targetShape, dtype=dtype)
            paddedImg[:nRows, :nCols] = img

    except Exception as e:
        # logger.error(f'Unable to pad!\n{e}')
        print(f"Unable to pad!\n{e}")

    return paddedImg

def padInverted(img: np.ndarray, logger = None, dtype = np.float64) -> np.ndarray:
    """
    This function pads a given image with black pixels,
    along its shorter side, into a square and returns
    the square image.
    If the image is portrait, black pixels will be
    padded on the right to form a square.
    If the image is landscape, black pixels will be
    padded on the bottom to form a square.
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to pad.
    Returns
    -------
    paddedImage : {numpy.ndarray}
        The padded square image, if padding was required
        and done.
    img : {numpy.ndarray}
        The original image, if no padding was required.
    """

    paddedImg = img

    try:
        if len(img.shape) == 2:
            nRows, nCols = img.shape
        else:
            nRows, nCols = img.shape[0], img.shape[1]

        # If padding is required...
        if nRows != nCols:
            # Take the longer side as the target shape.
            if nCols < nRows:
                targetShape = nCols
            elif nRows < nCols:
                targetShape = nRows

            # pad.
            paddedImg = img[:targetShape, :targetShape]


    except Exception as e:
        # logger.error(f'Unable to pad!\n{e}')
        print(f"Unable to pad!\n{e}")

    return paddedImg