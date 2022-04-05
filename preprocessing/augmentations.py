import random

import numpy as np
import albumentations as A
import preprocessing

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

def CLAHE(image: np.ndarray) -> dict:
    aug = A.CLAHE(p = 1)
    augmented = aug(image = image)
    return augmented

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

    image = preprocessing.pad(img = image)
    mask = preprocessing.pad(img = mask)

    aug = A.Compose([
        A.Crop(x_min = 0, y_min = 0, y_max = int(height*2/3), x_max = int(width*2/3), p = 0.4),
        A.RandomRotate90(p = 0.8),
        A.Transpose(p = 0.5),
        A.RandomBrightnessContrast(p = 0.5, contrast_limit = 0.08, brightness_limit=0.08),
        A.Resize(height = targetSize, width = targetSize),
        A.OneOf([
            A.HorizontalFlip(p = 0.8),
            A.VerticalFlip(p = 0.8)
        ], p = 0.8)])

    augmented = aug(image = image, mask = mask)

    return augmented['image'], augmented['mask']

