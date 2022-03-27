import random

import numpy as np
import albumentations as A

def padImageAndMask(image: np.ndarray, mask: np.ndarray, min_height: int, min_width: int) -> dict:
    aug = A.PadIfNeeded(min_height = min_height, min_width = min_width, p = 1)
    augmented = aug(image = image, mask = mask)
    return augmented

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


def augment(image: np.ndarray, mask = np.ndarray) -> dict:

    height, width = image.shape

    aug = A.Compose([
        A.OneOf([
            A.RandomSizedCrop(min_max_height = (200, 200), height = height, width = width, p = 0.5),
            A.PadIfNeeded(min_height = height, min_width = width, p = 0.5)
        ], p = 1),
        A.VerticalFlip(p = 0.5),
        A.RandomRotate90(p = 0.5),
        A.Transpose(p = 0.5),
        A.RandomBrightnessContrast(p = 0.5, contrast_limit = 0, brightness_limit=0.1),
        A.OneOf([
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5)
        ], p = 0.8)])

    augmented = aug(image = image, mask = mask)

    return augmented
