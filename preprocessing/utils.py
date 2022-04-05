import os
from itertools import combinations

import cv2
import numpy as np
import tensorflow as tf
from augmentations import augment
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from preprocessing.preprocessing import pad


def imgAugment(logger, self, x_img, y_img):
    """
    Follows the flow:
    1.         datasetPaths()
             _______|_______
            |               |
    2. loadFullImg()  loadMaskImg()
            |_______________|
                    |
    3.          tfParse()
                    |
    4.         imgAugment() *
                    |
    5.        makeTFDataset()

    Apply random image augmentation to full mammogram scans (x_img). Note
    that the same augmentation has to be applied to the masks (y_img), apart
    from the brightness augmentation.

    Parameters
    ----------
    x_img : {numpy.ndarray}
        Full mammogram scan to augment.
    y_img : {numpy.ndarray}
        Corresponding mask of `x_img`.

    Returns
    -------
    x_img : {numpy.ndarray}
        Augmented x_img.
    y_img : {numpy.ndarray}
        Augmented y_img.
    """

    try:

        aug_x, aug_y = augment(image = x_img, mask = y_img)

    except Exception as e:
        # logger.error(f'Unable to imgAugment!\n{e}')
        print(f"Unable to imgAugment!\n{e}")

    return x_img, y_img


def datasetPaths(
        logger,
        full_img_dir,
        mask_img_dir,
        extension,
):
    """
        Follows the flow:
        1.         datasetPaths() *
                 _______|_______
                |               |
        2. loadFullImg()  loadMaskImg()
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()

        Takes in the directories of the folder containing the full mammogram
        scans (x) and the ground truth masks (y) and returns a list of paths to
        individual full scans and masks.

        Parameters
        ----------
        full_img_dir : {str}
            Directory that contains the FULL training images.
        mask_img_dir : {str}
            Directory that contains the MASK training images.
        extension : {str}
            The file extension of the images (e.g. ".png")

        Returns
        -------
        x_paths_list: {list}
            A list of paths to individual FULL images in the training set.
        y_paths_list: {list}
            A list of paths to individual MASK images in the training set.
        """

    x_paths_list = []
    y_paths_list = []

    try:

        # =======================================
        #  1. Get paths of X (full) and y (mask)
        # =======================================

        # Get paths of train images and masks.
        for full in os.listdir(full_img_dir):
            if full.endswith(extension):
                x_paths_list.append(os.path.join(full_img_dir, full))

        for mask in os.listdir(mask_img_dir):
            if mask.endswith(extension):
                y_paths_list.append(os.path.join(mask_img_dir, mask))

        # ** IMPORTANT ** Sort so that FULL and MASK images are in an order
        # that corresponds with each other.
        x_paths_list.sort()
        y_paths_list.sort()

    except Exception as e:
        # logger.error(f'Unable to datasetPaths!\n{e}')
        print((f"Unable to datasetPaths!\n{e}"))

    return x_paths_list, y_paths_list


def loadFullImg(logger, path, dsize, mode: str):
    """
        Follows the flow:
        1.         datasetPaths()
                 _______|_______
                |               |
        2. loadFullImg()* loadMaskImg()
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()

        Takes in the path of a FULL image and loads it as a grayscale image (as
        a numpy array with 3 channels).

        Parameters
        ----------
        path : {str}
            The path of the FULL image.

        Returns
        -------
        full_img : {numpy.ndarray}
            The loaded image with shape = (self.target_size, self.target_size, 3)
        """

    try:
        # =====================================
        #  2a. Read images (full)
        # =====================================
        if not isinstance(path, str):
            path = path.decode()

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        paddedImage = pad(logger = logger, img = img)
        paddedImage = cv2.normalize(
                paddedImage,
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F,
        )

        img = cv2.resize(src = paddedImage, dsize = dsize)

        if mode == "net_data":

            # Min max normalise to [0, 1].
            norm_img = (img - img.min()) / (img.max() - img.min())

        else:
            norm_img = img

        if mode == "net_data":
            # Stack grayscale image to make channels=3.
            full_img = np.stack([norm_img, norm_img, norm_img], axis = -1)

        full_img = img

    except Exception as e:
        # logger.error(f'Unable to loadFullImg!\n{e}')
        print(f"Unable to loadFullImg!\n{e}")

    return full_img


def loadMaskImg(logger, path, dsize, mode: str):
    """
        Follows the flow:
        1.         datasetPaths()
                 _______|_______
                |               |
        2. loadFullImg()  loadMaskImg() *
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()

        Takes in the path of a MASK image and loads it as a grayscale image (as
        a numpy array with 1 channel).

        Parameters
        ----------
        path : {str}
            The path of the MASK image.

        Returns
        -------
        full_img : {numpy.ndarray}
            The loaded image with shape = (self.target_size, self.target_size, 1)
        """

    try:

        # ========================
        #  2b. Read images (mask)
        # ========================
        if not isinstance(path, str):
            path = path.decode()

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        paddedImage = pad(logger = logger, img = img)
        paddedImage = cv2.normalize(
                paddedImage,
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F,
        )

        img = cv2.resize(src = paddedImage, dsize = dsize)

        if mode == "net_data":
            # Min max normalise to [0, 1].
            norm_img = (img - img.min()) / (img.max() - img.min())

        else:
            norm_img = img

        if mode == "net_data":
            # Expand shape to (width, height, 1).
            mask_img = np.expand_dims(norm_img, axis = -1)

        mask_img = img


    except Exception as e:
        # logger.error(f'Unable to loadMaskImg!\n{e}')
        print(f"Unable to loadMaskImg!\n{e}")

    return mask_img


def checkAllDifferent(elems: list):
    if not len(elems):
        return False

    for pair in combinations(elems, 2):
        if np.array_equal(pair[0], pair[1]):
            return False

    return True


def augmentDataset(images: list, masks: list, augmentingFactor: int) -> (list, list):
    augmented_test_images = []
    augmented_test_masks = []
    toCheck = []
    created = []

    for img, msk in zip(images, masks):
        created.clear()
        toCheck.clear()

        while not checkAllDifferent(elems = toCheck):
            created.clear()
            toCheck.clear()
            toCheck.append(img)

            for _ in range(augmentingFactor):
                aug_x, aug_y = augment(image = img, mask = msk)
                created.append((aug_x, aug_y))
                toCheck.append(aug_x)

        for aug_x, aug_y in created:
            augmented_test_images.append(aug_x)
            augmented_test_masks.append(aug_y)

    return augmented_test_images, augmented_test_masks


def loadData(imagesPath: str, masksPath: str, mode: str, augmentationFactor: int = None):
    target_size = (
        256,
        256,
    )

    # Seeding.
    seed = "seed"
    tf.random.set_seed(seed)

    # ====================
    #  Create test images
    # ====================
    # Get paths to individual images.
    test_x, test_y = datasetPaths(
            logger = None,
            full_img_dir = imagesPath,
            mask_img_dir = masksPath,
            extension = ".png",
    )

    #  test_x = [test_x[0], test_x[1], test_x[2]]
    # test_y = [test_y[0], test_y[1], test_x[2]]

    # Read FULL images.
    imgs = [
        loadFullImg(logger = None, path = path, dsize = target_size, mode = mode)
        for path in test_x
    ]

    # Read MASK images.
    masks = [
        loadMaskImg(logger = None, path = path, dsize = target_size, mode = mode)
        for path in test_y
    ]

    if mode == "augment" and augmentationFactor is not None:
        augmented_images, augmented_masks = augmentDataset(images = imgs,
                                                           masks = masks,
                                                           augmentingFactor = augmentationFactor)
        masks = masks + augmented_masks
        imgs = imgs + augmented_images

    elif mode == "net_data":
        masks = np.array(masks, dtype = np.float64)
        imgs = np.array(imgs, dtype = np.float64)

    return imgs, masks


def saveImages(images: list[np.ndarray], dirPath: str, outputFormat: str, baseName: str):
    for i in range(len(images)):
        # Save preprocessed full mammogram image.
        savedFilePath = dirPath + baseName + "__" + str(i + 1).zfill(5) + outputFormat
        cv2.imwrite(savedFilePath, images[i])


def prepareData(imagesPath: str, masksPath: str, mode: str, augmentingFactor: int = None) -> tuple[list, list]:
    imgs, masks = loadData(imagesPath = imagesPath,
                           masksPath = masksPath,
                           augmentationFactor = augmentingFactor, mode = mode)

    if mode == "net_data":
        imgs = imgs.astype('float32')
        mean = np.mean(imgs)  # mean for data centering
        std = np.std(imgs)  # std for data normalization

        imgs -= mean
        imgs /= std
        masks = masks.astype('float32')

        masks /= 255.  # scale masks to [0, 1]
        masks = masks[..., np.newaxis]

    return imgs, masks


def makeFinalDataset(imagesDir: str,
                     masksDir: str,
                     trainingImgDir: str,
                     trainingMaskDir: str,
                     validationImgDir: str,
                     validationMaskDir: str,
                     testingImgDir: str,
                     testingMaskDir: str,
                     augmentingFactor: int,
                     trainRatio: float,
                     validationRatio: float,
                     testRatio: float,
                     outputFormat: str):
    imgs, imgsMasks, = prepareData(imagesPath = imagesDir,
                                   masksPath = masksDir,
                                   augmentingFactor = augmentingFactor, mode = 'augment')

    """    plt.imshow(imgs[0], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[1], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[2], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[3], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[4], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[5], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[6], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[7], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[8], cmap = plt.cm.gray)
        plt.show() """

    xTrain, xTest, yTrain, yTest = train_test_split(imgs, imgsMasks, test_size = 1 - trainRatio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    xVal, xTest, yVal, yTest = train_test_split(xTest, yTest,
                                                test_size = testRatio / (testRatio + validationRatio))

    saveImages(images = xTrain, dirPath = trainingImgDir, outputFormat = outputFormat, baseName = "Training-IMG")
    saveImages(images = yTrain, dirPath = trainingMaskDir, outputFormat = outputFormat, baseName = "Training-MSK")

    saveImages(images = xVal, dirPath = validationImgDir, outputFormat = outputFormat, baseName = "Validation-IMG")
    saveImages(images = yVal, dirPath = validationMaskDir, outputFormat = outputFormat, baseName = "Validation-MSK")

    saveImages(images = xTest, dirPath = testingImgDir, outputFormat = outputFormat, baseName = "Testing-IMG")
    saveImages(images = yTest, dirPath = testingMaskDir, outputFormat = outputFormat, baseName = "Testing-MSK")


def routineMakeFinalDataset():
    imagesPath = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Original-Training-Preprocessed-IMG/"
    masksPath = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Original-Training-Preprocessed-MSK/"
    trainingImgDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Training-Final-IMG/"
    trainingMaskDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Training-Final-MSK/"
    validationImgDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Validation-Final-IMG/"
    validationMaskDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Validation-Final-MSK/"
    testingImgDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Testing-Final-IMG/"
    testingMaskDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Testing-Final-MSK/"

    trainRatio = 0.75
    validationRatio = 0.15
    testRatio = 0.10

    augmentingFactor = 1

    outputFormat = '.png'

    makeFinalDataset(imagesDir = imagesPath,
                     masksDir = masksPath,
                     trainingImgDir = trainingImgDir,
                     trainingMaskDir = trainingMaskDir,
                     validationImgDir = validationImgDir,
                     validationMaskDir = validationMaskDir,
                     testingImgDir = testingImgDir,
                     testingMaskDir = testingMaskDir,
                     augmentingFactor = augmentingFactor,
                     trainRatio = trainRatio,
                     validationRatio = validationRatio,
                     testRatio = testRatio,
                     outputFormat = outputFormat)


def getDatasetsForNet() -> tuple[list, list, list, list, list, list]:
    trainingImgDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Training-Final-IMG/"
    trainingMaskDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Training-Final-MSK/"
    validationImgDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Validation-Final-IMG/"
    validationMaskDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Validation-Final-MSK/"
    testingImgDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Testing-Final-IMG/"
    testingMaskDir = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Testing-Final-MSK/"

    trainingImgs, trainingMasks = prepareData(imagesPath = trainingImgDir,
                                              masksPath = trainingMaskDir,
                                              augmentingFactor = None, mode = 'net_data')
    validationImgs, validationMasks = prepareData(imagesPath = validationImgDir,
                                                  masksPath = validationMaskDir,
                                                  augmentingFactor = None, mode = 'net_data')
    testingImgs, testingMasks = prepareData(imagesPath = testingImgDir,
                                            masksPath = testingMaskDir,
                                            augmentingFactor = None, mode = 'net_data')

    return trainingImgs, trainingMasks, validationImgs, validationMasks, testingImgs, testingMasks


if __name__ == '__main__':
    pass
